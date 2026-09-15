"""Parable V4.0 stack search.

A non-recursive PVS/alpha-beta search designed around the event constraints:
* one CPU core
* 120s + 0.5s increment
* 90s cold initialization budget
* NNUE is the static evaluator

The search deliberately avoids full SEE and expensive capture pre-analysis. Tactical
moves are searched directly so pruning overhead cannot dominate the search.
"""
import numpy as np
from numba import njit
from movegen import *
from nnue_core import *

VALS=np.array([0,100,320,330,500,900,0],dtype=np.int32)

# -------------------------- incremental Zobrist hash --------------------------
def _splitmix64(x):
    x=(x+0x9E3779B97F4A7C15)&0xFFFFFFFFFFFFFFFF
    z=x
    z=((z^(z>>30))*0xBF58476D1CE4E5B9)&0xFFFFFFFFFFFFFFFF
    z=((z^(z>>27))*0x94D049BB133111EB)&0xFFFFFFFFFFFFFFFF
    return x,(z^(z>>31))&0xFFFFFFFFFFFFFFFF

_seed=0x50415241424C4556
_ZP=np.zeros((12,64),np.uint64)
for _i in range(12):
    for _s in range(64):
        _seed,_v=_splitmix64(_seed);_ZP[_i,_s]=np.uint64(_v)
_seed,_v=_splitmix64(_seed);_ZSIDE=np.uint64(_v)
_ZCAST=np.zeros(16,np.uint64)
for _i in range(16):_seed,_v=_splitmix64(_seed);_ZCAST[_i]=np.uint64(_v)
_ZEP=np.zeros(8,np.uint64)
for _i in range(8):_seed,_v=_splitmix64(_seed);_ZEP[_i]=np.uint64(_v)
_ZHALF=np.zeros(101,np.uint64)
for _i in range(101):_seed,_v=_splitmix64(_seed);_ZHALF[_i]=np.uint64(_v)

@njit(cache=False,inline='always')
def _pidx(pc):
    pt=abs(int(pc))-1
    return pt if pc>0 else pt+6

@njit(cache=False)
def hash_board(board,side,rights,ep):
    h=np.uint64(0)
    for s in range(64):
        pc=int(board[s])
        if pc:h^=_ZP[_pidx(pc),s]
    if side==BLACK:h^=_ZSIDE
    h^=_ZCAST[rights&15]
    if ep>=0:h^=_ZEP[ep&7]
    return h

@njit(cache=False,inline='always')
def hash_after_move(key,board_after,m,side,rights,ep,captured,nr,nep):
    h=key^_ZSIDE^_ZCAST[rights&15]^_ZCAST[nr&15]
    if ep>=0:h^=_ZEP[ep&7]
    if nep>=0:h^=_ZEP[nep&7]
    fr=m_from(m);to=m_to(m);pr=m_promo(m);fl=m_flag(m)
    moved_after=int(board_after[to]);moving_before=side*P if pr else moved_after
    h^=_ZP[_pidx(moving_before),fr]^_ZP[_pidx(moved_after),to]
    if captured!=0:
        cs=to-8*side if fl==FLAG_EP else to
        h^=_ZP[_pidx(captured),cs]
    if fl==FLAG_CASTLE:
        if to==6:rf,rt=7,5
        elif to==2:rf,rt=0,3
        elif to==62:rf,rt=63,61
        else:rf,rt=56,59
        rook=side*R
        h^=_ZP[_pidx(rook),rf]^_ZP[_pidx(rook),rt]
    return h

@njit(cache=False,inline='always')
def hash_after_null(key,ep):
    h=key^_ZSIDE
    if ep>=0:h^=_ZEP[ep&7]
    return h

@njit(cache=False,inline='always')
def has_nonpawn_material(board,side):
    for s in range(64):
        pc=int(board[s])
        if pc*side>0:
            pt=abs(pc)
            if pt==N or pt==B or pt==R or pt==Q:return True
    return False

# ------------------------------- move ordering -------------------------------
@njit(cache=False,inline='always')
def is_capture(board,m):return board[m_to(m)]!=0 or m_flag(m)==FLAG_EP

@njit(cache=False,inline='always')
def capture_gain(board,m):
    to=m_to(m);pr=m_promo(m);vic=abs(int(board[to]))
    if m_flag(m)==FLAG_EP:vic=P
    gain=int(VALS[vic])
    if pr:gain+=int(VALS[pr])-100
    return gain

@njit(cache=False,inline='always')
def order_score(board,m,ttm,k1,k2,history,side):
    if m==ttm:return 2000000
    fr=m_from(m);to=m_to(m);pr=m_promo(m);att=abs(int(board[fr]));vic=abs(int(board[to]))
    if m_flag(m)==FLAG_EP:vic=P
    if vic:return 1000000+int(VALS[vic])*16-int(VALS[att])
    if pr:return 900000+int(VALS[pr])
    if m==k1:return 800000
    if m==k2:return 799000
    return int(history[0 if side==WHITE else 1,fr,to])

@njit(cache=False)
def sort_moves_cached(board,moves,scores,n,ttm,k1,k2,history,side):
    for i in range(n):scores[i]=order_score(board,moves[i],ttm,k1,k2,history,side)
    for i in range(1,n):
        m=moves[i];sc=scores[i];j=i-1
        while j>=0 and scores[j]<sc:
            moves[j+1]=moves[j];scores[j+1]=scores[j];j-=1
        moves[j+1]=m;scores[j+1]=sc

@njit(cache=False)
def sort_tactical(board,moves,scores,n):
    for i in range(n):
        m=moves[i];fr=m_from(m);to=m_to(m);pr=m_promo(m)
        vic=abs(int(board[to]));att=abs(int(board[fr]))
        if m_flag(m)==FLAG_EP:vic=P
        sc=int(VALS[vic])*16-int(VALS[att]) if vic else 0
        if pr:sc+=10000+int(VALS[pr])
        scores[i]=sc
    for i in range(1,n):
        m=moves[i];sc=scores[i];j=i-1
        while j>=0 and scores[j]<sc:
            moves[j+1]=moves[j];scores[j+1]=scores[j];j-=1
        moves[j+1]=m;scores[j+1]=sc

@njit(cache=False,inline='always')
def tt_load_score(s,ply):
    if s>MATE-1000:return s-ply
    if s<-MATE+1000:return s+ply
    return s

@njit(cache=False,inline='always')
def tt_store_score(s,ply):
    if s>MATE-1000:return s+ply
    if s<-MATE+1000:return s-ply
    return s

# --------------------------- explicit-stack search ---------------------------
@njit(cache=False)
def search_position(board,side,rights,ep,halfmove,depth,alpha,beta,base_ply,wk,bk,key,root_key,
                    buffers,scorebuf,acc_stack,stats,node_limit,
                    tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,killers,history):
    """Search one position with an explicit DFS stack; board is restored before return."""
    L=MAX_PLY
    st_side=np.empty(L,np.int8);st_rights=np.empty(L,np.int16);st_ep=np.empty(L,np.int16)
    st_half=np.empty(L,np.int16);st_depth=np.empty(L,np.int16);st_alpha=np.empty(L,np.int32);st_beta=np.empty(L,np.int32)
    st_alpha0=np.empty(L,np.int32);st_wk=np.empty(L,np.int16);st_bk=np.empty(L,np.int16);st_key=np.empty(L,np.uint64)
    st_stage=np.zeros(L,np.int8);st_n=np.zeros(L,np.int16);st_i=np.zeros(L,np.int16);st_searched=np.zeros(L,np.int16)
    st_best=np.empty(L,np.int32);st_bestm=np.empty(L,np.int32);st_inc=np.zeros(L,np.int8);st_q=np.zeros(L,np.int8);st_static=np.zeros(L,np.int32)
    st_allow_null=np.ones(L,np.int8);st_null_done=np.zeros(L,np.int8)
    st_ttidx=np.empty(L,np.int32);st_ttm=np.empty(L,np.int32);st_ret=np.empty(L,np.int32)
    cur_move=np.empty(L,np.int32);cur_quiet=np.zeros(L,np.int8);cur_red=np.zeros(L,np.int8);cur_full_depth=np.zeros(L,np.int16);cur_phase=np.zeros(L,np.int8)
    edge_move=np.full(L,-1,np.int32);edge_cap=np.zeros(L,np.int8)

    sp=0
    st_side[0]=side;st_rights[0]=rights;st_ep[0]=ep;st_half[0]=halfmove;st_depth[0]=depth
    st_alpha[0]=alpha;st_beta[0]=beta;st_wk[0]=wk;st_bk[0]=bk;st_key[0]=key;st_allow_null[0]=1;st_null_done[0]=0

    while True:
        # Centralized node return/unwind path so node-limit abort always restores board.
        if st_stage[sp]==3:
            rv=int(st_ret[sp])
            if sp==0:return rv
            em=int(edge_move[sp])
            if em>=0:unmake_move(board,em,-int(st_side[sp]),int(edge_cap[sp]))
            sp-=1
            if stats[1]:
                st_ret[sp]=0;st_stage[sp]=3;continue
            # Child score is from child's POV.
            v=-rv
            # Parent processes it in stage 2 via st_ret[parent].
            st_ret[sp]=v
            continue

        if st_stage[sp]==0:
            stats[0]+=1
            if stats[0]>=node_limit:
                stats[1]=1;st_ret[sp]=0;st_stage[sp]=3;continue
            ply=base_ply+sp
            if ply>=MAX_PLY-2:
                st_ret[sp]=evaluate_nnue(acc_stack[ply],int(st_side[sp]),board);st_stage[sp]=3;continue
            if int(st_half[sp])>=100:
                rk=int(st_wk[sp]) if int(st_side[sp])==WHITE else int(st_bk[sp])
                if rk>=0 and is_attacked(board,rk,-int(st_side[sp])):
                    rm=buffers[ply];rn=gen_legal_ks(board,int(st_side[sp]),int(st_rights[sp]),int(st_ep[sp]),rm,rk)
                    if rn==0:
                        st_ret[sp]=-MATE+ply;st_stage[sp]=3;continue
                st_ret[sp]=0;st_stage[sp]=3;continue
            # Cheap cycle detection: same-side repetition on current search path.
            rep=False;j=sp-2
            while j>=0:
                if st_key[j]==st_key[sp]:rep=True;break
                j-=2
            if not rep and (sp&1)==1 and st_key[sp]==root_key:rep=True
            if rep:
                st_ret[sp]=0;st_stage[sp]=3;continue

            sd=int(st_depth[sp]);sa=int(st_alpha[sp]);sb=int(st_beta[sp]);skey=st_key[sp]
            st_alpha0[sp]=sa
            king=int(st_wk[sp]) if int(st_side[sp])==WHITE else int(st_bk[sp])
            inc=king>=0 and is_attacked(board,king,-int(st_side[sp]));st_inc[sp]=1 if inc else 0

            if sd>0:
                hmc=int(st_half[sp]);hmc=100 if hmc>100 else hmc
                tkey=skey^_ZHALF[hmc];idx=int(tkey & np.uint64(len(tt_keys)-1));st_ttidx[sp]=idx;ttm=np.int32(-1)
                if tt_keys[idx]==tkey:
                    ttm=tt_moves[idx]
                    if int(tt_depths[idx])>=sd:
                        ts=tt_load_score(int(tt_scores[idx]),ply);fl=int(tt_flags[idx])
                        if fl==0 or (fl==1 and ts>=sb) or (fl==2 and ts<=sa):
                            st_ret[sp]=ts;st_stage[sp]=3;continue
                st_ttm[sp]=ttm
                if st_allow_null[sp] and not st_null_done[sp] and sd>=4 and not inc and sb==sa+1 and has_nonpawn_material(board,int(st_side[sp])):
                    red=2+(1 if sd>=7 else 0)
                    for pp in range(2):
                        for jj in range(NNUE_DIM):acc_stack[ply+1,pp,jj]=acc_stack[ply,pp,jj]
                    edge_move[sp+1]=-1;edge_cap[sp+1]=0;st_stage[sp]=4
                    sp+=1
                    st_side[sp]=-int(st_side[sp-1]);st_rights[sp]=int(st_rights[sp-1]);st_ep[sp]=-1;st_half[sp]=int(st_half[sp-1])
                    st_depth[sp]=sd-1-red;st_alpha[sp]=-sb;st_beta[sp]=-sb+1;st_wk[sp]=int(st_wk[sp-1]);st_bk[sp]=int(st_bk[sp-1])
                    st_key[sp]=hash_after_null(st_key[sp-1],int(st_ep[sp-1]));st_allow_null[sp]=0;st_null_done[sp]=0;st_stage[sp]=0
                    continue
                moves=buffers[ply];n=gen_legal_ks(board,int(st_side[sp]),int(st_rights[sp]),int(st_ep[sp]),moves,king)
                if n==0:
                    st_ret[sp]=-MATE+ply if inc else 0;st_stage[sp]=3;continue
                sort_moves_cached(board,moves,scorebuf[ply],n,ttm,killers[ply,0],killers[ply,1],history,int(st_side[sp]))
                st_n[sp]=n;st_i[sp]=0;st_searched[sp]=0;st_best[sp]=-MATE;st_bestm[sp]=-1;st_q[sp]=0;st_stage[sp]=1
                continue

            # Quiescence node.
            stand=evaluate_nnue(acc_stack[ply],int(st_side[sp]),board);st_static[sp]=stand
            if not inc:
                if stand>=sb:
                    st_ret[sp]=sb;st_stage[sp]=3;continue
                if stand>sa:sa=stand;st_alpha[sp]=sa
            moves=buffers[ply]
            if inc:
                n=gen_legal_ks(board,int(st_side[sp]),int(st_rights[sp]),int(st_ep[sp]),moves,king);has_legal=(n>0)
            else:
                n,has_legal=gen_tactical_legal_ks(board,int(st_side[sp]),int(st_rights[sp]),int(st_ep[sp]),moves,king)
            if n==0:
                st_ret[sp]=(-MATE+ply if inc else (sa if has_legal else 0));st_stage[sp]=3;continue
            sort_tactical(board,moves,scorebuf[ply],n)
            st_n[sp]=n;st_i[sp]=0;st_searched[sp]=0;st_best[sp]=sa;st_bestm[sp]=-1;st_q[sp]=1;st_stage[sp]=1
            continue

        if st_stage[sp]==1:
            ply=base_ply+sp
            if int(st_i[sp])>=int(st_n[sp]):
                rv=int(st_best[sp])
                if not st_q[sp]:
                    idx=int(st_ttidx[sp]);d=int(st_depth[sp]);k=st_key[sp];hmc=int(st_half[sp]);hmc=100 if hmc>100 else hmc;tkey=k^_ZHALF[hmc]
                    # Depth-preferred direct replacement: protect much deeper collisions.
                    if tt_keys[idx]==tkey or int(tt_depths[idx])<=d+2:
                        tt_keys[idx]=tkey;tt_scores[idx]=tt_store_score(rv,ply);tt_depths[idx]=d;tt_moves[idx]=st_bestm[sp]
                        a0=int(st_alpha0[sp]);bb=int(st_beta[sp])
                        if rv<=a0:tt_flags[idx]=2
                        elif rv>=bb:tt_flags[idx]=1
                        else:tt_flags[idx]=0
                st_ret[sp]=rv;st_stage[sp]=3;continue

            mi=int(st_i[sp]);st_i[sp]=mi+1;m=int(buffers[ply,mi])
            pside=int(st_side[sp]);prights=int(st_rights[sp]);pep=int(st_ep[sp]);pkey=st_key[sp]
            fr=m_from(m);moving=int(board[fr]);cap,nr,nep=make_move(board,m,pside,prights,pep)
            nwk=int(st_wk[sp]);nbk=int(st_bk[sp])
            if abs(moving)==K:
                if pside==WHITE:nwk=m_to(m)
                else:nbk=m_to(m)
            nkey=hash_after_move(pkey,board,m,pside,prights,pep,cap,nr,nep)
            nh=0 if abs(moving)==P or cap!=0 else int(st_half[sp])+1
            update_child_after_move_ks(board,m,pside,cap,acc_stack[ply],acc_stack[ply+1],nwk,nbk)

            cur_move[sp]=m;edge_move[sp+1]=m;edge_cap[sp+1]=cap
            quiet=(cap==0 and m_flag(m)!=FLAG_EP and m_promo(m)==0)
            cur_quiet[sp]=1 if quiet else 0;cur_phase[sp]=0
            if st_q[sp]:
                child_depth=0;cur_red[sp]=0;cur_full_depth[sp]=0
                ca=-int(st_beta[sp]);cb=-int(st_alpha[sp])
            else:
                ok=nbk if pside==WHITE else nwk
                gives=ok>=0 and is_attacked(board,ok,pside)
                nd=int(st_depth[sp])-1+(1 if gives and not st_inc[sp] and int(st_depth[sp])<=3 else 0)
                cur_full_depth[sp]=nd;red=0
                searched=int(st_searched[sp])
                if searched>0 and quiet and not gives and not st_inc[sp] and int(st_depth[sp])>=3 and searched>=4:
                    red=1
                    if int(st_depth[sp])>=6 and searched>=8:red+=1
                    if int(st_depth[sp])>=9 and searched>=16:red+=1
                    h=int(history[0 if pside==WHITE else 1,fr,m_to(m)])
                    if h>12000 and red>1:red-=1
                    if red>nd:red=nd
                cur_red[sp]=red;child_depth=nd-red
                if searched==0:ca=-int(st_beta[sp]);cb=-int(st_alpha[sp])
                else:ca=-int(st_alpha[sp])-1;cb=-int(st_alpha[sp])

            st_stage[sp]=2
            sp+=1
            st_side[sp]=-pside;st_rights[sp]=nr;st_ep[sp]=nep;st_half[sp]=nh;st_depth[sp]=child_depth
            st_alpha[sp]=ca;st_beta[sp]=cb;st_wk[sp]=nwk;st_bk[sp]=nbk;st_key[sp]=nkey;st_allow_null[sp]=1;st_null_done[sp]=0;st_stage[sp]=0
            continue

        # stage 4: null search returned. If it fails low, re-enter once with null disabled.
        if st_stage[sp]==4:
            if stats[1]:st_ret[sp]=0;st_stage[sp]=3;continue
            if int(st_ret[sp])>=int(st_beta[sp]):
                st_ret[sp]=int(st_beta[sp]);st_stage[sp]=3;continue
            st_null_done[sp]=1;stats[0]-=1;st_stage[sp]=0;continue

        # stage 2: a child has returned and has already been unmade.
        if st_stage[sp]==2:
            if stats[1]:st_ret[sp]=0;st_stage[sp]=3;continue
            v=int(st_ret[sp]);m=int(cur_move[sp]);ply=base_ply+sp
            if not st_q[sp] and int(st_searched[sp])>0:
                phase=int(cur_phase[sp]);red=int(cur_red[sp]);a=int(st_alpha[sp]);b=int(st_beta[sp])
                need=False;ca=0;cb=0;cd=int(cur_full_depth[sp])
                if phase==0 and red>0 and v>a:
                    cur_phase[sp]=1;ca=-a-1;cb=-a;need=True
                elif phase<=1 and v>a and v<b:
                    cur_phase[sp]=2;ca=-b;cb=-a;need=True
                if need:
                    pside=int(st_side[sp]);prights=int(st_rights[sp]);pep=int(st_ep[sp]);pkey=st_key[sp]
                    fr=m_from(m);moving=int(board[fr]);cap,nr,nep=make_move(board,m,pside,prights,pep)
                    nwk=int(st_wk[sp]);nbk=int(st_bk[sp])
                    if abs(moving)==K:
                        if pside==WHITE:nwk=m_to(m)
                        else:nbk=m_to(m)
                    nkey=hash_after_move(pkey,board,m,pside,prights,pep,cap,nr,nep)
                    nh=0 if abs(moving)==P or cap!=0 else int(st_half[sp])+1
                    update_child_after_move_ks(board,m,pside,cap,acc_stack[ply],acc_stack[ply+1],nwk,nbk)
                    edge_move[sp+1]=m;edge_cap[sp+1]=cap
                    sp+=1
                    st_side[sp]=-pside;st_rights[sp]=nr;st_ep[sp]=nep;st_half[sp]=nh;st_depth[sp]=cd
                    st_alpha[sp]=ca;st_beta[sp]=cb;st_wk[sp]=nwk;st_bk[sp]=nbk;st_key[sp]=nkey;st_allow_null[sp]=1;st_null_done[sp]=0;st_stage[sp]=0
                    continue

            # Finalize this move.
            st_searched[sp]=int(st_searched[sp])+1
            if v>int(st_best[sp]):st_best[sp]=v;st_bestm[sp]=m
            if v>int(st_alpha[sp]):st_alpha[sp]=v
            if int(st_alpha[sp])>=int(st_beta[sp]):
                if not st_q[sp] and cur_quiet[sp]:
                    if killers[ply,0]!=m:killers[ply,1]=killers[ply,0];killers[ply,0]=m
                    si=0 if int(st_side[sp])==WHITE else 1;fr=m_from(m);to=m_to(m);d=int(st_depth[sp])
                    history[si,fr,to]=min(1000000,history[si,fr,to]+d*d*12)
                if not st_q[sp]:
                    idx=int(st_ttidx[sp]);d=int(st_depth[sp]);k=st_key[sp];rv=int(st_alpha[sp]);hmc=int(st_half[sp]);hmc=100 if hmc>100 else hmc;tkey=k^_ZHALF[hmc]
                    if tt_keys[idx]==tkey or int(tt_depths[idx])<=d+2:
                        tt_keys[idx]=tkey;tt_scores[idx]=tt_store_score(rv,ply);tt_depths[idx]=d;tt_moves[idx]=m;tt_flags[idx]=1
                st_ret[sp]=int(st_alpha[sp]);st_stage[sp]=3;continue
            st_stage[sp]=1
            continue

    return 0

@njit(cache=False)
def root_search(board,side,rights,ep,halfmove,depth,buffers,scorebuf,acc_stack,stats,node_limit,
                tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,killers,history,prev_best,a,b,ban_move):
    wk,bk=find_kings(board);king=wk if side==WHITE else bk
    key=hash_board(board,side,rights,ep)
    moves=buffers[0];n=gen_legal_ks(board,side,rights,ep,moves,king)
    if n==0:return np.int32(-1),-MATE
    hmc=100 if halfmove>100 else halfmove;tkey=key^_ZHALF[hmc]
    idx=int(tkey & np.uint64(len(tt_keys)-1));ttm=tt_moves[idx] if tt_keys[idx]==tkey else prev_best
    if prev_best>=0 and prev_best!=ban_move:ttm=prev_best
    sort_moves_cached(board,moves,scorebuf[0],n,ttm,killers[0,0],killers[0,1],history,side)
    alpha=a;beta=b;best=-MATE;bestm=np.int32(-1);searched=0
    for i in range(n):
        m=moves[i]
        if m==ban_move:continue
        fr=m_from(m);moving=int(board[fr]);cap,nr,nep=make_move(board,m,side,rights,ep)
        nwk=wk;nbk=bk
        if abs(moving)==K:
            if side==WHITE:nwk=m_to(m)
            else:nbk=m_to(m)
        nkey=hash_after_move(key,board,m,side,rights,ep,cap,nr,nep)
        nh=0 if abs(moving)==P or cap!=0 else halfmove+1
        update_child_after_move_ks(board,m,side,cap,acc_stack[0],acc_stack[1],nwk,nbk)
        if searched==0:
            v=-search_position(board,-side,nr,nep,nh,depth-1,-beta,-alpha,1,nwk,nbk,nkey,key,buffers,scorebuf,acc_stack,stats,node_limit,tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,killers,history)
        else:
            v=-search_position(board,-side,nr,nep,nh,depth-1,-alpha-1,-alpha,1,nwk,nbk,nkey,key,buffers,scorebuf,acc_stack,stats,node_limit,tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,killers,history)
            if not stats[1] and v>alpha and v<beta:
                v=-search_position(board,-side,nr,nep,nh,depth-1,-beta,-alpha,1,nwk,nbk,nkey,key,buffers,scorebuf,acc_stack,stats,node_limit,tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,killers,history)
        unmake_move(board,m,side,cap)
        if stats[1]:return bestm,best
        searched+=1
        if v>best:best=v;bestm=m
        if v>alpha:alpha=v
        if alpha>=beta:break
    return bestm,best
