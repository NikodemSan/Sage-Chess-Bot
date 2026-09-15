"""Parable V5R search core.

Compiler-bounded recursive Numba PVS tuned for the event's one-core, 120+0.5
format and Parable's expensive HalfKP evaluator. The core intentionally keeps the
highest-value selective-search techniques and avoids compiler-heavy machinery.
"""
import math
import numpy as np
from numba import njit
from movegen import *
from nnue_core import *

VALS=np.array([0,100,320,330,500,900,20000],dtype=np.int32)
MAX_HIST=32767
_LMR=np.zeros((MAX_PLY+1,257),dtype=np.int8)
for _d in range(3,MAX_PLY+1):
    for _m in range(3,257):
        r=int(0.55+(math.log(float(_d))*math.log(float(_m)))/2.25)
        if r<1:r=1
        if r>_d-1:r=_d-1
        _LMR[_d,_m]=np.int8(r)

# Zobrist

def _splitmix64(x):
    x=(x+0x9E3779B97F4A7C15)&0xFFFFFFFFFFFFFFFF;z=x
    z=((z^(z>>30))*0xBF58476D1CE4E5B9)&0xFFFFFFFFFFFFFFFF
    z=((z^(z>>27))*0x94D049BB133111EB)&0xFFFFFFFFFFFFFFFF
    return x,(z^(z>>31))&0xFFFFFFFFFFFFFFFF
_seed=0x50415241424C4556
_ZP=np.zeros((12,64),np.uint64)
for _i in range(12):
    for _s in range(64):_seed,_v=_splitmix64(_seed);_ZP[_i,_s]=np.uint64(_v)
_seed,_v=_splitmix64(_seed);_ZSIDE=np.uint64(_v)
_ZCAST=np.zeros(16,np.uint64)
for _i in range(16):_seed,_v=_splitmix64(_seed);_ZCAST[_i]=np.uint64(_v)
_ZEP=np.zeros(8,np.uint64)
for _i in range(8):_seed,_v=_splitmix64(_seed);_ZEP[_i]=np.uint64(_v)
_ZHALF=np.zeros(101,np.uint64)
for _i in range(101):_seed,_v=_splitmix64(_seed);_ZHALF[_i]=np.uint64(_v)

@njit(cache=False)
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

@njit(cache=False)
def hash_after_move(key,board,m,side,rights,ep,captured,nr,nep):
    h=key^_ZSIDE^_ZCAST[rights&15]^_ZCAST[nr&15]
    if ep>=0:h^=_ZEP[ep&7]
    if nep>=0:h^=_ZEP[nep&7]
    fr=m_from(m);to=m_to(m);pr=m_promo(m);fl=m_flag(m)
    moved=int(board[to]);before=side*P if pr else moved
    h^=_ZP[_pidx(before),fr]^_ZP[_pidx(moved),to]
    if captured:
        cs=to-8*side if fl==FLAG_EP else to;h^=_ZP[_pidx(captured),cs]
    if fl==FLAG_CASTLE:
        if to==6:rf,rt=7,5
        elif to==2:rf,rt=0,3
        elif to==62:rf,rt=63,61
        else:rf,rt=56,59
        rook=side*R;h^=_ZP[_pidx(rook),rf]^_ZP[_pidx(rook),rt]
    return h

@njit(cache=False)
def hash_after_null(key,ep):
    h=key^_ZSIDE
    if ep>=0:h^=_ZEP[ep&7]
    return h

@njit(cache=False)
def nonpawn_value(board,side):
    total=0
    for s in range(64):
        pc=int(board[s])
        if pc*side>0:
            pt=abs(pc)
            if pt==N or pt==B or pt==R or pt==Q:total+=int(VALS[pt])
    return total

@njit(cache=False)
def hist_update(history,si,fr,to,bonus):
    if bonus>MAX_HIST:bonus=MAX_HIST
    elif bonus<-MAX_HIST:bonus=-MAX_HIST
    old=int(history[si,fr,to]);old+=bonus-(old*abs(bonus))//MAX_HIST
    if old>MAX_HIST:old=MAX_HIST
    elif old<-MAX_HIST:old=-MAX_HIST
    history[si,fr,to]=old

@njit(cache=False)
def eval_probe(key,keys,scores):
    i=int(key & np.uint64(len(keys)-1))
    if keys[i]==key:return True,int(scores[i])
    return False,0

@njit(cache=False)
def eval_store(key,score,keys,scores):
    i=int(key & np.uint64(len(keys)-1));keys[i]=key
    if score>30000:score=30000
    elif score<-30000:score=-30000
    scores[i]=np.int16(score)

@njit(cache=False)
def tt_load(s,ply):
    if s>MATE-1000:return s-ply
    if s<-MATE+1000:return s+ply
    return s
@njit(cache=False)
def tt_save(s,ply):
    if s>MATE-1000:return s+ply
    if s<-MATE+1000:return s-ply
    return s

@njit(cache=False)
def score_moves(board,moves,scores,n,ttm,k1,k2,history,side):
    si=0 if side==WHITE else 1
    for i in range(n):
        m=moves[i]
        if m==ttm:scores[i]=3000000;continue
        fr=m_from(m);to=m_to(m);pr=m_promo(m);att=abs(int(board[fr]));vic=abs(int(board[to]))
        if m_flag(m)==FLAG_EP:vic=P
        if vic or pr:
            gain=int(VALS[vic])+(int(VALS[pr])-100 if pr else 0)
            scores[i]=1800000+gain*24-int(VALS[att])
        elif m==k1:scores[i]=1600000
        elif m==k2:scores[i]=1500000
        else:scores[i]=200000+int(history[si,fr,to])

@njit(cache=False)
def score_tactical(board,moves,scores,n):
    for i in range(n):
        m=moves[i];fr=m_from(m);to=m_to(m);pr=m_promo(m);att=abs(int(board[fr]));vic=abs(int(board[to]))
        if m_flag(m)==FLAG_EP:vic=P
        gain=int(VALS[vic])+(int(VALS[pr])-100 if pr else 0)
        scores[i]=gain*32-int(VALS[att])

@njit(cache=False)
def pick_best(moves,scores,start,n):
    bi=start;bs=int(scores[start])
    for j in range(start+1,n):
        if int(scores[j])>bs:bi=j;bs=int(scores[j])
    if bi!=start:
        tm=moves[start];moves[start]=moves[bi];moves[bi]=tm
        ts=scores[start];scores[start]=scores[bi];scores[bi]=ts
    return moves[start]

@njit(cache=False)
def repeated(path_keys,ply,key):
    j=ply-2
    while j>=0:
        if path_keys[j]==key:return True
        j-=2
    return False

@njit(cache=False)
def qsearch(board,side,rights,ep,halfmove,alpha,beta,ply,wk,bk,key,buffers,scorebuf,acc,path_keys,stats,node_limit,ec_keys,ec_scores,eval_x,eval_h):
    stats[0]+=1
    if stats[0]>=node_limit:stats[1]=1;return 0
    if ply>=MAX_PLY-2:return evaluate_nnue_scratch(acc[ply],side,board,eval_x,eval_h)
    path_keys[ply]=key
    if repeated(path_keys,ply,key):return 0
    king=wk if side==WHITE else bk;inc=king>=0 and is_attacked(board,king,-side)
    if halfmove>=100:
        if inc and gen_legal_ks(board,side,rights,ep,buffers[ply],king)==0:return -MATE+ply
        return 0
    hit,stand=eval_probe(key,ec_keys,ec_scores)
    if not hit:
        stand=evaluate_nnue_scratch(acc[ply],side,board,eval_x,eval_h);stats[3]+=1;eval_store(key,stand,ec_keys,ec_scores)
    if not inc:
        if stand>=beta:return beta
        if stand>alpha:alpha=stand
    moves=buffers[ply]
    if inc:n=gen_legal_ks(board,side,rights,ep,moves,king);has=True
    else:n,has=gen_tactical_legal_ks(board,side,rights,ep,moves,king)
    if n==0:return (-MATE+ply) if inc else (alpha if has else 0)
    score_tactical(board,moves,scorebuf[ply],n)
    for i in range(n):
        m=int(pick_best(moves,scorebuf[ply],i,n));fr=m_from(m);to=m_to(m);pr=m_promo(m);moving=int(board[fr])
        cap,nr,nep=make_move(board,m,side,rights,ep)
        nwk=wk;nbk=bk
        if abs(moving)==K:
            if side==WHITE:nwk=to
            else:nbk=to
        oppk=nbk if side==WHITE else nwk;gives=oppk>=0 and is_attacked(board,oppk,side)
        if not inc and not gives:
            gain=int(VALS[abs(int(cap))])+(int(VALS[pr])-100 if pr else 0)
            if pr==0 and stand+gain+140<=alpha:
                unmake_move(board,m,side,cap);stats[7]+=1;continue
            if int(VALS[abs(moving)])-gain>=350 and is_attacked(board,to,-side):
                unmake_move(board,m,side,cap);stats[7]+=1;continue
        nkey=hash_after_move(key,board,m,side,rights,ep,cap,nr,nep);nh=0 if abs(moving)==P or cap else halfmove+1
        update_child_after_move_ks(board,m,side,cap,acc[ply],acc[ply+1],nwk,nbk)
        v=-qsearch(board,-side,nr,nep,nh,-beta,-alpha,ply+1,nwk,nbk,nkey,buffers,scorebuf,acc,path_keys,stats,node_limit,ec_keys,ec_scores,eval_x,eval_h)
        unmake_move(board,m,side,cap)
        if stats[1]:return 0
        if v>=beta:return beta
        if v>alpha:alpha=v
    return alpha

@njit(cache=False)
def search_position(board,side,rights,ep,halfmove,depth,alpha,beta,ply,wk,bk,key,allow_null,buffers,scorebuf,acc,path_keys,stats,node_limit,tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,tt_ages,ec_keys,ec_scores,killers,history,generation,eval_x,eval_h):
    stats[0]+=1
    if stats[0]>=node_limit:stats[1]=1;return 0
    if ply>=MAX_PLY-2:return evaluate_nnue_scratch(acc[ply],side,board,eval_x,eval_h)
    path_keys[ply]=key
    if repeated(path_keys,ply,key):return 0
    ma=-MATE+ply;mb=MATE-ply-1
    if alpha<ma:alpha=ma
    if beta>mb:beta=mb
    if alpha>=beta:return alpha
    alpha0=alpha
    king=wk if side==WHITE else bk;inc=king>=0 and is_attacked(board,king,-side)
    if halfmove>=100:
        if inc and gen_legal_ks(board,side,rights,ep,buffers[ply],king)==0:return -MATE+ply
        return 0
    if depth<=0:
        stats[0]-=1
        return qsearch(board,side,rights,ep,halfmove,alpha,beta,ply,wk,bk,key,buffers,scorebuf,acc,path_keys,stats,node_limit,ec_keys,ec_scores,eval_x,eval_h)

    hmc=100 if halfmove>100 else halfmove;tkey=key^_ZHALF[hmc];idx=int(tkey & np.uint64(len(tt_keys)-1));ttm=np.int32(-1)
    if tt_keys[idx]==tkey:
        stats[4]+=1;ttm=tt_moves[idx]
        if int(tt_depths[idx])>=depth:
            ts=tt_load(int(tt_scores[idx]),ply);fl=int(tt_flags[idx]);pv=(beta-alpha)>1
            if fl==0 or (not pv and ((fl==1 and ts>=beta) or (fl==2 and ts<=alpha))):stats[5]+=1;return ts

    hasstatic=False;static=0;hit,sev=eval_probe(key,ec_keys,ec_scores)
    if hit:hasstatic=True;static=sev
    elif not inc and depth<=2:
        static=evaluate_nnue_scratch(acc[ply],side,board,eval_x,eval_h);stats[3]+=1;eval_store(key,static,ec_keys,ec_scores);hasstatic=True
    if beta==alpha+1 and not inc and hasstatic and depth<=2 and nonpawn_value(board,side)>=500:
        margin=90+95*depth
        if static-margin>=beta:stats[7]+=1;return static-margin//3

    if allow_null and depth>=4 and not inc and beta==alpha+1 and nonpawn_value(board,side)>=500 and ((not hasstatic) or static>=beta-90):
        red=2+depth//4
        if red>5:red=5
        for pp in range(2):
            for j in range(NNUE_DIM):acc[ply+1,pp,j]=acc[ply,pp,j]
        v=-search_position(board,-side,rights,-1,halfmove,max(0,depth-1-red),-beta,-beta+1,ply+1,wk,bk,hash_after_null(key,ep),False,buffers,scorebuf,acc,path_keys,stats,node_limit,tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,tt_ages,ec_keys,ec_scores,killers,history,generation,eval_x,eval_h)
        if stats[1]:return 0
        if v>=beta:stats[6]+=1;return v

    moves=buffers[ply];n=gen_legal_ks(board,side,rights,ep,moves,king)
    if n==0:return -MATE+ply if inc else 0
    score_moves(board,moves,scorebuf[ply],n,ttm,killers[ply,0],killers[ply,1],history,side)
    si=0 if side==WHITE else 1;best=-MATE;bestm=np.int32(-1);searched=0
    for i in range(n):
        m=int(pick_best(moves,scorebuf[ply],i,n));fr=m_from(m);to=m_to(m);pr=m_promo(m);fl=m_flag(m);moving=int(board[fr])
        quiet=(board[to]==0 and fl!=FLAG_EP and pr==0);cap,nr,nep=make_move(board,m,side,rights,ep)
        nwk=wk;nbk=bk
        if abs(moving)==K:
            if side==WHITE:nwk=to
            else:nbk=to
        oppk=nbk if side==WHITE else nwk;gives=oppk>=0 and is_attacked(board,oppk,side);h=int(history[si,fr,to]) if quiet else 0
        if searched>0 and quiet and not gives and not inc:
            if depth<=3 and beta==alpha+1 and h<5000 and searched>=5+3*depth*depth:
                unmake_move(board,m,side,cap);stats[7]+=1;continue
            if hasstatic and depth<=2 and static+125+125*depth<=alpha and h<8000:
                unmake_move(board,m,side,cap);stats[7]+=1;continue
        if searched>0 and cap and pr==0 and not gives and not inc and depth<=2:
            gain=int(VALS[abs(int(cap))]);att=int(VALS[abs(moving)])
            if att-gain>=420 and is_attacked(board,to,-side):
                unmake_move(board,m,side,cap);stats[7]+=1;continue
        ext=1 if inc and n==1 and depth<=3 else 0;fd=depth-1+ext
        if fd<0:fd=0
        red=0
        if searched>0 and quiet and not gives and not inc and depth>=3:
            mno=searched+1
            if mno>256:mno=256
            red=int(_LMR[depth if depth<=MAX_PLY else MAX_PLY,mno])
            if beta-alpha>1 and red>0:red-=1
            if h>7000 and red>0:red-=1
            elif h<-7000:red+=1
            if m==killers[ply,0] or m==killers[ply,1]:
                if red>0:red-=1
            if red>fd-1:red=max(0,fd-1)
        nkey=hash_after_move(key,board,m,side,rights,ep,cap,nr,nep);nh=0 if abs(moving)==P or cap else halfmove+1
        update_child_after_move_ks(board,m,side,cap,acc[ply],acc[ply+1],nwk,nbk)
        if searched==0:
            v=-search_position(board,-side,nr,nep,nh,fd,-beta,-alpha,ply+1,nwk,nbk,nkey,True,buffers,scorebuf,acc,path_keys,stats,node_limit,tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,tt_ages,ec_keys,ec_scores,killers,history,generation,eval_x,eval_h)
        else:
            rd=max(0,fd-red)
            v=-search_position(board,-side,nr,nep,nh,rd,-alpha-1,-alpha,ply+1,nwk,nbk,nkey,True,buffers,scorebuf,acc,path_keys,stats,node_limit,tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,tt_ages,ec_keys,ec_scores,killers,history,generation,eval_x,eval_h)
            if not stats[1] and red>0 and v>alpha:
                v=-search_position(board,-side,nr,nep,nh,fd,-alpha-1,-alpha,ply+1,nwk,nbk,nkey,True,buffers,scorebuf,acc,path_keys,stats,node_limit,tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,tt_ages,ec_keys,ec_scores,killers,history,generation,eval_x,eval_h)
            if not stats[1] and v>alpha and v<beta:
                v=-search_position(board,-side,nr,nep,nh,fd,-beta,-alpha,ply+1,nwk,nbk,nkey,True,buffers,scorebuf,acc,path_keys,stats,node_limit,tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,tt_ages,ec_keys,ec_scores,killers,history,generation,eval_x,eval_h)
        unmake_move(board,m,side,cap)
        if stats[1]:return 0
        searched+=1
        if v>best:best=v;bestm=np.int32(m)
        if v>alpha:alpha=v
        if alpha>=beta:
            if quiet:
                bonus=min(16000,48*depth*depth+80*depth)
                if killers[ply,0]!=m:killers[ply,1]=killers[ply,0];killers[ply,0]=m
                hist_update(history,si,fr,to,bonus)
            break
    if bestm>=0:
        flag=0
        if best<=alpha0:flag=2
        elif best>=beta:flag=1
        if tt_keys[idx]==tkey or int(tt_ages[idx])!=generation or int(tt_depths[idx])<=depth+2:
            tt_keys[idx]=tkey;tt_scores[idx]=tt_save(best,ply);tt_depths[idx]=depth;tt_flags[idx]=flag;tt_moves[idx]=bestm;tt_ages[idx]=generation
    return best

@njit(cache=False)
def root_search(board,side,rights,ep,halfmove,depth,buffers,scorebuf,acc,path_keys,stats,node_limit,tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,tt_ages,ec_keys,ec_scores,killers,history,generation,prev_best,a,b,ban_move,eval_x,eval_h):
    wk,bk=find_kings(board);king=wk if side==WHITE else bk;key=hash_board(board,side,rights,ep);path_keys[0]=key
    moves=buffers[0];n=gen_legal_ks(board,side,rights,ep,moves,king)
    if n==0:return np.int32(-1),-MATE
    hmc=100 if halfmove>100 else halfmove;tkey=key^_ZHALF[hmc];idx=int(tkey & np.uint64(len(tt_keys)-1));ttm=tt_moves[idx] if tt_keys[idx]==tkey else prev_best
    if prev_best>=0 and prev_best!=ban_move:ttm=prev_best
    score_moves(board,moves,scorebuf[0],n,ttm,killers[0,0],killers[0,1],history,side)
    alpha=a;alpha0=a;best=-MATE;bestm=np.int32(-1);searched=0
    for i in range(n):
        m=int(pick_best(moves,scorebuf[0],i,n))
        if m==ban_move:continue
        fr=m_from(m);to=m_to(m);moving=int(board[fr]);quiet=(board[to]==0 and m_flag(m)!=FLAG_EP and m_promo(m)==0)
        cap,nr,nep=make_move(board,m,side,rights,ep);nwk=wk;nbk=bk
        if abs(moving)==K:
            if side==WHITE:nwk=to
            else:nbk=to
        gives=(nbk if side==WHITE else nwk)>=0 and is_attacked(board,nbk if side==WHITE else nwk,side)
        nkey=hash_after_move(key,board,m,side,rights,ep,cap,nr,nep);nh=0 if abs(moving)==P or cap else halfmove+1
        update_child_after_move_ks(board,m,side,cap,acc[0],acc[1],nwk,nbk);fd=depth-1
        if searched==0:
            v=-search_position(board,-side,nr,nep,nh,fd,-b,-alpha,1,nwk,nbk,nkey,True,buffers,scorebuf,acc,path_keys,stats,node_limit,tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,tt_ages,ec_keys,ec_scores,killers,history,generation,eval_x,eval_h)
        else:
            rd=fd-1 if quiet and not gives and depth>=6 and searched>=4 else fd
            if rd<0:rd=0
            v=-search_position(board,-side,nr,nep,nh,rd,-alpha-1,-alpha,1,nwk,nbk,nkey,True,buffers,scorebuf,acc,path_keys,stats,node_limit,tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,tt_ages,ec_keys,ec_scores,killers,history,generation,eval_x,eval_h)
            if not stats[1] and rd!=fd and v>alpha:
                v=-search_position(board,-side,nr,nep,nh,fd,-alpha-1,-alpha,1,nwk,nbk,nkey,True,buffers,scorebuf,acc,path_keys,stats,node_limit,tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,tt_ages,ec_keys,ec_scores,killers,history,generation,eval_x,eval_h)
            if not stats[1] and v>alpha and v<b:
                v=-search_position(board,-side,nr,nep,nh,fd,-b,-alpha,1,nwk,nbk,nkey,True,buffers,scorebuf,acc,path_keys,stats,node_limit,tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,tt_ages,ec_keys,ec_scores,killers,history,generation,eval_x,eval_h)
        unmake_move(board,m,side,cap)
        if stats[1]:return bestm,best
        searched+=1
        if v>best:best=v;bestm=np.int32(m)
        if v>alpha:alpha=v
        if alpha>=b:break
    if bestm>=0:
        flag=0
        if best<=alpha0:flag=2
        elif best>=b:flag=1
        if tt_keys[idx]==tkey or int(tt_ages[idx])!=generation or int(tt_depths[idx])<=depth+2:
            tt_keys[idx]=tkey;tt_scores[idx]=tt_save(best,0);tt_depths[idx]=depth;tt_flags[idx]=flag;tt_moves[idx]=bestm;tt_ages[idx]=generation
    return bestm,best
