"""Selective PVS/alpha-beta search using the incrementally updated NNUE evaluator."""
import numpy as np
from numba import njit
from movegen import *
from nnue_core import *

VALS=np.array([0,100,320,330,500,900,0],dtype=np.int32)

@njit(cache=False)
def hash_board(board,side,rights,ep):
    h=np.uint64(1469598103934665603); prime=np.uint64(1099511628211)
    for s in range(64):
        x=int(board[s])
        if x:
            h ^= np.uint64((x+7)*131+s*17+1); h*=prime
    h ^= np.uint64(1 if side==WHITE else 2); h*=prime
    h ^= np.uint64(rights+17); h*=prime
    if ep>=0: h ^= np.uint64(ep+71); h*=prime
    return h

@njit(cache=False,inline='always')
def is_capture(board,m): return board[m_to(m)]!=0 or m_flag(m)==FLAG_EP

@njit(cache=False,inline='always')
def order_score(board,m,ttm,k1,k2,history,side):
    if m==ttm: return 2000000
    fr=m_from(m); to=m_to(m); pr=m_promo(m)
    attacker=abs(int(board[fr])); victim=abs(int(board[to]))
    if m_flag(m)==FLAG_EP: victim=P
    if victim: return 1000000+int(VALS[victim])*16-int(VALS[attacker])
    if pr: return 900000+int(VALS[pr])
    if m==k1: return 800000
    if m==k2: return 799000
    return int(history[0 if side==WHITE else 1,fr,to])

@njit(cache=False)
def sort_moves(board,moves,n,ttm,k1,k2,history,side):
    for i in range(n):
        bi=i; bs=order_score(board,moves[i],ttm,k1,k2,history,side)
        for j in range(i+1,n):
            sc=order_score(board,moves[j],ttm,k1,k2,history,side)
            if sc>bs: bi=j; bs=sc
        if bi!=i:
            t=moves[i]; moves[i]=moves[bi]; moves[bi]=t

@njit(cache=False,inline='always')
def has_nonpawn_material(board,side):
    for s in range(64):
        pc=int(board[s])
        if pc*side>0 and abs(pc) in (N,B,R,Q): return True
    return False

@njit(cache=False)
def qsearch(board,side,rights,ep,alpha,beta,ply,buffers,acc_stack,ft_w,ft_b,head_w,head_b,stats,node_limit):
    if stats[1]: return 0
    stats[0]+=1
    if stats[0]>=node_limit: stats[1]=1; return 0
    king=find_king(board,side); inc=king>=0 and is_attacked(board,king,-side)
    stand=evaluate_nnue(acc_stack[ply],side,head_w,head_b)
    if not inc:
        if stand>=beta: return beta
        if stand>alpha: alpha=stand
    if ply>=MAX_PLY-2: return stand
    moves=buffers[ply]; n=gen_legal(board,side,rights,ep,moves)
    if n==0: return -MATE+ply if inc else 0
    # Simple tactical ordering: highest victim / promotions first.
    for i in range(n):
        bi=i; bs=-1
        for j in range(i,n):
            m=moves[j]; to=m_to(m); fr=m_from(m); pr=m_promo(m)
            victim=abs(int(board[to])); attacker=abs(int(board[fr]))
            if m_flag(m)==FLAG_EP: victim=P
            sc=(100000+int(VALS[victim])*16-int(VALS[attacker])) if victim else 0
            if pr: sc+=90000+int(VALS[pr])
            if sc>bs: bi=j; bs=sc
        if bi!=i:
            t=moves[i]; moves[i]=moves[bi]; moves[bi]=t
    for i in range(n):
        m=moves[i]
        if not inc and not is_capture(board,m) and m_promo(m)==0: continue
        cap,nr,nep=make_move(board,m,side,rights,ep)
        update_child_after_move(board,m,side,cap,acc_stack[ply],acc_stack[ply+1],ft_w,ft_b)
        v=-qsearch(board,-side,nr,nep,-beta,-alpha,ply+1,buffers,acc_stack,ft_w,ft_b,head_w,head_b,stats,node_limit)
        unmake_move(board,m,side,cap)
        if stats[1]: return 0
        if v>=beta: return beta
        if v>alpha: alpha=v
    return alpha

@njit(cache=False)
def negamax(board,side,rights,ep,depth,alpha,beta,ply,buffers,acc_stack,ft_w,ft_b,head_w,head_b,stats,node_limit,
            tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,killers,history,allow_null):
    if stats[1]: return 0
    stats[0]+=1
    if stats[0]>=node_limit: stats[1]=1; return 0
    if ply>=MAX_PLY-2: return evaluate_nnue(acc_stack[ply],side,head_w,head_b)
    alpha0=alpha; key=hash_board(board,side,rights,ep); idx=int(key & np.uint64(len(tt_keys)-1)); ttm=np.int32(-1)
    if tt_keys[idx]==key:
        ttm=tt_moves[idx]
        if int(tt_depths[idx])>=depth:
            ts=int(tt_scores[idx]); fl=int(tt_flags[idx])
            if fl==0: return ts
            if fl==1 and ts>=beta: return ts
            if fl==2 and ts<=alpha: return ts
    king=find_king(board,side); inc=king>=0 and is_attacked(board,king,-side)
    if depth<=0:
        return qsearch(board,side,rights,ep,alpha,beta,ply,buffers,acc_stack,ft_w,ft_b,head_w,head_b,stats,node_limit)
    # Null-move pruning, avoiding check and pawn-only endings where zugzwang is common.
    if allow_null and depth>=3 and not inc and has_nonpawn_material(board,side):
        red=2+depth//4
        for p in range(2):
            for j in range(NNUE_DIM): acc_stack[ply+1,p,j]=acc_stack[ply,p,j]
        v=-negamax(board,-side,rights,-1,depth-1-red,-beta,-beta+1,ply+1,buffers,acc_stack,ft_w,ft_b,head_w,head_b,stats,node_limit,
                   tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,killers,history,False)
        if stats[1]: return 0
        if v>=beta: return beta
    moves=buffers[ply]; n=gen_legal(board,side,rights,ep,moves)
    if n==0: return -MATE+ply if inc else 0
    sort_moves(board,moves,n,ttm,killers[ply,0],killers[ply,1],history,side)
    best=-MATE; bestm=np.int32(-1); searched=0
    for i in range(n):
        m=moves[i]; quiet=not is_capture(board,m) and m_promo(m)==0
        cap,nr,nep=make_move(board,m,side,rights,ep)
        update_child_after_move(board,m,side,cap,acc_stack[ply],acc_stack[ply+1],ft_w,ft_b)
        ok=find_king(board,-side); gives=ok>=0 and is_attacked(board,ok,side)
        nd=depth-1+(1 if gives and depth<=4 else 0)
        if searched==0:
            v=-negamax(board,-side,nr,nep,nd,-beta,-alpha,ply+1,buffers,acc_stack,ft_w,ft_b,head_w,head_b,stats,node_limit,
                       tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,killers,history,True)
        else:
            red=0
            if quiet and not gives and not inc and depth>=3 and searched>=4:
                red=1+(1 if depth>=6 and searched>=10 else 0)
                if red>nd: red=nd
            v=-negamax(board,-side,nr,nep,nd-red,-alpha-1,-alpha,ply+1,buffers,acc_stack,ft_w,ft_b,head_w,head_b,stats,node_limit,
                       tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,killers,history,True)
            if not stats[1] and v>alpha and red>0:
                v=-negamax(board,-side,nr,nep,nd,-alpha-1,-alpha,ply+1,buffers,acc_stack,ft_w,ft_b,head_w,head_b,stats,node_limit,
                           tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,killers,history,True)
            if not stats[1] and v>alpha and v<beta:
                v=-negamax(board,-side,nr,nep,nd,-beta,-alpha,ply+1,buffers,acc_stack,ft_w,ft_b,head_w,head_b,stats,node_limit,
                           tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,killers,history,True)
        unmake_move(board,m,side,cap)
        if stats[1]: return 0
        searched+=1
        if v>best: best=v; bestm=m
        if v>alpha: alpha=v
        if alpha>=beta:
            if quiet:
                if killers[ply,0]!=m: killers[ply,1]=killers[ply,0]; killers[ply,0]=m
                si=0 if side==WHITE else 1; fr=m_from(m); to=m_to(m)
                history[si,fr,to]=min(1000000,history[si,fr,to]+depth*depth*16)
            break
    tt_keys[idx]=key; tt_scores[idx]=best; tt_depths[idx]=depth; tt_moves[idx]=bestm
    if best<=alpha0: tt_flags[idx]=2
    elif best>=beta: tt_flags[idx]=1
    else: tt_flags[idx]=0
    return best

@njit(cache=False)
def root_search(board,side,rights,ep,depth,buffers,acc_stack,ft_w,ft_b,head_w,head_b,stats,node_limit,
                tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,killers,history,prev_best,a,b):
    moves=buffers[0]; n=gen_legal(board,side,rights,ep,moves)
    if n==0: return np.int32(-1),-MATE
    key=hash_board(board,side,rights,ep); idx=int(key & np.uint64(len(tt_keys)-1))
    ttm=tt_moves[idx] if tt_keys[idx]==key else prev_best
    if prev_best>=0: ttm=prev_best
    sort_moves(board,moves,n,ttm,killers[0,0],killers[0,1],history,side)
    alpha=a; beta=b; best=-MATE; bestm=moves[0]
    for i in range(n):
        m=moves[i]; cap,nr,nep=make_move(board,m,side,rights,ep)
        update_child_after_move(board,m,side,cap,acc_stack[0],acc_stack[1],ft_w,ft_b)
        if i==0:
            v=-negamax(board,-side,nr,nep,depth-1,-beta,-alpha,1,buffers,acc_stack,ft_w,ft_b,head_w,head_b,stats,node_limit,
                       tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,killers,history,True)
        else:
            v=-negamax(board,-side,nr,nep,depth-1,-alpha-1,-alpha,1,buffers,acc_stack,ft_w,ft_b,head_w,head_b,stats,node_limit,
                       tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,killers,history,True)
            if not stats[1] and v>alpha and v<beta:
                v=-negamax(board,-side,nr,nep,depth-1,-beta,-alpha,1,buffers,acc_stack,ft_w,ft_b,head_w,head_b,stats,node_limit,
                           tt_keys,tt_scores,tt_depths,tt_flags,tt_moves,killers,history,True)
        unmake_move(board,m,side,cap)
        if stats[1]: return bestm,best
        if v>best: best=v; bestm=m
        if v>alpha: alpha=v
        if alpha>=beta: break
    return bestm,best
