"""Original team NNUE candidate B Mirrored HalfKP-256 -> 32 -> 32 -> 1."""
import os
import numpy as np
from numba import njit
from movegen import WHITE,BLACK,P,N,B,R,Q,K,FLAG_EP,FLAG_CASTLE,m_from,m_to,m_promo,m_flag,find_king
NNUE_DIM=256;NNUE_FEATURES=20480;FT_SCALE=256;HEAD_SCALE=1024;CP_SCALE=600;H1_DIV=256;H2_DIV=256;ACT_MAX=256
MIRRORED=True;DUAL_ACT=False;USE_BUCKETS=False
_WPATH=os.path.join(os.path.dirname(__file__),'weights','nnue_weights.npz');_z=np.load(_WPATH,allow_pickle=False)
FT_W=np.ascontiguousarray(_z['ft_w'],dtype=np.int16);FT_B=np.ascontiguousarray(_z['ft_b'],dtype=np.int32);H1_W=np.ascontiguousarray(_z['h1_w'],dtype=np.int16);H1_B=np.ascontiguousarray(_z['h1_b'],dtype=np.int64);H2_W=np.ascontiguousarray(_z['h2_w'],dtype=np.int16);H2_B=np.ascontiguousarray(_z['h2_b'],dtype=np.int64);OUT_W=np.ascontiguousarray(_z['out_w'],dtype=np.int16);OUT_B=np.ascontiguousarray(_z['out_b'],dtype=np.int64);_z.close()
@njit(cache=False,inline='always')
def canon_sq(sq,persp):return sq if persp==WHITE else (sq^63)
@njit(cache=False,inline='always')
def feature_index(king_sq,piece_sq,pc,persp):
    k=canon_sq(king_sq,persp);s=canon_sq(piece_sq,persp)
    if MIRRORED and (k&7)>=4:k^=7;s^=7
    pt=abs(int(pc));t=(pt-1)+(0 if int(pc)*persp>0 else 5)
    if MIRRORED:
        kb=(k>>3)*4+(k&7);return (kb*10+t)*64+s
    return (k*10+t)*64+s
@njit(cache=False)
def rebuild_one(board,persp,out):
    king=find_king(board,persp)
    for j in range(NNUE_DIM):out[j]=int(FT_B[j])
    if king<0:return
    for sq in range(64):
        pc=int(board[sq])
        if pc==0 or abs(pc)==K:continue
        idx=feature_index(king,sq,pc,persp)
        for j in range(NNUE_DIM):out[j]+=int(FT_W[idx,j])
@njit(cache=False)
def rebuild_both(board,out2):rebuild_one(board,WHITE,out2[0]);rebuild_one(board,BLACK,out2[1])
@njit(cache=False,inline='always')
def add_feature(acc,idx,sign):
    for j in range(NNUE_DIM):acc[j]+=sign*int(FT_W[idx,j])
@njit(cache=False)
def update_child_after_move(board_after,m,side,captured,parent_acc,child_acc):
    for p in range(2):
        for j in range(NNUE_DIM):child_acc[p,j]=parent_acc[p,j]
    fr=m_from(m);to=m_to(m);pr=m_promo(m);fl=m_flag(m);moved_after=int(board_after[to]);moving_before=side*P if pr else moved_after;is_king=abs(moving_before)==K
    for pi in range(2):
        persp=WHITE if pi==0 else BLACK
        if is_king and persp==side:rebuild_one(board_after,persp,child_acc[pi]);continue
        king=find_king(board_after,persp)
        if not is_king:add_feature(child_acc[pi],feature_index(king,fr,moving_before,persp),-1);add_feature(child_acc[pi],feature_index(king,to,moved_after,persp),1)
        if captured!=0:
            cs=to-8*side if fl==FLAG_EP else to;add_feature(child_acc[pi],feature_index(king,cs,captured,persp),-1)
        if fl==FLAG_CASTLE:
            if to==6:rf,rt=7,5
            elif to==2:rf,rt=0,3
            elif to==62:rf,rt=63,61
            else:rf,rt=56,59
            rook=side*R;add_feature(child_acc[pi],feature_index(king,rf,rook,persp),-1);add_feature(child_acc[pi],feature_index(king,rt,rook,persp),1)
@njit(cache=False,inline='always')
def clip_ft(x):
    if x<=0:return 0
    if x>=FT_SCALE:return FT_SCALE
    return x
@njit(cache=False,inline='always')
def clip_hidden(x):
    if x<=0:return 0
    if x>=ACT_MAX:return ACT_MAX
    return x
@njit(cache=False)
def evaluate_nnue(acc2,side,board):
    first=0 if side==WHITE else 1;second=1-first;h1=np.empty(32,np.int64);h2=np.empty(32,np.int64)
    for u in range(32):
        total=np.int64(H1_B[u])
        for j in range(NNUE_DIM):total+=np.int64(H1_W[u,j])*np.int64(clip_ft(int(acc2[first,j])))
        off=NNUE_DIM
        for j in range(NNUE_DIM):total+=np.int64(H1_W[u,off+j])*np.int64(clip_ft(int(acc2[second,j])))
        if DUAL_ACT:
            off2=2*NNUE_DIM
            for j in range(NNUE_DIM):
                x=clip_ft(int(acc2[first,j]));total+=np.int64(H1_W[u,off2+j])*np.int64((x*x)//FT_SCALE)
            off3=3*NNUE_DIM
            for j in range(NNUE_DIM):
                x=clip_ft(int(acc2[second,j]));total+=np.int64(H1_W[u,off3+j])*np.int64((x*x)//FT_SCALE)
        h1[u]=clip_hidden(int(total//H1_DIV))
    for u in range(32):
        total=np.int64(H2_B[u])
        for j in range(32):total+=np.int64(H2_W[u,j])*h1[j]
        h2[u]=clip_hidden(int(total//H2_DIV))
    bucket=0
    if USE_BUCKETS:
        cnt=0
        for s in range(64):
            if board[s]!=0:cnt+=1
        bucket=(cnt-2)//4
        if bucket<0:bucket=0
        if bucket>7:bucket=7
    total=np.int64(OUT_B[bucket])
    for j in range(32):total+=np.int64(OUT_W[bucket,j])*h2[j]
    return int((total*CP_SCALE)//(HEAD_SCALE*FT_SCALE))
