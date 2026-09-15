"""HalfKP-style efficiently updatable neural evaluator.

The network uses one shared 40,960 x 256 sparse feature transformer.  Features are
king-relative and perspective-relative: for each perspective, the friendly king square
anchors the locations/types of all non-king pieces.  Black is canonicalised by rotating
the board 180 degrees so one learned table is shared by both colours.

At search time we maintain two int32 accumulators. Non-king moves update them by only
adding/removing the corresponding feature rows. A king move rebuilds only the moved
king's perspective, since that changes its king anchor. This is the defining NNUE update
pattern rather than a dense network recomputed at every leaf.
"""
import numpy as np
from numba import njit
from movegen import WHITE, BLACK, P,N,B,R,Q,K, FLAG_EP,FLAG_CASTLE, m_from,m_to,m_promo,m_flag,find_king

NNUE_DIM=256
NNUE_FEATURES=64*10*64  # king square x (own/enemy P/N/B/R/Q) x piece square
FT_SCALE=256
HEAD_SCALE=1024
CP_SCALE=600

@njit(cache=False, inline='always')
def canon_sq(sq,persp):
    return sq if persp==WHITE else (sq ^ 63)

@njit(cache=False, inline='always')
def feature_index(king_sq, piece_sq, pc, persp):
    # kings are deliberately not feature pieces; king location is the feature bucket.
    k=canon_sq(king_sq,persp); s=canon_sq(piece_sq,persp)
    pt=abs(int(pc))
    # P,N,B,R,Q -> 0..4; enemy versions -> 5..9
    t=(pt-1) + (0 if int(pc)*persp>0 else 5)
    return (k*10+t)*64+s

@njit(cache=False)
def rebuild_one(board,persp,ft_w,ft_b,out):
    king=find_king(board,persp)
    for j in range(NNUE_DIM): out[j]=int(ft_b[j])
    if king<0: return
    for sq in range(64):
        pc=int(board[sq])
        if pc==0 or abs(pc)==K: continue
        idx=feature_index(king,sq,pc,persp)
        for j in range(NNUE_DIM): out[j]+=int(ft_w[idx,j])

@njit(cache=False)
def rebuild_both(board,ft_w,ft_b,out2):
    rebuild_one(board,WHITE,ft_w,ft_b,out2[0])
    rebuild_one(board,BLACK,ft_w,ft_b,out2[1])

@njit(cache=False, inline='always')
def add_feature(acc,ft_w,idx,sign):
    for j in range(NNUE_DIM): acc[j]+=sign*int(ft_w[idx,j])

@njit(cache=False)
def update_child_after_move(board_after,m,side,captured,parent_acc,child_acc,ft_w,ft_b):
    """Build child accumulators from parent after board_after already contains move m.

    `captured` is the piece value returned by movegen.make_move().
    """
    # First inherit both accumulators.
    for p in range(2):
        for j in range(NNUE_DIM): child_acc[p,j]=parent_acc[p,j]
    fr=m_from(m); to=m_to(m); pr=m_promo(m); fl=m_flag(m)
    moved_after=int(board_after[to])
    moving_before=side*P if pr else moved_after
    is_king=abs(moving_before)==K

    # For each perspective, either rebuild (its king moved) or apply sparse deltas.
    for pi in range(2):
        persp=WHITE if pi==0 else BLACK
        if is_king and persp==side:
            rebuild_one(board_after,persp,ft_w,ft_b,child_acc[pi])
            continue
        king=find_king(board_after,persp)
        # Moving piece itself, unless it is a king (kings are anchors, not features).
        if not is_king:
            old_idx=feature_index(king,fr,moving_before,persp)
            add_feature(child_acc[pi],ft_w,old_idx,-1)
            new_idx=feature_index(king,to,moved_after,persp)
            add_feature(child_acc[pi],ft_w,new_idx,1)
        # Captured piece. EP capture lives behind destination square.
        if captured!=0:
            cs=to-8*side if fl==FLAG_EP else to
            ci=feature_index(king,cs,captured,persp)
            add_feature(child_acc[pi],ft_w,ci,-1)
        # Castling moves a rook as well as the king.
        if fl==FLAG_CASTLE:
            if to==6: rf,rt=7,5
            elif to==2: rf,rt=0,3
            elif to==62: rf,rt=63,61
            else: rf,rt=56,59
            rook=side*R
            add_feature(child_acc[pi],ft_w,feature_index(king,rf,rook,persp),-1)
            add_feature(child_acc[pi],ft_w,feature_index(king,rt,rook,persp),1)

@njit(cache=False, inline='always')
def clipped(x):
    if x<=0: return 0
    if x>=FT_SCALE: return FT_SCALE
    return x

@njit(cache=False)
def evaluate_nnue(acc2,side,head_w,head_b):
    # Input order is side-to-move first, opponent second, exactly as trained.
    first=0 if side==WHITE else 1
    second=1-first
    total=np.int64(head_b)
    for j in range(NNUE_DIM):
        total += np.int64(head_w[j])*np.int64(clipped(int(acc2[first,j])))
    off=NNUE_DIM
    for j in range(NNUE_DIM):
        total += np.int64(head_w[off+j])*np.int64(clipped(int(acc2[second,j])))
    # head output was trained in units cp/CP_SCALE.
    return int((total*CP_SCALE)//(HEAD_SCALE*FT_SCALE))
