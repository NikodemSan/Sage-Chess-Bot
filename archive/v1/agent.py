"""AI Chessathon submission: original PVS engine with a team-trained HalfKP NNUE.

The neural network is loaded and JIT compilation/calibration is performed at module import,
inside the competition's free initialisation window. No external chess engine, published
network, or runtime move/evaluation database is used.
"""
import os,time
import numpy as np
from movegen import *
from nnue_core import *
from search_core import *

# Load our quantised network trained from random initialisation.
_WPATH=os.path.join(os.path.dirname(__file__),'weights','nnue_weights.npz')
_z=np.load(_WPATH,allow_pickle=False)
_FT_W=np.ascontiguousarray(_z['ft_w'],dtype=np.int16)
_FT_B=np.ascontiguousarray(_z['ft_b'],dtype=np.int32)
_HEAD_W=np.ascontiguousarray(_z['head_w'],dtype=np.int16)
_HEAD_B=np.int64(_z['head_b'].item())
_z.close()

TT_SIZE=1<<19
_TT_KEYS=np.zeros(TT_SIZE,np.uint64)
_TT_SCORES=np.zeros(TT_SIZE,np.int32)
_TT_DEPTHS=np.zeros(TT_SIZE,np.int8)
_TT_FLAGS=np.zeros(TT_SIZE,np.int8)
_TT_MOVES=np.full(TT_SIZE,-1,np.int32)
_BUFFERS=np.zeros((MAX_PLY,256),np.int32)
_ACC=np.zeros((MAX_PLY,2,NNUE_DIM),np.int32)
_KILLERS=np.full((MAX_PLY,2),-1,np.int32)
_HISTORY=np.zeros((2,64,64),np.int32)


def _clear_search_state():
    _TT_KEYS.fill(0); _TT_SCORES.fill(0); _TT_DEPTHS.fill(0); _TT_FLAGS.fill(0); _TT_MOVES.fill(-1)
    _KILLERS.fill(-1); _HISTORY.fill(0)


def _iterative(board,side,rights,ep,node_limit,aspiration=True):
    stats=np.zeros(2,np.int64)
    rebuild_both(board,_FT_W,_FT_B,_ACC[0])
    n=int(gen_legal(board,side,rights,ep,_BUFFERS[0]))
    if n<=0: return -1,-MATE,0,stats
    best=int(_BUFFERS[0,0])
    if n==1: return best,0,0,stats
    score=int(evaluate_nnue(_ACC[0],side,_HEAD_W,_HEAD_B)); completed=0
    for depth in range(1,19):
        if stats[1]: break
        if aspiration and depth>=4:
            margin=70+10*min(depth,8); a=max(-MATE,score-margin); b=min(MATE,score+margin)
        else: a,b=-MATE,MATE
        m,sc=root_search(board,side,rights,ep,depth,_BUFFERS,_ACC,_FT_W,_FT_B,_HEAD_W,_HEAD_B,stats,int(node_limit),
                         _TT_KEYS,_TT_SCORES,_TT_DEPTHS,_TT_FLAGS,_TT_MOVES,_KILLERS,_HISTORY,np.int32(best),int(a),int(b))
        if stats[1]: break
        m=int(m); sc=int(sc)
        if aspiration and depth>=4 and (sc<=a or sc>=b):
            m2,sc2=root_search(board,side,rights,ep,depth,_BUFFERS,_ACC,_FT_W,_FT_B,_HEAD_W,_HEAD_B,stats,int(node_limit),
                               _TT_KEYS,_TT_SCORES,_TT_DEPTHS,_TT_FLAGS,_TT_MOVES,_KILLERS,_HISTORY,np.int32(m),-MATE,MATE)
            if stats[1]: break
            m,sc=int(m2),int(sc2)
        best,score,completed=m,sc,depth
        if abs(score)>=MATE-100: break
    return best,score,completed,stats


def _calibrate():
    # Compiles all hot Numba paths and obtains a conservative machine-specific node rate.
    fen='r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1'
    b,s,r,e,hm,fm=parse_fen(fen)
    _iterative(b,s,r,e,25000,False)
    _clear_search_state()
    t=time.perf_counter(); _,_,_,st=_iterative(b,s,r,e,140000,False); dt=max(0.01,time.perf_counter()-t)
    measured=int(st[0]/dt)
    # Search cost varies sharply with tactical branching. Use a large safety haircut because
    # flagging loses the whole game, while leaving a few tenths of a second unused usually
    # costs only a fraction of a ply.
    safe=max(45000,min(500000,int(measured*0.48)))
    _clear_search_state()
    return safe,measured

_CAL_NPS,_RAW_NPS=_calibrate()


def _budget_ms(t):
    t=max(0,int(t))
    if t>=90000: b=1750
    elif t>=60000: b=1500
    elif t>=35000: b=1150
    elif t>=20000: b=850
    elif t>=10000: b=600
    elif t>=5000: b=380
    elif t>=2000: b=220
    elif t>=800: b=110
    else: b=40
    if t>280: b=min(b,max(30,(t-200)//3))
    else: b=min(b,22)
    return max(18,int(b))


def get_move(fen: str,time_left_ms: int) -> str:
    board,side,rights,ep,halfmove,fullmove=parse_fen(fen)
    _HISTORY[:]//=2
    n=int(gen_legal(board,side,rights,ep,_BUFFERS[0]))
    if n<=0: return '0000'
    if n==1: return move_to_uci(int(_BUFFERS[0,0]))
    ms=_budget_ms(time_left_ms)
    limit=max(7000,min(1800000,int(_CAL_NPS*(ms/1000.0))))
    best,score,depth,stats=_iterative(board,side,rights,ep,limit,True)
    if best<0:
        n=int(gen_legal(board,side,rights,ep,_BUFFERS[0])); best=int(_BUFFERS[0,0])
    return move_to_uci(best)
