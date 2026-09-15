"""Parable V4.0 — stack-search NNUE engine for AI Chessathon.

Evaluator: B Mirrored HalfKP-256 -> 32 -> 32 -> 1 (trained weights in weights/).
Search: non-recursive PVS + TT + LMR + null move + tactical quiescence.
Runtime target: one CPU core, 120s + 0.5s increment, <=90s cold initialization.
"""
import time
import numpy as np
from movegen import *
from nnue_core import *
from search_core import *

ARCHITECTURE_NAME='B Mirrored HalfKP-256 -> 32 -> 32 -> 1 / Parable V4 stack search'
TT_SIZE=1<<20
_TT_KEYS=np.zeros(TT_SIZE,np.uint64)
_TT_SCORES=np.zeros(TT_SIZE,np.int32)
_TT_DEPTHS=np.zeros(TT_SIZE,np.int8)
_TT_FLAGS=np.zeros(TT_SIZE,np.int8)
_TT_MOVES=np.full(TT_SIZE,-1,np.int32)
_BUFFERS=np.zeros((MAX_PLY,256),np.int32)
_SCOREBUF=np.zeros((MAX_PLY,256),np.int32)
_ACC=np.zeros((MAX_PLY,2,NNUE_DIM),np.int32)
_KILLERS=np.full((MAX_PLY,2),-1,np.int32)
_HISTORY=np.zeros((2,64,64),np.int32)
_GAME_KEYS=[]
_LAST_FULLMOVE=0
_ENGINE_SIDE=0
_AFTER_OUR_MOVE=None
_NPS_EST=80000.0


def _clear_search_state():
    _TT_KEYS.fill(0);_TT_SCORES.fill(0);_TT_DEPTHS.fill(0);_TT_FLAGS.fill(0);_TT_MOVES.fill(-1)
    _KILLERS.fill(-1);_HISTORY.fill(0)


def _root_once(board,side,rights,ep,halfmove,depth,node_limit,prev_best=-1,a=-MATE,b=MATE,ban_move=-1):
    stats=np.zeros(2,np.int64)
    wk,bk=find_kings(board);rebuild_both_ks(board,_ACC[0],wk,bk)
    m,sc=root_search(board,side,rights,ep,halfmove,depth,_BUFFERS,_SCOREBUF,_ACC,stats,int(node_limit),
                     _TT_KEYS,_TT_SCORES,_TT_DEPTHS,_TT_FLAGS,_TT_MOVES,_KILLERS,_HISTORY,
                     np.int32(prev_best),int(a),int(b),np.int32(ban_move))
    return int(m),int(sc),stats


def _iterative(board,side,rights,ep,halfmove,node_limit,soft_ms,hard_ms,aspiration=True,ban_move=-1):
    stats=np.zeros(2,np.int64);wk,bk=find_kings(board);rebuild_both_ks(board,_ACC[0],wk,bk)
    n=int(gen_legal_ks(board,side,rights,ep,_BUFFERS[0],wk if side==WHITE else bk))
    if n<=0:return -1,-MATE,0,stats,0.0
    best=int(_BUFFERS[0,0])
    if best==ban_move and n>1:best=int(_BUFFERS[0,1])
    if n==1:return best,0,0,stats,0.0
    score=int(evaluate_nnue(_ACC[0],side,board));completed=0;stable=0
    start=time.perf_counter();last_best=best;last_score=score
    for depth in range(1,19):
        if stats[1]:break
        if aspiration and depth>=4:
            margin=55+8*min(depth,10);a=max(-MATE,score-margin);b=min(MATE,score+margin)
        else:a,b=-MATE,MATE
        m,sc=root_search(board,side,rights,ep,halfmove,depth,_BUFFERS,_SCOREBUF,_ACC,stats,int(node_limit),
                         _TT_KEYS,_TT_SCORES,_TT_DEPTHS,_TT_FLAGS,_TT_MOVES,_KILLERS,_HISTORY,
                         np.int32(best),int(a),int(b),np.int32(ban_move))
        if stats[1]:break
        m=int(m);sc=int(sc)
        if m<0:break
        if aspiration and depth>=4 and (sc<=a or sc>=b):
            m2,sc2=root_search(board,side,rights,ep,halfmove,depth,_BUFFERS,_SCOREBUF,_ACC,stats,int(node_limit),
                               _TT_KEYS,_TT_SCORES,_TT_DEPTHS,_TT_FLAGS,_TT_MOVES,_KILLERS,_HISTORY,
                               np.int32(m),-MATE,MATE,np.int32(ban_move))
            if stats[1]:break
            m,sc=int(m2),int(sc2)
        if m==last_best and abs(sc-last_score)<=24:stable+=1
        else:stable=0
        last_best,last_score=m,sc;best,score,completed=m,sc,depth
        if abs(score)>=MATE-100:break
        elapsed_ms=(time.perf_counter()-start)*1000.0
        # Stop near the soft budget only when the PV has settled; otherwise spend toward hard budget.
        if depth>=4 and elapsed_ms>=soft_ms and stable>=2:break
        if elapsed_ms>=hard_ms*0.92:break
    elapsed=time.perf_counter()-start
    return best,score,completed,stats,elapsed


def _startup_compile_and_calibrate():
    """Force JIT during the referee's init window, then measure actual local NPS."""
    global _NPS_EST
    fen='r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1'
    b,s,r,e,hm,fm=parse_fen(fen)
    # First call pays cold JIT cost. Small limit minimizes non-compilation work.
    _root_once(b,s,r,e,hm,2,900)
    _clear_search_state()
    t=time.perf_counter();_,_,st=_root_once(b,s,r,e,hm,7,12000);dt=max(0.01,time.perf_counter()-t)
    measured=float(st[0])/dt
    # Hard-node budgets use a mild safety haircut; actual game moves continuously relearn NPS.
    _NPS_EST=max(20000.0,min(500000.0,measured*0.95))
    _clear_search_state()


_startup_compile_and_calibrate()


def _time_budget_ms(t):
    """Soft/hard budgets tuned for 120+0.5: preserve increment but spend on unstable positions."""
    t=max(0,int(t))
    if t>=90000:return 1500,2600
    if t>=60000:return 1300,2200
    if t>=35000:return 1050,1800
    if t>=20000:return 800,1350
    if t>=10000:return 550,900
    if t>=5000:return 340,560
    if t>=2000:return 180,300
    if t>=800:return 80,140
    if t>=500:return 35,65
    if t>=300:return 18,36
    return 6,14


def _would_immediate_draw(board,side,rights,ep,halfmove,m):
    """Whether playing m immediately reaches a claimable draw, with mate precedence."""
    moving=int(board[m_from(m)]);wascap=is_capture(board,m)
    cap,nr,nep=make_move(board,m,side,rights,ep)
    key=int(hash_board(board,-side,nr,nep))
    nh=0 if abs(moving)==P or wascap else halfmove+1
    fifty=(nh>=100);mate=False
    if fifty:
        wk,bk=find_kings(board);opp=-side;ok=wk if opp==WHITE else bk
        if ok>=0 and is_attacked(board,ok,side):
            n=int(gen_legal_ks(board,opp,nr,nep,_BUFFERS[1],ok))
            if n==0:fifty=False;mate=True
    repetition=(sum(1 for k in _GAME_KEYS if k==key)>=2)
    unmake_move(board,m,side,cap)
    return False if mate else (fifty or repetition)


def _remember_after_move(board,side,rights,ep,m):
    global _AFTER_OUR_MOVE
    _GAME_KEYS.append(int(hash_board(board,side,rights,ep)))
    cap,nr,nep=make_move(board,m,side,rights,ep)
    _GAME_KEYS.append(int(hash_board(board,-side,nr,nep)))
    _AFTER_OUR_MOVE=board.copy()
    unmake_move(board,m,side,cap)
    if len(_GAME_KEYS)>500:del _GAME_KEYS[:-500]


def get_move(fen:str,time_left_ms:int)->str:
    global _NPS_EST,_LAST_FULLMOVE,_ENGINE_SIDE,_AFTER_OUR_MOVE
    board,side,rights,ep,halfmove,fullmove=parse_fen(fen)
    new_game=False
    if _ENGINE_SIDE!=0:
        if side!=_ENGINE_SIDE or fullmove!=_LAST_FULLMOVE+1:new_game=True
        elif _AFTER_OUR_MOVE is not None:
            changed=int(np.count_nonzero(board!=_AFTER_OUR_MOVE))
            if changed<2 or changed>4:new_game=True
    if new_game:
        _GAME_KEYS.clear();_clear_search_state();_AFTER_OUR_MOVE=None
    _ENGINE_SIDE=side;_LAST_FULLMOVE=fullmove
    _HISTORY[:]//=2
    wk,bk=find_kings(board);n=int(gen_legal_ks(board,side,rights,ep,_BUFFERS[0],wk if side==WHITE else bk))
    if n<=0:return '0000'
    if n==1:
        best=int(_BUFFERS[0,0]);_remember_after_move(board,side,rights,ep,best);return move_to_uci(best)

    soft_ms,hard_ms=_time_budget_ms(time_left_ms)
    if time_left_ms<300:node_floor=250
    elif time_left_ms<800:node_floor=500
    elif time_left_ms<2000:node_floor=800
    elif time_left_ms<5000:node_floor=1800
    else:node_floor=4000
    if time_left_ms>=10000:safety=0.90
    elif time_left_ms>=5000:safety=0.82
    elif time_left_ms>=2000:safety=0.76
    else:safety=0.65
    hard_nodes=max(node_floor,min(2200000,int(_NPS_EST*(hard_ms/1000.0)*safety)))
    best,score,depth,stats,elapsed=_iterative(board,side,rights,ep,halfmove,hard_nodes,soft_ms,hard_ms,True,-1)
    if elapsed>0.02 and stats[0]>1000:
        observed=float(stats[0])/elapsed
        observed=max(15000.0,min(600000.0,observed))
        blend=0.35 if observed<_NPS_EST else 0.12
        _NPS_EST=(1.0-blend)*_NPS_EST+blend*observed
    if best<0:best=int(_BUFFERS[0,0])

    # Immediate FIDE draw handling at the root. A draw is worth 0: never throw it
    # away for a searched alternative that is not still clearly positive.
    best_draw=_would_immediate_draw(board,side,rights,ep,halfmove,best)
    if best_draw and depth>=3 and score>80:
        alt_nodes=max(1200,min(max(1200,hard_nodes//3),int(_NPS_EST*0.30)))
        alt,asc,ad,ast,aelapsed=_iterative(board,side,rights,ep,halfmove,alt_nodes,100,300,True,best)
        if alt>=0 and ad>=2 and asc>50 and not _would_immediate_draw(board,side,rights,ep,halfmove,alt):best=alt
    elif score<-80 and not best_draw:
        # When losing, take an immediately available repetition/50-move draw.
        for i in range(n):
            cand=int(_BUFFERS[0,i])
            if _would_immediate_draw(board,side,rights,ep,halfmove,cand):
                best=cand;break
    _remember_after_move(board,side,rights,ep,best)
    return move_to_uci(best)
