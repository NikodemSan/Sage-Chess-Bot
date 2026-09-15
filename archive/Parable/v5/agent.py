"""Parable V5R — tournament search rewrite for AI Chessathon.

Evaluator (unchanged): team-trained B Mirrored HalfKP-256 -> 32 -> 32 -> 1.
Search: recursive Numba PVS with aged 2M TT, NNUE eval cache, adaptive null
move, logarithmic/history-aware LMR, shallow RFP/futility/LMP, selective
quiescence, killers/history/countermoves/capture history, root reductions,
mate-distance pruning and aspiration windows.

Target: one CPU core, 120s + 0.5s increment, <=90s cold initialization.
"""
import time
import numpy as np
from movegen import *
from nnue_core import *
from search_core import *

ARCHITECTURE_NAME='B Mirrored HalfKP-256 -> 32 -> 32 -> 1 / Parable V5R recursive advanced search'
TT_SIZE=1<<21
EVAL_CACHE_SIZE=1<<19
_TT_KEYS=np.zeros(TT_SIZE,np.uint64)
_TT_SCORES=np.zeros(TT_SIZE,np.int32)
_TT_DEPTHS=np.zeros(TT_SIZE,np.int8)
_TT_FLAGS=np.zeros(TT_SIZE,np.int8)
_TT_MOVES=np.full(TT_SIZE,-1,np.int32)
_TT_AGES=np.zeros(TT_SIZE,np.uint8)
_EC_KEYS=np.zeros(EVAL_CACHE_SIZE,np.uint64)
_EC_SCORES=np.zeros(EVAL_CACHE_SIZE,np.int16)
_BUFFERS=np.zeros((MAX_PLY,256),np.int32)
_SCOREBUF=np.zeros((MAX_PLY,256),np.int32)
_ACC=np.zeros((MAX_PLY,2,NNUE_DIM),np.int32)
_PATH_KEYS=np.zeros(MAX_PLY,np.uint64)
_EVAL_X=np.empty(2*NNUE_DIM,np.int16)
_EVAL_H=np.empty(64,np.int64)
_KILLERS=np.full((MAX_PLY,2),-1,np.int32)
_HISTORY=np.zeros((2,64,64),np.int32)
_GAME_KEYS=[]
_LAST_FULLMOVE=0
_ENGINE_SIDE=0
_AFTER_OUR_MOVE=None
_NPS_EST=100000.0
_TT_GEN=1
_MOVE_COUNT=0


def _clear_search_state():
    global _TT_GEN
    _TT_KEYS.fill(0);_TT_SCORES.fill(0);_TT_DEPTHS.fill(0);_TT_FLAGS.fill(0);_TT_MOVES.fill(-1);_TT_AGES.fill(0)
    _EC_KEYS.fill(0);_EC_SCORES.fill(0)
    _KILLERS.fill(-1);_HISTORY.fill(0)
    _BUFFERS.fill(0);_SCOREBUF.fill(0);_ACC.fill(0);_PATH_KEYS.fill(0);_EVAL_X.fill(0);_EVAL_H.fill(0)
    _TT_GEN=1


def _root_once(board,side,rights,ep,halfmove,depth,node_limit,prev_best=-1,a=-MATE,b=MATE,ban_move=-1):
    stats=np.zeros(8,np.int64)
    wk,bk=find_kings(board);rebuild_both_ks(board,_ACC[0],wk,bk)
    m,sc=root_search(board,side,rights,ep,halfmove,depth,_BUFFERS,_SCOREBUF,_ACC,_PATH_KEYS,stats,int(node_limit),
                     _TT_KEYS,_TT_SCORES,_TT_DEPTHS,_TT_FLAGS,_TT_MOVES,_TT_AGES,_EC_KEYS,_EC_SCORES,
                     _KILLERS,_HISTORY,int(_TT_GEN),
                     np.int32(prev_best),int(a),int(b),np.int32(ban_move),_EVAL_X,_EVAL_H)
    return int(m),int(sc),stats


def _iterative(board,side,rights,ep,halfmove,node_limit,soft_ms,hard_ms,aspiration=True,ban_move=-1):
    stats=np.zeros(8,np.int64);wk,bk=find_kings(board);rebuild_both_ks(board,_ACC[0],wk,bk)
    n=int(gen_legal_ks(board,side,rights,ep,_BUFFERS[0],wk if side==WHITE else bk))
    if n<=0:return -1,-MATE,0,stats,0.0
    best=int(_BUFFERS[0,0])
    if best==ban_move and n>1:best=int(_BUFFERS[0,1])
    if n==1:return best,0,0,stats,0.0
    score=int(evaluate_nnue(_ACC[0],side,board))
    root_key=np.uint64(hash_board(board,side,rights,ep));eval_store(root_key,score,_EC_KEYS,_EC_SCORES)
    completed=0;stable=0
    start=time.perf_counter();last_best=best;last_score=score
    for depth in range(1,33):
        if stats[1]:break
        if aspiration and depth>=4:
            delta=min(72,30+3*depth);a=max(-MATE,score-delta);b=min(MATE,score+delta)
        else:delta=0;a,b=-MATE,MATE
        while True:
            m,sc=root_search(board,side,rights,ep,halfmove,depth,_BUFFERS,_SCOREBUF,_ACC,_PATH_KEYS,stats,int(node_limit),
                             _TT_KEYS,_TT_SCORES,_TT_DEPTHS,_TT_FLAGS,_TT_MOVES,_TT_AGES,_EC_KEYS,_EC_SCORES,
                             _KILLERS,_HISTORY,int(_TT_GEN),
                             np.int32(best),int(a),int(b),np.int32(ban_move),_EVAL_X,_EVAL_H)
            if stats[1]:break
            m=int(m);sc=int(sc)
            if m<0:break
            if not aspiration or depth<4:break
            if sc<=a and a>-MATE:
                delta=min(4096,max(32,delta*2));a=max(-MATE,a-delta);continue
            if sc>=b and b<MATE:
                delta=min(4096,max(32,delta*2));b=min(MATE,b+delta);continue
            break
        if stats[1] or int(m)<0:break
        if m==last_best and abs(sc-last_score)<=28:stable+=1
        else:stable=0
        last_best,last_score=m,sc;best,score,completed=m,sc,depth
        if abs(score)>=MATE-100:break
        elapsed_ms=(time.perf_counter()-start)*1000.0
        if depth>=5 and elapsed_ms>=soft_ms and stable>=2:break
        if elapsed_ms>=hard_ms*0.82:break
    elapsed=time.perf_counter()-start
    return best,score,completed,stats,elapsed


def _startup_compile_and_calibrate():
    """Force JIT compilation into the referee's 90-second initialization window."""
    global _NPS_EST
    fen='r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1'
    b,s,r,e,hm,fm=parse_fen(fen)
    # First call triggers compilation. Keep it shallow enough that compile time,
    # not search work, dominates startup.
    _root_once(b,s,r,e,hm,5,5000)
    wk,bk=find_kings(b);rebuild_both_ks(b,_ACC[0],wk,bk)
    _=evaluate_nnue(_ACC[0],s,b)
    hk=np.uint64(hash_board(b,s,r,e));eval_store(hk,0,_EC_KEYS,_EC_SCORES)
    _clear_search_state()
    t=time.perf_counter();_,_,st=_root_once(b,s,r,e,hm,7,30000);dt=max(0.01,time.perf_counter()-t)
    measured=float(st[0])/dt
    # Deliberately conservative because the hard tournament clock matters more
    # than spending every nominal millisecond on unusually tactical positions.
    _NPS_EST=max(18000.0,min(500000.0,measured*0.82))
    _clear_search_state()


_startup_compile_and_calibrate()


def _time_budget_ms(t):
    """120+0.5 schedule, with extra reserve for a recursive Python/Numba engine."""
    t=max(0,int(t))
    if t>=100000:return 2100,3500
    if t>=75000:return 1800,3050
    if t>=50000:return 1450,2500
    if t>=35000:return 1150,2000
    if t>=22000:return 900,1550
    if t>=12000:return 650,1100
    if t>=7000:return 430,750
    if t>=4000:return 270,470
    if t>=2000:return 150,260
    if t>=1000:return 80,140
    if t>=600:return 40,75
    if t>=350:return 20,40
    return 6,14


def _would_immediate_draw(board,side,rights,ep,halfmove,m):
    moving=int(board[m_from(m)]);wascap=(board[m_to(m)]!=0 or m_flag(m)==FLAG_EP)
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
    global _NPS_EST,_LAST_FULLMOVE,_ENGINE_SIDE,_AFTER_OUR_MOVE,_TT_GEN,_MOVE_COUNT
    board,side,rights,ep,halfmove,fullmove=parse_fen(fen)
    new_game=False
    if _ENGINE_SIDE!=0:
        if side!=_ENGINE_SIDE or fullmove!=_LAST_FULLMOVE+1:new_game=True
        elif _AFTER_OUR_MOVE is not None:
            changed=int(np.count_nonzero(board!=_AFTER_OUR_MOVE))
            if changed<2 or changed>4:new_game=True
    if new_game:
        _GAME_KEYS.clear();_clear_search_state();_AFTER_OUR_MOVE=None;_MOVE_COUNT=0
    _ENGINE_SIDE=side;_LAST_FULLMOVE=fullmove;_MOVE_COUNT+=1
    _TT_GEN=(_TT_GEN+1)&255
    if _TT_GEN==0:_TT_GEN=1
    if (_MOVE_COUNT&7)==0:
        _HISTORY[:]=(_HISTORY*3)//4
    wk,bk=find_kings(board);n=int(gen_legal_ks(board,side,rights,ep,_BUFFERS[0],wk if side==WHITE else bk))
    if n<=0:return '0000'
    if n==1:
        best=int(_BUFFERS[0,0]);_remember_after_move(board,side,rights,ep,best);return move_to_uci(best)

    soft_ms,hard_ms=_time_budget_ms(time_left_ms)
    if time_left_ms<350:node_floor=250
    elif time_left_ms<1000:node_floor=600
    elif time_left_ms<2500:node_floor=1200
    elif time_left_ms<6000:node_floor=2500
    else:node_floor=6000
    if time_left_ms>=15000:safety=0.80
    elif time_left_ms>=5000:safety=0.73
    elif time_left_ms>=1500:safety=0.65
    else:safety=0.56
    hard_nodes=max(node_floor,min(3200000,int(_NPS_EST*(hard_ms/1000.0)*safety)))
    best,score,depth,stats,elapsed=_iterative(board,side,rights,ep,halfmove,hard_nodes,soft_ms,hard_ms,True,-1)
    if elapsed>0.02 and stats[0]>1200:
        observed=float(stats[0])/elapsed;observed=max(15000.0,min(600000.0,observed))
        blend=0.40 if observed<_NPS_EST else 0.12
        _NPS_EST=(1.0-blend)*_NPS_EST+blend*observed
    if best<0:best=int(_BUFFERS[0,0])

    best_draw=_would_immediate_draw(board,side,rights,ep,halfmove,best)
    if best_draw and depth>=3 and score>90 and time_left_ms>1200:
        alt_nodes=max(1200,min(max(1200,hard_nodes//4),int(_NPS_EST*0.18)))
        alt,asc,ad,ast,aelapsed=_iterative(board,side,rights,ep,halfmove,alt_nodes,70,190,True,best)
        if alt>=0 and ad>=2 and asc>55 and not _would_immediate_draw(board,side,rights,ep,halfmove,alt):best=alt
    elif score<-90 and not best_draw:
        for i in range(n):
            cand=int(_BUFFERS[0,i])
            if _would_immediate_draw(board,side,rights,ep,halfmove,cand):best=cand;break
    _remember_after_move(board,side,rights,ep,best)
    return move_to_uci(best)
