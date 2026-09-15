"""AI Chessathon NNUE architecture candidate: B Mirrored HalfKP-256 -> 32 -> 32 -> 1.
Bootstrap weights are runnable; replace them with same-data trained weights for the architecture tournament.
"""
import time
import numpy as np
from movegen import *
from nnue_core import *
from search_core import *
ARCHITECTURE_NAME='B Mirrored HalfKP-256 -> 32 -> 32 -> 1'
TT_SIZE=1<<19
_TT_KEYS=np.zeros(TT_SIZE,np.uint64);_TT_SCORES=np.zeros(TT_SIZE,np.int32);_TT_DEPTHS=np.zeros(TT_SIZE,np.int8);_TT_FLAGS=np.zeros(TT_SIZE,np.int8);_TT_MOVES=np.full(TT_SIZE,-1,np.int32)
_BUFFERS=np.zeros((MAX_PLY,256),np.int32);_ACC=np.zeros((MAX_PLY,2,NNUE_DIM),np.int32);_KILLERS=np.full((MAX_PLY,2),-1,np.int32);_HISTORY=np.zeros((2,64,64),np.int32);_GAME_KEYS=[]

def _clear_search_state():
    _TT_KEYS.fill(0);_TT_SCORES.fill(0);_TT_DEPTHS.fill(0);_TT_FLAGS.fill(0);_TT_MOVES.fill(-1);_KILLERS.fill(-1);_HISTORY.fill(0)

def _iterative(board,side,rights,ep,node_limit,aspiration=True,ban_move=-1):
    stats=np.zeros(2,np.int64);rebuild_both(board,_ACC[0]);n=int(gen_legal(board,side,rights,ep,_BUFFERS[0]))
    if n<=0:return -1,-MATE,0,stats
    best=int(_BUFFERS[0,0])
    if best==ban_move and n>1:best=int(_BUFFERS[0,1])
    if n==1:return best,0,0,stats
    score=int(evaluate_nnue(_ACC[0],side,board));completed=0
    for depth in range(1,19):
        if stats[1]:break
        if aspiration and depth>=4:
            margin=70+10*min(depth,8);a=max(-MATE,score-margin);b=min(MATE,score+margin)
        else:a,b=-MATE,MATE
        m,sc=root_search(board,side,rights,ep,depth,_BUFFERS,_ACC,stats,int(node_limit),_TT_KEYS,_TT_SCORES,_TT_DEPTHS,_TT_FLAGS,_TT_MOVES,_KILLERS,_HISTORY,np.int32(best),int(a),int(b),np.int32(ban_move))
        if stats[1]:break
        m=int(m);sc=int(sc)
        if m<0:break
        if aspiration and depth>=4 and (sc<=a or sc>=b):
            m2,sc2=root_search(board,side,rights,ep,depth,_BUFFERS,_ACC,stats,int(node_limit),_TT_KEYS,_TT_SCORES,_TT_DEPTHS,_TT_FLAGS,_TT_MOVES,_KILLERS,_HISTORY,np.int32(m),-MATE,MATE,np.int32(ban_move))
            if stats[1]:break
            m,sc=int(m2),int(sc2)
        best,score,completed=m,sc,depth
        if abs(score)>=MATE-100:break
    return best,score,completed,stats

def _calibrate():
    fen='r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1';b,s,r,e,hm,fm=parse_fen(fen);_iterative(b,s,r,e,6000,False);_clear_search_state();t=time.perf_counter();_,_,_,st=_iterative(b,s,r,e,20000,False);dt=max(.01,time.perf_counter()-t);measured=int(st[0]/dt);safe=max(16000,min(400000,int(measured*.44)));_clear_search_state();return safe,measured
_CAL_NPS,_RAW_NPS=_calibrate()

def _budget_ms(t):
    t=max(0,int(t))
    if t>=90000:b=1750
    elif t>=60000:b=1500
    elif t>=35000:b=1150
    elif t>=20000:b=850
    elif t>=10000:b=600
    elif t>=5000:b=380
    elif t>=2000:b=220
    elif t>=800:b=110
    else:b=40
    if t>280:b=min(b,max(30,(t-200)//3))
    else:b=min(b,22)
    return max(18,int(b))

def _material_edge(board,side):
    vals=(0,1,3,3,5,9,0);bal=0
    for sq in range(64):
        pc=int(board[sq])
        if pc:bal+=(1 if pc*side>0 else -1)*vals[abs(pc)]
    return bal

def _child_info(board,side,rights,ep,halfmove,m):
    moving=int(board[m_from(m)]);wascap=is_capture(board,m);cap,nr,nep=make_move(board,m,side,rights,ep);key=int(hash_board(board,-side,nr,nep));nh=0 if abs(moving)==P or wascap else halfmove+1;unmake_move(board,m,side,cap);return key,nh

def _would_immediate_draw(board,side,rights,ep,halfmove,m):
    key,nh=_child_info(board,side,rights,ep,halfmove,m)
    return nh>=100 or sum(1 for k in _GAME_KEYS if k==key)>=2

def _best_nondraw_static(board,side,rights,ep,halfmove):
    moves=np.zeros(256,np.int32);n=int(gen_legal(board,side,rights,ep,moves));best=-1;bestv=-10**9;tmp=np.zeros((2,NNUE_DIM),np.int32)
    for i in range(n):
        m=int(moves[i])
        if _would_immediate_draw(board,side,rights,ep,halfmove,m):continue
        cap,nr,nep=make_move(board,m,side,rights,ep);rebuild_both(board,tmp);v=-int(evaluate_nnue(tmp,-side,board));unmake_move(board,m,side,cap)
        if v>bestv:bestv=v;best=m
    return best

def _remember_after_move(board,side,rights,ep,m):
    _GAME_KEYS.append(int(hash_board(board,side,rights,ep)));cap,nr,nep=make_move(board,m,side,rights,ep);_GAME_KEYS.append(int(hash_board(board,-side,nr,nep)));unmake_move(board,m,side,cap)
    if len(_GAME_KEYS)>500:del _GAME_KEYS[:-500]

def get_move(fen:str,time_left_ms:int)->str:
    board,side,rights,ep,halfmove,fullmove=parse_fen(fen);_HISTORY[:]//=2;n=int(gen_legal(board,side,rights,ep,_BUFFERS[0]))
    if n<=0:return '0000'
    if n==1:
        best=int(_BUFFERS[0,0]);_remember_after_move(board,side,rights,ep,best);return move_to_uci(best)
    ms=_budget_ms(time_left_ms);limit=max(5000,min(1300000,int(_CAL_NPS*(ms/1000.0))));best,score,depth,stats=_iterative(board,side,rights,ep,limit,True,-1)
    if best<0:best=int(_BUFFERS[0,0])
    # Referee claims threefold/50-move draws automatically. When ahead, refuse a root move
    # that immediately hands over such a draw; when behind, preserving a draw is rational.
    ahead=(score>40 or _material_edge(board,side)>0)
    if ahead and _would_immediate_draw(board,side,rights,ep,halfmove,best):
        alt,asc,ad,ast=_iterative(board,side,rights,ep,max(5000,limit//2),True,best)
        if alt>=0 and not _would_immediate_draw(board,side,rights,ep,halfmove,alt):best=alt
        else:
            alt2=_best_nondraw_static(board,side,rights,ep,halfmove)
            if alt2>=0:best=alt2
    _remember_after_move(board,side,rights,ep,best);return move_to_uci(best)
