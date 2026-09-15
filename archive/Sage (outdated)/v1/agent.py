"""Shared comparison agent. Advanced per-model search tuning is a later experiment."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_key]='1'
import time
from pathlib import Path
import chess
import zengine
_START=time.perf_counter()
ENGINE=zengine.Engine(str(Path(__file__).parent/'weights.npz'))
for _fen in (chess.STARTING_FEN,'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1',
             '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1'):
    ENGINE.search(_fen,30,60,max_depth=4)
ENGINE.new_game()

def get_move(fen: str,time_left_ms: int)->str:
    board=chess.Board(fen)
    fallback=next(iter(board.legal_moves),None)
    if fallback is None: raise ValueError('Called in terminal position')
    if time_left_ms<100: return fallback.uci()
    soft,hard=zengine.budget(time_left_ms,500,board.ply())
    uci,score,depth=ENGINE.search(fen,soft,hard)
    move=chess.Move.from_uci(uci)
    if move not in board.legal_moves: raise RuntimeError('Search correctness failure: illegal move')
    ENGINE.push_own_move(ENGINE.last_move)
    return uci
