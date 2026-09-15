"""Sage Stage 3. Team-owned original search plus measured local improvements."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
import sys,time,hashlib
from pathlib import Path
import chess
START=time.perf_counter()
import zengine,zsearch,zeval
BUILD='sage-stage3er2-mopup-safe-antirepeat-20260910'
PATH=Path(__file__).with_name('weights.npz')
WEIGHTS=hashlib.sha256(PATH.read_bytes()).hexdigest()
FEATURES={'staged_tt':zsearch.STAGED_TT,'fast_legal':zsearch.FAST_LEGAL,
          'cache_chains':zeval.CACHE_CHAINS,'corrections':zsearch.CORRECTIONS,'mopup':zeval.MOPUP_ENABLED,'anti_repeat':zengine.ANTI_REPEAT}
ENGINE=zengine.Engine(str(PATH))
for fen in (chess.STARTING_FEN,'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1',
            '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1'):
    ENGINE.search(fen,30,60,max_depth=4)
ENGINE.new_game()
print(f'{BUILD} init={time.perf_counter()-START:.3f}s weights={WEIGHTS}',file=sys.stderr,flush=True)

def get_move(fen:str,time_left_ms:int)->str:
    board=chess.Board(fen);begin=time.perf_counter()
    fallback=next(iter(board.legal_moves),None)
    if fallback is None:raise ValueError('Called in terminal position')
    if time_left_ms<100:
        move=fallback.uci()
        ENGINE.last_info={'build':BUILD,'move':move,'fen':fen,'depth':0,'score_cp':None,
                          'forced_move':board.legal_moves.count()==1,'fallback':board.legal_moves.count()>1,
                          'stats':{'nodes':0,'stop':0},'clock_before_ms':time_left_ms,'weights_sha256':WEIGHTS}
        return move
    soft,hard=zengine.budget(time_left_ms,500,board.ply())
    move,score,depth=ENGINE.search(fen,soft,hard)
    if chess.Move.from_uci(move) not in board.legal_moves:raise RuntimeError('Illegal search move')
    pv=zsearch.find_pv(ENGINE.bd,ENGINE.st,ENGINE.undo,ENGINE.hs,ENGINE.tt_key,ENGINE.tt_data,max_len=12)
    # The inherited driver can select a partial-iteration move while the root
    # TT still contains the previous completed iteration's variation.
    if not pv or pv[0]!=move:pv=[move]
    ENGINE.push_own_move(ENGINE.last_move)
    ENGINE.last_info.update(build=BUILD,move=move,fen=fen,pv=pv,weights_sha256=WEIGHTS,features=FEATURES,
                           elapsed_ms=round((time.perf_counter()-begin)*1000,3),
                           soft_ms=soft,hard_ms=hard,clock_before_ms=time_left_ms)
    print(f"S3E {move} d{depth} s{score} n{int(ENGINE.sinfo[0])} t{ENGINE.last_info['elapsed_ms']} stop{int(ENGINE.sinfo[1])}",file=sys.stderr,flush=True)
    return move
