"""Sage Fusion competition entry: get_move(fen, remaining milliseconds).

Team-supplied Sage/Fable core with original search and reliability improvements.
Only the preinstalled Python stack is used; no background thinking or network.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import json
import sys
import time
from pathlib import Path
import chess

_HERE = Path(__file__).resolve().parent
_START = time.perf_counter()
with (_HERE / 'config.json').open() as _f:
    _config = json.load(_f)
os.environ.setdefault('SAGE_TT_BITS', str(_config['tt_bits']))
import zengine
import zboard as zb
from endgames import Endgames, position_key
from collections import Counter
_SEEN = Counter()
_ENDGAMES = Endgames(_HERE / "tablebases")

_MODEL = os.environ.get('SAGE_MODEL', _config['model'])
if _MODEL not in ('150m', '50m', 'ensemble'):
    raise ValueError('SAGE_MODEL must be 150m, 50m or ensemble')
ENGINE = zengine.Engine(str(_HERE / 'models' / ('sage50m.npz' if _MODEL == '50m' else 'sage150m.npz')))
if _MODEL == 'ensemble':
    ENGINE.blend_with(str(_HERE / 'models' / 'sage50m.npz'))
# Compile during the free init window, including special-move/evaluation paths.
for _fen in (chess.STARTING_FEN,
             'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1',
             '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1',
             '4k3/P7/8/8/8/8/7p/4K3 w - - 0 1'):
    ENGINE.search(_fen, 12, 25, max_depth=3)
    ENGINE.push_own_move(int(ENGINE.last_move))
ENGINE.new_game()
_LAST_REQUEST = None
_LAST_REPLY = None
print(f'Sage Fusion ready: model={_MODEL}, TT={ENGINE.tt_key.nbytes*2//1048576} MiB, init={time.perf_counter()-_START:.2f}s', file=sys.stderr)


def _record(fen, uci):
    """Restore the root before recording even if search raised mid-branch."""
    ENGINE.set_position(fen)
    n = zb.gen_moves(ENGINE.bd, ENGINE.st, ENGINE.mvbuf[0], False)
    for i in range(n):
        m = int(ENGINE.mvbuf[0, i])
        if zb.move_to_uci(m) == uci:
            ENGINE.last_move = m
            ENGINE.push_own_move(m)
            return


def get_move(fen: str, time_left_ms: int) -> str:
    global _LAST_REQUEST, _LAST_REPLY
    start = time.perf_counter()
    if fen == _LAST_REQUEST:
        return _LAST_REPLY
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        raise ValueError('get_move called after game end')
    _SEEN[position_key(board)] += 1
    fallback = legal[0].uci()
    chosen = fallback
    table_move = None
    if time_left_ms >= 100 and len(board.piece_map()) <= 4:
        table_move = _ENDGAMES.choose(board, _SEEN, start + min(0.05, time_left_ms / 10000))
        if table_move in legal:
            chosen = table_move.uci()
    if table_move is None and len(legal) > 1 and time_left_ms >= 80:
        soft, hard = zengine.budget(time_left_ms, 500, board.ply())
        overhead = (time.perf_counter() - start) * 1000
        try:
            proposed, score, depth = ENGINE.search(fen, max(0, soft-overhead), max(0, hard-overhead))
            move = chess.Move.from_uci(proposed)
            if move in legal:
                chosen = proposed
            else:
                print(f'Search returned an illegal move: {proposed}; using legal fallback', file=sys.stderr)
        except Exception as exc:
            print(f'Search fallback: {type(exc).__name__}: {exc}', file=sys.stderr)
    try:
        _record(fen, chosen)
    except Exception as exc:
        # A history/bookkeeping error must not throw away a legal move.
        ENGINE.game_hashes.clear()
        print(f'History reset: {type(exc).__name__}: {exc}', file=sys.stderr)
    board.push_uci(chosen)
    _SEEN[position_key(board)] += 1
    _LAST_REQUEST, _LAST_REPLY = fen, chosen
    return chosen
