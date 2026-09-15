"""The submission entrypoint. The platform imports this file and calls get_move.

Everything heavy happens at import: numba compiles the board, evaluation and search, and the
NNUE weights (if shipped alongside) are loaded. The search itself lives in zboard / zeval /
zsearch / zengine; this file only manages the clock and guards the protocol.
"""

import os
import random
import time
import warnings

_t_import = time.perf_counter()
warnings.filterwarnings("ignore", message=".*object mode.*")

import chess  # noqa: E402

import zengine  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _find(name: str) -> str | None:
    for cand in (os.path.join(HERE, "weights", name), os.path.join(HERE, name)):
        if os.path.exists(cand):
            return cand
    return None


NNUE_PATH = _find("nnue.npz")
PSQT_PATH = _find("psqt.npz")

INCREMENT_MS = 500
PONDER = True

ENGINE = zengine.Engine(NNUE_PATH)
if PSQT_PATH:
    ENGINE.load_psqt(PSQT_PATH)
if ENGINE.mode >= 1:
    # blend of NNUE and tuned tables: measured +211 Elo over either alone (see README notes)
    ENGINE.blend = 1
    ENGINE.mode = 2

# Warm up: compile every jitted path inside the init budget, not on the clock.
for _fen in (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
):
    _uci, _score, _depth = ENGINE.search(_fen, 60, 120, max_depth=6)
    ENGINE.push_own_move(ENGINE.last_move)
    _b = chess.Board(_fen)
    _b.push(chess.Move.from_uci(_uci))
    ENGINE.start_ponder(_b.fen())
    time.sleep(0.05)
    ENGINE.stop_ponder()
ENGINE.new_game()
print(f"init done in {time.perf_counter() - _t_import:.1f}s, eval mode {('classical', 'nnue', 'nnue+psqt blend')[ENGINE.mode]}")

_moves_made = 0


def _ply_from_fen(fen: str) -> int:
    parts = fen.split()
    try:
        full = int(parts[5])
    except (IndexError, ValueError):
        full = 1
    return 2 * (full - 1) + (0 if parts[1] == "w" else 1)


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation for the side to move in fen."""
    global _moves_made
    t0 = time.perf_counter()
    ENGINE.stop_ponder()
    board = chess.Board(fen)
    ply = max(_ply_from_fen(fen), 2 * _moves_made)
    soft, hard = zengine.budget(time_left_ms, INCREMENT_MS, ply)
    uci = ""
    try:
        uci, score, depth = ENGINE.search(fen, soft, hard)
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            print(f"engine proposed illegal move {uci} in {fen}")
            uci = ""
        else:
            ENGINE.push_own_move(ENGINE.last_move)
            print(f"move {uci} score {score} depth {depth} time {(time.perf_counter() - t0) * 1000:.0f}ms "
                  f"soft {soft:.0f} hard {hard:.0f} left {time_left_ms}")
            board.push(mv)
            if PONDER and not board.is_game_over(claim_draw=True):
                ENGINE.start_ponder(board.fen())
    except Exception as exc:  # noqa: BLE001 - never crash the game
        print(f"search failed: {exc!r}")
        uci = ""
    if not uci:
        # last resort: a capture if one exists, else any legal move
        legal = list(board.legal_moves)
        caps = [m for m in legal if board.is_capture(m)]
        uci = random.choice(caps or legal).uci()
    _moves_made += 1
    return uci
