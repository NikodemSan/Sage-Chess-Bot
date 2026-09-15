"""AI Chessathon entry: Sage.

A numba-jitted alpha-beta searcher (zboard / zsearch / zengine) driving an original residual
NNUE evaluation (zeval, weights.npz) trained from scratch by the team. Everything that runs is
Python from this zip plus the preinstalled stack; nothing is downloaded and no third-party engine
code or network is included.

This file is the safety layer: it owns the clock, keeps the game history the engine needs for
repetition detection, and guarantees a legal move comes back whatever happens inside the search.
"""
import os

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

import sys
import time
import traceback
from pathlib import Path

import chess

import zboard
import zengine

# --- event constants (docs page) -------------------------------------------------------------
INCREMENT_MS = 500
PLY_CAP = 600
# Draws by repetition, fifty moves or stalemate are scored this many centipawns below level
# for us: in a level position we keep playing rather than accept a half point.
CONTEMPT_CP = 20
# Round-trip reserve the referee charges outside the search itself (JSON, pipes, python-chess).
OVERHEAD_MS = 150

_START = time.perf_counter()
_HERE = Path(__file__).resolve().parent

ENGINE = zengine.Engine(str(_HERE / "weights.npz"))
ENGINE.contempt = CONTEMPT_CP
ENGINE.ply_cap = PLY_CAP

# Warm every jitted path inside the init budget: normal, in-check, en passant, promotion,
# castling, endgame. Compilation is keyed by signature, so one call per function is enough;
# the extra positions simply exercise the corresponding code before it matters.
for _fen in (
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "8/P4k2/8/8/8/8/2K5/8 w - - 0 1",
    "rnb1kbnr/pppp1ppp/8/4p3/7q/5P2/PPPPP1PP/RNBQKBNR w KQkq - 1 3",
):
    ENGINE.search(_fen, 150, 300, max_depth=6)
ENGINE.new_game()
print(f"[sage] init {time.perf_counter() - _START:.1f}s", file=sys.stderr, flush=True)


def budget(time_left_ms: int, ply: int) -> tuple[float, float]:
    """Soft and hard search limits in ms.

    The soft limit is the target spend for the move; the hard limit stops the search mid-iteration
    and is never allowed near the clock. Both fall off sharply when the clock is short, and the
    increment alone carries the game once the base time is gone."""
    tl = max(time_left_ms - OVERHEAD_MS, 30)
    moves_left = max(18, 40 - ply // 4)
    soft = tl / moves_left + INCREMENT_MS * 0.8
    soft = min(soft, tl * 0.2)
    hard = min(tl * 0.25, soft * 2.5)
    if tl < 10000:
        hard = min(hard, tl * 0.15)
    if tl < 4000:
        soft = min(soft, tl * 0.08)
        hard = min(hard, tl * 0.18)
    if tl < 600:
        soft = min(soft, 25.0)
        hard = min(hard, 50.0)
    return max(soft, 4.0), max(hard, 8.0)


def _record_history(fen: str, uci: str) -> None:
    """Keep the engine's game history in step when the move did not come from its own search."""
    try:
        ENGINE.set_position(fen)
        n = zboard.gen_moves(ENGINE.bd, ENGINE.st, ENGINE.mvbuf[0], False)
        for i in range(n):
            m = int(ENGINE.mvbuf[0, i])
            if zboard.move_to_uci(m) == uci:
                ENGINE.push_own_move(m)
                return
    except Exception:  # history is an optimisation, never worth a crash
        pass


_VAL = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}


def _material(board: chess.Board, colour: chess.Color) -> int:
    total = 0
    for piece_type, value in _VAL.items():
        total += value * (len(board.pieces(piece_type, colour)) - len(board.pieces(piece_type, not colour)))
    return total


def fallback_move(board: chess.Board, deadline: float) -> chess.Move:
    """Engine-free move choice in pure python-chess: mate in one, else the move that keeps the most
    material after the opponent's best capture reply. Runs in a few tens of milliseconds and stops
    early at the deadline."""
    legal = list(board.legal_moves)
    best = legal[0]
    best_score = -10**9
    me = board.turn
    for move in legal:
        board.push(move)
        try:
            if board.is_checkmate():
                return move
            score = _material(board, me)
            if board.is_check():
                score -= 10
            if time.perf_counter() < deadline:
                worst = 0
                for reply in board.legal_moves:
                    if not board.is_capture(reply) and not reply.promotion:
                        continue
                    board.push(reply)
                    worst = min(worst, _material(board, me) - score)
                    board.pop()
                score += worst
        finally:
            board.pop()
        if score > best_score:
            best_score = score
            best = move
        if time.perf_counter() >= deadline + 0.05:
            break
    return best


def get_move(fen: str, time_left_ms: int) -> str:
    t0 = time.perf_counter()
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        raise ValueError("get_move called in a terminal position")
    ply = board.ply()
    ENGINE.game_ply = ply
    soft, hard = budget(int(time_left_ms), ply)
    max_depth = 64 if time_left_ms >= 250 else 3
    try:
        uci, score, depth = ENGINE.search(fen, soft, hard, max_depth=max_depth)
        move = chess.Move.from_uci(uci)
        if move not in legal:
            raise RuntimeError(f"search returned illegal move {uci} in {fen}")
        ENGINE.push_own_move(ENGINE.last_move)
        spent = (time.perf_counter() - t0) * 1000.0
        print(f"[sage] ply {ply} {uci} cp {score} d{depth} {int(ENGINE.sinfo[0])}n {spent:.0f}ms left {time_left_ms}", file=sys.stderr, flush=True)
        return uci
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        move = fallback_move(board, t0 + max(0.02, min(0.15, time_left_ms / 4000.0)))
        if move not in legal:
            move = legal[0]
        uci = move.uci()
        _record_history(fen, uci)
        return uci
