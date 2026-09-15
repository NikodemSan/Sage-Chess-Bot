"""AI Chessathon submission entrypoint.

The platform imports this file once per game and calls get_move(fen, time_left_ms). Expensive
Numba compilation and competition-opening pre-analysis are performed during the 90-second init
window. The trained NNUE and tuned PSQT files are loaded unchanged. Pondering is deliberately
absent because the current competition suspends our process while the opponent moves.
"""

import os
import random
import sys
import time
import warnings

_t_import = time.perf_counter()
warnings.filterwarnings("ignore", message=".*object mode.*")

import chess  # noqa: E402

import zengine  # noqa: E402
from zopenings import SEED_POSITIONS  # noqa: E402
from zbook import BOOK, fen_key  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _find(name: str) -> str | None:
    for cand in (os.path.join(HERE, "weights", name), os.path.join(HERE, name)):
        if os.path.exists(cand):
            return cand
    return None


NNUE_PATH = _find("nnue.npz")
PSQT_PATH = _find("psqt.npz")
INCREMENT_MS = 500

ENGINE = zengine.Engine(NNUE_PATH)
if PSQT_PATH:
    ENGINE.load_psqt(PSQT_PATH)
if ENGINE.mode >= 1:
    # Preserve the exact evaluation used by the trained submission: same NNUE, same PSQT,
    # same 5:3 blend. Search/rule/clock changes do not alter or reinterpret the weights.
    ENGINE.blend = 1
    ENGINE.mode = 2

# Compile the hot Numba paths during the free init budget.  These positions exercise castling,
# tactical/check search, and an endgame.  Keep this phase short because platform hardware can be
# slower than the development host.
for _fen in (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
):
    try:
        ENGINE.search(_fen, 45, 90, max_depth=6, record_history=False)
    except Exception:
        # Warm-up must never jeopardise the init handshake. Normal search still has a legality
        # guard and last-resort move path below.
        pass
ENGINE.new_game()

# Public tournament pages repeatedly show a small family of curated openings, and a public
# balanced-opening corpus contains many matching named FENs. The corpus is a plausible analogue,
# not a proven exact source. Use part of the otherwise-free init window to analyse those positions
# with Fable's own evaluator/search and leave the resulting entries in the persistent TT.
#
# Keep this deliberately shallow. The offline Fable-native book already covers the first three of
# our moves around these starts, so deep import-time searches add little while increasing the risk
# of a pathological position consuming the 90-second initialization budget. 250 ms hard per seed
# plus a 55-second absolute cutoff gives a wide safety margin after Numba compilation.
_seeded = 0
for _name, _fen in SEED_POSITIONS:
    if time.perf_counter() - _t_import >= 55.0:
        break
    try:
        ENGINE.search(_fen, 150, 250, max_depth=12, record_history=False)
        _seeded += 1
    except Exception:
        pass
# Clear hypothetical game history/heuristics but retain the TT knowledge learned above.
ENGINE.new_game(clear_tt=False)
print(
    f"init done in {time.perf_counter() - _t_import:.1f}s, seeded {_seeded}/{len(SEED_POSITIONS)} "
    f"likely starts, {len(BOOK)} prepared book positions, eval mode "
    f"{('classical', 'nnue', 'nnue+psqt blend')[ENGINE.mode]}",
    file=sys.stderr,
)

# Exact 3-piece Syzygy WDL guard. The current rules explicitly permit tablebases. We ship the
# complete orthodox 3-piece WDL set (K+P/R/Q/B/N vs K). It is used only to veto a move that would
# throw away the theoretical W/D/L result; Fable's own search still chooses among optimal moves.
_TB = None
try:
    import chess.syzygy as _syzygy  # noqa: E402

    _tb_dir = os.path.join(HERE, "syzygy")
    if os.path.isdir(_tb_dir):
        _TB = _syzygy.open_tablebase(_tb_dir)
except Exception as _tb_exc:  # noqa: BLE001
    print(f"tablebase disabled: {_tb_exc!r}", file=sys.stderr)
    _TB = None


def _tb_optimal_moves(board: chess.Board) -> set[str] | None:
    """Return exact WDL-optimal UCI moves for covered 3-piece positions, else None."""
    if _TB is None or len(board.piece_map()) != 3 or board.castling_rights:
        return None
    try:
        _TB.probe_wdl(board)  # verify the root material class is covered
    except Exception:
        return None
    best = -3
    out: set[str] = set()
    for mv in list(board.legal_moves):
        board.push(mv)
        try:
            # A capture can reduce a covered 3-piece ending to bare kings. Handle terminal children
            # directly instead of requiring a 2-piece table file. Otherwise Syzygy WDL is from the
            # child side-to-move's view, so negate it for the mover.
            outcome = board.outcome(claim_draw=False)
            if outcome is not None:
                if outcome.winner is None:
                    value = 0
                else:
                    value = 2 if outcome.winner != board.turn else -2
            else:
                value = -int(_TB.probe_wdl(board))
        except Exception:
            board.pop()
            continue
        board.pop()
        if value > best:
            best = value
            out = {mv.uci()}
        elif value == best:
            out.add(mv.uci())
    return out or None


_moves_made = 0


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation for the side to move in *fen*."""
    global _moves_made
    t0 = time.perf_counter()
    board = chess.Board(fen)

    # Tournament-specific opening preparation: only the first two of our moves are eligible, so a
    # later repetition of the same board cannot accidentally reuse an opening move under a very
    # different halfmove clock. Every entry was generated offline by this exact Fable build.
    if _moves_made <= 2:
        prepared = BOOK.get(fen_key(fen))
        if prepared:
            try:
                pm = chess.Move.from_uci(prepared)
                if pm in board.legal_moves and ENGINE.record_uci_move(fen, prepared):
                    _moves_made += 1
                    print(f"book {prepared} time {(time.perf_counter()-t0)*1000:.1f}ms", file=sys.stderr)
                    return prepared
            except Exception as exc:  # noqa: BLE001
                print(f"book miss/failure {prepared}: {exc!r}", file=sys.stderr)

    legal_count = board.legal_moves.count()
    soft, hard = zengine.budget(
        time_left_ms,
        INCREMENT_MS,
        moves_made=_moves_made,
        legal_count=legal_count,
        in_check=board.is_check(),
    )
    tb_moves = _tb_optimal_moves(board)
    if tb_moves:
        # Exact WDL means there is little value spending a normal multi-second budget here. Search
        # briefly so the trained evaluator/search chooses naturally among result-preserving moves.
        soft = min(soft, 70.0)
        hard = min(hard, 130.0)

    uci = ""
    recorded = False
    try:
        uci, score, depth = ENGINE.search(fen, soft, hard)
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            print(f"engine proposed illegal move {uci} in {fen}", file=sys.stderr)
            uci = ""
        else:
            if tb_moves and uci not in tb_moves:
                # Search may not understand a theoretical edge case. Preserve the exact WDL result
                # instead. Deterministic order keeps runs reproducible; all candidates are exact.
                uci = sorted(tb_moves)[0]
                mv = chess.Move.from_uci(uci)
                ENGINE.record_uci_move(fen, uci)
                recorded = True
                print(f"tablebase veto -> {uci}", file=sys.stderr)
            if not recorded:
                # search() leaves the root restored and last_move set to the returned internal move.
                ENGINE.push_own_move(ENGINE.last_move)
                recorded = True
            print(
                f"move {uci} score {score} depth {depth} "
                f"time {(time.perf_counter() - t0) * 1000:.0f}ms "
                f"soft {soft:.0f} hard {hard:.0f} left {time_left_ms}",
                file=sys.stderr,
            )
    except Exception as exc:  # noqa: BLE001 - a search failure must not forfeit the game
        print(f"search failed: {exc!r}", file=sys.stderr)
        uci = ""

    if not uci:
        # Last resort: prefer a capture, otherwise any legal move. Also record this fallback in
        # the engine's repetition history so one exception cannot poison later draw detection.
        legal = list(board.legal_moves)
        if not legal:
            # The referee should never call get_move on a terminal position, but avoid a cryptic
            # random.choice error if it does.
            raise RuntimeError("get_move called with no legal moves")
        caps = [m for m in legal if board.is_capture(m)]
        uci = random.choice(caps or legal).uci()

    if not recorded:
        try:
            ENGINE.record_uci_move(fen, uci)
        except Exception as exc:  # noqa: BLE001
            print(f"failed to record fallback {uci}: {exc!r}", file=sys.stderr)

    _moves_made += 1
    return uci
