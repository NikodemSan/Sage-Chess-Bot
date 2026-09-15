"""Engine driver: owns arrays, iterative deepening, time management and game history."""

import os
import time

import numpy as np

import zboard as zb
import zeval as ze
import zsearch as zs

HERE = os.path.dirname(os.path.abspath(__file__))


class Engine:
    def __init__(self, nnue_path: str | None = None, tt_bits: int = zs.TT_BITS) -> None:
        self.bd, self.st, self.undo, self.hs = zb.new_state()
        self.mvbuf = np.zeros((zb.MAX_PLY + 2, 256), dtype=np.int32)
        self.scbuf = np.zeros((zb.MAX_PLY + 2, 256), dtype=np.int32)
        self.killers = np.zeros((zb.MAX_PLY + 2, 2), dtype=np.int32)
        self.history = np.zeros((13, 120), dtype=np.int32)
        # Piece x destination x captured-piece-type: ~22 KiB as int16, small enough to stay cache-friendly
        # and far below the tournament memory limit. It is touched only for main-search captures.
        self.capture_history = np.zeros((13, 120, 7), dtype=np.int16)
        size = 1 << tt_bits
        assert size == zs.TT_SIZE, "TT size is compiled into the search"
        self.tt_key = np.zeros(size, dtype=np.int64)
        self.tt_data = np.zeros(size, dtype=np.int64)
        self.eval_key = np.zeros(zs.EVAL_SIZE, dtype=np.int64)
        self.eval_val = np.zeros(zs.EVAL_SIZE, dtype=np.int32)
        self.sinfo = np.zeros(11, dtype=np.int64)
        # Keep draw values neutral.  The previous build's intended +15 contempt was never wired
        # into sinfo, so enabling it here would silently change playing style without A/B evidence.
        self.contempt = 0
        self.tt_generation = 0
        self.tinfo = np.zeros(2, dtype=np.float64)
        self.tbl_mg = ze.TBL_MG.copy()
        self.tbl_eg = ze.TBL_EG.copy()
        self.mode = 0
        self.blend = 0
        self.w1, self.b1, self.w2, self.b2 = ze.dummy_nn()
        self.acc = np.zeros((zb.MAX_PLY + 2, 2, self.w1.shape[1]), dtype=np.int16)
        if nnue_path and os.path.exists(nnue_path):
            self.load_nnue(nnue_path)
        self.game_hashes: list[int] = []
        self.last_move = 0

    def load_nnue(self, path: str) -> None:
        z = np.load(path)
        self.w1 = np.ascontiguousarray(z["w1"].astype(np.int16))
        self.b1 = np.ascontiguousarray(z["b1"].astype(np.int16))
        self.w2 = np.ascontiguousarray(z["w2"].astype(np.int16))
        self.b2 = np.int32(int(z["b2"]))
        self.acc = np.zeros((zb.MAX_PLY + 2, 2, self.w1.shape[1]), dtype=np.int16)
        self.mode = 1 if self.blend == 0 else 2

    def load_psqt(self, path: str) -> None:
        z = np.load(path)
        self.tbl_mg = np.ascontiguousarray(z["mg"].astype(np.int32))
        self.tbl_eg = np.ascontiguousarray(z["eg"].astype(np.int32))

    # --- position handling ---------------------------------------------------------------------
    def set_position(self, fen: str, *, record_history: bool = True) -> None:
        zb.set_fen(self.bd, self.st, fen)
        self.st[zb.ST_PLY] = 0
        h = zb.compute_hash(self.bd, self.st)
        if record_history:
            # Keep every real game position once. Search() can be re-entered on the same root
            # during testing, so do not duplicate an unchanged root hash.
            if not self.game_hashes or self.game_hashes[-1] != h:
                self.game_hashes.append(int(h))
            if len(self.game_hashes) > 600:
                self.game_hashes = self.game_hashes[-600:]
            hl = len(self.game_hashes) - 1
            self.st[zb.ST_HISTLEN] = hl
            for i, x in enumerate(self.game_hashes):
                self.hs[i] = x
        else:
            # Import-time opening analysis uses hypothetical searches, not game
            # history.  Start them from a clean root so they cannot create false repetitions.
            self.st[zb.ST_HISTLEN] = 0
            self.hs[0] = h
        if self.mode >= 1:
            ze.nn_refresh(self.bd, self.acc, 0, self.w1, self.b1)

    def push_own_move(self, m: int) -> None:
        """Record the real position after our move for future repetition detection."""
        if m == 0:
            return
        if self.mode >= 1:
            ze.nn_push(self.bd, self.st, m, self.acc, 0, self.w1)
        if zb.make_move(self.bd, self.st, self.undo, self.hs, m):
            self.game_hashes.append(int(self.hs[self.st[zb.ST_HISTLEN] + 1]))
            if len(self.game_hashes) > 600:
                self.game_hashes = self.game_hashes[-600:]
            zb.unmake_move(self.bd, self.st, self.undo, m)

    def record_uci_move(self, fen: str, uci: str) -> bool:
        """Safely record a legal UCI move, used by the protocol fallback path."""
        self.set_position(fen)
        n = zb.gen_moves(self.bd, self.st, self.mvbuf[0], False)
        for i in range(n):
            m = int(self.mvbuf[0, i])
            if zb.move_to_uci(m) != uci:
                continue
            if zb.make_move(self.bd, self.st, self.undo, self.hs, m):
                zb.unmake_move(self.bd, self.st, self.undo, m)
                self.last_move = m
                self.push_own_move(m)
                return True
        return False

    def new_game(self, *, clear_tt: bool = True) -> None:
        self.game_hashes = []
        if clear_tt:
            self.tt_key[:] = 0
            self.tt_data[:] = 0
            self.eval_key[:] = 0
            self.eval_val[:] = 0
            self.tt_generation = 0
        self.history[:] = 0
        self.capture_history[:] = 0
        self.killers[:] = 0
        self.last_move = 0

    # --- search --------------------------------------------------------------------------------
    def search(
        self,
        fen: str,
        soft_ms: float,
        hard_ms: float,
        max_depth: int = 64,
        verbose: bool = False,
        *,
        record_history: bool = True,
    ) -> tuple[str, int, int]:
        t0 = time.perf_counter()
        self.set_position(fen, record_history=record_history)
        self.sinfo[:] = 0
        self.tinfo[0] = t0 + hard_ms / 1000.0
        # The clock check itself crosses Numba objmode, so do it often only when the hard budget
        # is genuinely tight.  Masks are 2^n-1, making the hot-path test a single bitwise op.
        if hard_ms <= 200.0:
            # In flag-danger territory prefer a few percent less NPS over overshooting the clock.
            self.sinfo[zs.SI_TIME_MASK] = 31
        elif hard_ms <= 500.0:
            self.sinfo[zs.SI_TIME_MASK] = 63
        elif hard_ms <= 1500.0:
            self.sinfo[zs.SI_TIME_MASK] = 255
        else:
            self.sinfo[zs.SI_TIME_MASK] = 511
        self.tt_generation = (self.tt_generation + 1) & 0xFF
        if self.tt_generation == 0:
            self.tt_generation = 1
        self.sinfo[zs.SI_GEN] = self.tt_generation
        self.sinfo[zs.SI_ROOT_SIDE] = int(self.st[zb.ST_SIDE])
        self.sinfo[zs.SI_CONTEMPT] = int(self.contempt)
        self.killers[:] = 0
        self.history //= 2
        # Retain useful capture-order knowledge across our moves, but age it aggressively because
        # each curated game starts from a different tactical context and the process is fresh.
        self.capture_history //= 2

        n = zb.gen_moves(self.bd, self.st, self.mvbuf[0], False)
        legal: list[int] = []
        for i in range(n):
            m = int(self.mvbuf[0, i])
            if zb.make_move(self.bd, self.st, self.undo, self.hs, m):
                zb.unmake_move(self.bd, self.st, self.undo, m)
                legal.append(m)
        if not legal:
            raise RuntimeError("no legal moves")
        if len(legal) == 1:
            # Critical state fix: the caller records last_move after every returned move.
            self.last_move = legal[0]
            return zb.move_to_uci(legal[0]), 0, 1

        best_move = legal[0]
        prev_best = 0
        best_score = 0
        depth = 1
        prev_score = 0
        last_depth = 0

        while depth <= max_depth:
            old_prev_score = prev_score
            window = 25 if depth >= 5 else zs.INF
            alpha = max(-zs.INF, prev_score - window)
            beta = min(zs.INF, prev_score + window)
            while True:
                self.sinfo[zs.SI_ROOT_BEST] = 0
                self.st[zb.ST_PLY] = 0
                score = zs.negamax(
                    self.bd,
                    self.st,
                    self.undo,
                    self.hs,
                    self.mvbuf,
                    self.scbuf,
                    depth,
                    alpha,
                    beta,
                    self.sinfo,
                    self.tinfo,
                    self.eval_key,
                    self.eval_val,
                    self.tt_key,
                    self.tt_data,
                    self.killers,
                    self.history,
                    self.capture_history,
                    self.mode,
                    self.tbl_mg,
                    self.tbl_eg,
                    self.acc,
                    self.w1,
                    self.w2,
                    self.b2,
                    True,
                )
                if self.sinfo[zs.SI_STOP]:
                    break
                if score <= alpha:
                    alpha = max(-zs.INF, alpha - window * 3)
                    window *= 3
                    continue
                if score >= beta:
                    beta = min(zs.INF, beta + window * 3)
                    window *= 3
                    continue
                break

            rb = int(self.sinfo[zs.SI_ROOT_BEST])
            if self.sinfo[zs.SI_STOP]:
                # A root move that completed before the interrupt is safer than the fallback from
                # the previous depth, but only if it is one of the verified legal root moves.
                if rb != 0 and rb in legal and int(self.sinfo[zs.SI_ROOT_DEPTH]) >= 1:
                    best_move = rb
                    best_score = int(self.sinfo[zs.SI_ROOT_SCORE])
                break

            if rb != 0 and rb in legal:
                best_move = rb
                best_score = int(score)
            prev_score = int(score)
            last_depth = depth
            elapsed = (time.perf_counter() - t0) * 1000.0

            if verbose:
                pv = zs.find_pv(self.bd, self.st, self.undo, self.hs, self.tt_key, self.tt_data)
                n_nodes = int(self.sinfo[zs.SI_NODES])
                print(
                    f"d{depth} s={score} n={n_nodes} t={elapsed:.0f}ms "
                    f"nps={n_nodes / max(elapsed, 1) * 1000:.0f} pv={' '.join(pv)}"
                )

            if abs(score) > zs.MATE_BOUND and depth >= 4:
                break

            changed = best_move != prev_best
            swing = depth >= 5 and abs(int(score) - int(old_prev_score)) >= 55
            prev_best = best_move

            # Bank time on easy/stable positions, but spend the full soft budget when the PV or
            # score is unstable.  This is especially valuable at 120s+0.5s because quick easy
            # moves preserve the increment for genuinely difficult positions later.
            if depth >= 5:
                # Once the PV/score is unstable, the extra completed iteration is worth more
                # than banking a small amount of clock. Stable positions remain close to the old
                # engine's proven 0.5x soft-stop behavior.
                limit = soft_ms * (1.18 if (changed or swing) else 0.52)
                if elapsed >= limit:
                    break
            depth += 1

        self.st[zb.ST_PLY] = 0
        self.last_move = best_move
        return zb.move_to_uci(best_move), best_score, last_depth


def budget(
    time_left_ms: int,
    increment_ms: int = 500,
    *,
    moves_made: int = 0,
    legal_count: int = 20,
    in_check: bool = False,
) -> tuple[float, float]:
    """Return soft/hard search budgets in milliseconds.

    The game starts from a curated FEN, so its FEN fullmove number is not a measure of how much of
    *our* 120-second clock has already been spent.  Budget from moves made by this process instead.
    """
    tl = max(float(time_left_ms) - 160.0, 40.0)  # protocol + Python safety reserve

    # Typical games still have dozens of our moves remaining even when the opening FEN starts at
    # move 10-20.  Taper only as this process actually plays moves.
    moves_left = max(17.0, 37.0 - 0.48 * float(moves_made))
    soft = tl / moves_left + float(increment_ms) * 0.78
    # The first few moves are from a curated, fresh-clock position and seed a persistent TT that
    # later calls can reuse.  A small up-front investment is unusually useful.
    if moves_made < 3 and tl > 30000.0:
        soft *= 1.10

    # Spend a little more in forcing/high-branching positions and bank time in nearly forced ones.
    if in_check or legal_count >= 36:
        soft *= 1.12
    elif legal_count <= 8:
        soft *= 0.90

    soft = min(soft, tl * 0.24)
    hard = min(tl * 0.40, max(soft * 3.8, soft + 30.0))

    # In clock trouble, deliberately move well inside the increment so a sequence of easy moves
    # can rebuild the clock instead of flagging while chasing one extra ply.
    if tl < 5000.0:
        soft = min(soft, max(90.0, tl * 0.07))
        hard = min(hard, max(180.0, tl * 0.17))
    if tl < 1200.0:
        soft = min(soft, max(35.0, tl * 0.05))
        hard = min(hard, max(80.0, tl * 0.13))

    return max(soft, 5.0), max(hard, 10.0)
