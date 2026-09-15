"""Engine driver: owns the arrays, runs iterative deepening under a time budget, keeps game history."""

import os
import threading
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
        size = 1 << tt_bits
        assert size == zs.TT_SIZE, "TT size is compiled into the search"
        self.tt_key = np.zeros(size, dtype=np.int64)
        self.tt_data = np.zeros(size, dtype=np.int64)
        self.sinfo = np.zeros(zs.SINFO_LEN, dtype=np.int64)
        self.counters = np.zeros((13, 120), dtype=np.int32)
        self.stk = np.zeros((zb.MAX_PLY + 2, 2), dtype=np.int32)
        self.contempt = 0
        self.ply_cap = 600
        self.game_ply = 0
        self.tinfo = np.zeros(2, dtype=np.float64)
        self.tbl_mg = ze.TBL_MG.copy()
        self.tbl_eg = ze.TBL_EG.copy()
        self.mode = 0
        self.blend = 0
        self.w1, self.b1, self.w2, self.b2 = ze.dummy_nn()
        self.acc = np.zeros((zb.MAX_PLY + 2, 2, self.w1.shape[1] + 16), dtype=np.int32)
        if nnue_path and os.path.exists(nnue_path):
            self.load_nnue(nnue_path)
        self.game_hashes: list[int] = []
        self.last_fen = ""
        self.last_move = 0
        self._ponder: threading.Thread | None = None
        self.pondering = False
        self.abort = False

    def load_nnue(self, path: str) -> None:
        z = np.load(path)
        self.w1 = np.ascontiguousarray(z["w1"].astype(np.int16))
        self.b1 = np.ascontiguousarray(z["b1"].astype(np.int16))
        self.w2 = np.ascontiguousarray(z["w2"].astype(np.int16))
        self.b2 = np.int32(int(z["b2"]))
        self.acc = np.zeros((zb.MAX_PLY + 2, 2, self.w1.shape[1] + 16), dtype=np.int32)
        self.mode = 1 if self.blend == 0 else 2

    def load_psqt(self, path: str) -> None:
        z = np.load(path)
        self.tbl_mg = np.ascontiguousarray(z["mg"].astype(np.int32))
        self.tbl_eg = np.ascontiguousarray(z["eg"].astype(np.int32))

    # --- position handling ---------------------------------------------------------------------
    def set_position(self, fen: str) -> None:
        zb.set_fen(self.bd, self.st, fen)
        self.st[zb.ST_PLY] = 0
        h = zb.compute_hash(self.bd, self.st)
        # game history: keep the positions we have seen so repetition is detected
        if self.game_hashes and self.game_hashes[-1] == h:
            pass
        else:
            self.game_hashes.append(int(h))
        if len(self.game_hashes) > 600:
            self.game_hashes = self.game_hashes[-600:]
        hl = len(self.game_hashes) - 1
        self.st[zb.ST_HISTLEN] = hl
        for i, x in enumerate(self.game_hashes):
            self.hs[i] = x
        if self.mode >= 1:
            ze.nn_refresh(self.bd, self.acc, 0, self.w1, self.b1)

    def push_own_move(self, m: int) -> None:
        """Record the position after our move so the opponent's reply is checked for repetition."""
        if self.mode >= 1:
            ze.nn_push(self.bd, self.st, m, self.acc, 0, self.w1)
        if zb.make_move(self.bd, self.st, self.undo, self.hs, m):
            self.game_hashes.append(int(self.hs[self.st[zb.ST_HISTLEN] + 1]))
            zb.unmake_move(self.bd, self.st, self.undo, m)

    def new_game(self) -> None:
        self.game_hashes = []
        self.tt_key[:] = 0
        self.tt_data[:] = 0
        self.history[:] = 0
        self.counters[:] = 0

    # --- search --------------------------------------------------------------------------------
    def search(self, fen: str, soft_ms: float, hard_ms: float, max_depth: int = 64, verbose: bool = False) -> tuple[str, int, int]:
        t0 = time.perf_counter()
        if self.abort:
            raise RuntimeError("aborted")
        self.set_position(fen)
        generation = (int(self.sinfo[zs.SI_GEN]) + 1) & 0xFF
        self.sinfo[:] = 0
        self.sinfo[zs.SI_GEN] = generation
        self.sinfo[zs.SI_ROOT_SIDE] = int(self.st[zb.ST_SIDE])
        self.sinfo[zs.SI_CONTEMPT] = int(self.contempt)
        self.sinfo[zs.SI_PLY_CAP] = max(1, min(zb.MAX_PLY, self.ply_cap - self.game_ply))
        self.stk[:] = 0
        self.tinfo[0] = 0.0 if self.abort else t0 + hard_ms / 1000.0
        self.killers[:] = 0
        self.history //= 2
        n = zb.gen_moves(self.bd, self.st, self.mvbuf[0], False)
        legal = []
        for i in range(n):
            m = int(self.mvbuf[0, i])
            if zb.make_move(self.bd, self.st, self.undo, self.hs, m):
                zb.unmake_move(self.bd, self.st, self.undo, m)
                legal.append(m)
        if not legal:
            raise RuntimeError("no legal moves")
        if len(legal) == 1:
            self.last_move = legal[0]
            return zb.move_to_uci(legal[0]), 0, 1
        best_move = legal[0]
        prev_best = 0
        best_score = 0
        depth = 1
        prev_score = 0
        last_depth = 0
        while depth <= max_depth and not self.abort:
            window = 25 if depth >= 5 else zs.INF
            alpha = max(-zs.INF, prev_score - window)
            beta = min(zs.INF, prev_score + window)
            while True:
                self.sinfo[zs.SI_ROOT_BEST] = 0
                self.st[zb.ST_PLY] = 0
                score = zs.negamax(self.bd, self.st, self.undo, self.hs, self.mvbuf, self.scbuf, depth, alpha, beta,
                                   self.sinfo, self.tinfo, self.tt_key, self.tt_data, self.killers, self.history,
                                   self.counters, self.stk, self.mode, self.tbl_mg, self.tbl_eg, self.acc,
                                   self.w1, self.w2, self.b2, True)
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
                # a root move that completed at this depth with a better score than the fallback is safe
                if rb != 0 and rb in legal and int(self.sinfo[zs.SI_ROOT_DEPTH]) >= 1:
                    best_move = rb
                    best_score = int(self.sinfo[zs.SI_ROOT_SCORE])
                break
            if rb != 0 and rb in legal:
                best_move = rb
                best_score = int(score)
            dropped = depth >= 6 and prev_score - int(score) > 40
            prev_score = int(score)
            last_depth = depth
            elapsed = (time.perf_counter() - t0) * 1000.0
            if verbose:
                pv = zs.find_pv(self.bd, self.st, self.undo, self.hs, self.tt_key, self.tt_data)
                print(f"d{depth} s={score} n={int(self.sinfo[zs.SI_NODES])} t={elapsed:.0f}ms nps={int(self.sinfo[zs.SI_NODES]) / max(elapsed, 1) * 1000:.0f} pv={' '.join(pv)}")
            if abs(score) > zs.MATE_BOUND and depth >= 4 and not self.pondering:
                break
            changed = best_move != prev_best
            prev_best = best_move
            # Another iteration usually costs about as much as everything so far, so a new one is
            # only started while there is room for it. Unstable moves or a falling score earn more.
            limit = soft_ms * (0.9 if changed else 0.5)
            if dropped:
                limit = max(limit, soft_ms * 0.8)
            if elapsed > limit:
                break
            depth += 1
        self.st[zb.ST_PLY] = 0
        self.last_move = best_move
        return zb.move_to_uci(best_move), best_score, last_depth



def budget(time_left_ms: int, increment_ms: int = 500, ply: int = 0) -> tuple[float, float]:
    """Soft/hard budgets in ms. Keeps a reserve so the clock never runs out."""
    tl = max(time_left_ms - 150, 50)
    moves_left = max(18, 40 - ply // 4)
    soft = tl / moves_left + increment_ms * 0.8
    soft = min(soft, tl * 0.25)
    hard = min(tl * 0.4, soft * 4.0)
    if tl < 3000:
        soft = min(soft, tl * 0.08)
        hard = min(hard, tl * 0.2)
    return max(soft, 5.0), max(hard, 10.0)
