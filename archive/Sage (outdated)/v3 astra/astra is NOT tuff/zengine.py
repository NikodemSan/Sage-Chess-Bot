"""Engine driver: owns the arrays, runs iterative deepening under a time budget, keeps game history."""

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
        size = 1 << tt_bits
        assert size == zs.TT_SIZE, "TT size is compiled into the search"
        self.tt_key = np.zeros(size, dtype=np.int64)
        self.tt_data = np.zeros(size, dtype=np.int64)
        self.sinfo = np.zeros(10, dtype=np.int64)
        self.contempt = 0
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
        self.last_completed_depth = 0
        self.abort = False

    def load_nnue(self, path: str) -> None:
        from sl_spec import BASELINE_HASH
        import json
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            contract = meta["contract"]
            if contract["baseline_sha256"] != BASELINE_HASH or not meta.get("trained_from_scratch"):
                raise ValueError("Incompatible Sage model contract/provenance")
            if z["w1"].shape != (6145, 256) or z["w2"].shape != (4104,):
                raise ValueError("Unexpected Sage weight dimensions")
            self.w1 = np.ascontiguousarray(z["w1"], dtype=np.int16)
            self.b1 = np.ascontiguousarray(z["b1"], dtype=np.int16)
            self.w2 = np.ascontiguousarray(z["w2"], dtype=np.int16)
            self.b2 = np.int32(int(z["b2"]))
        self.acc = np.zeros((zb.MAX_PLY + 2, 2, self.w1.shape[1] + 16), dtype=np.int32)
        self.mode = 1

    def blend_with(self, secondary_path: str, primary_share: float = 0.75) -> None:
        """Combine independent hidden units, not unrelated weight coordinates.

        The baseline is added once. Head coefficients are scaled and rounded;
        no training or claim of a new trained checkpoint is implied.
        """
        with np.load(secondary_path, allow_pickle=False) as z:
            other_w1 = np.ascontiguousarray(z["w1"], dtype=np.int16)
            other_w2 = np.ascontiguousarray(z["w2"], dtype=np.int16)
        if self.w1.shape != other_w1.shape:
            raise ValueError("Ensemble architectures differ")
        h = self.w1.shape[1]
        first = self.w2.reshape(8, 2*h+1).astype(np.float64)
        second = other_w2.reshape(8, 2*h+1).astype(np.float64)
        head = np.concatenate((primary_share*first[:,:h], (1-primary_share)*second[:,:h],
                               primary_share*first[:,h:2*h], (1-primary_share)*second[:,h:2*h],
                               primary_share*first[:,2*h:] + (1-primary_share)*second[:,2*h:]), axis=1)
        self.w1 = np.ascontiguousarray(np.concatenate((self.w1, other_w1), axis=1))
        self.w2 = np.ascontiguousarray(np.rint(head).astype(np.int16).ravel())
        self.acc = np.zeros((zb.MAX_PLY + 2, 2, 2*h + 16), dtype=np.int32)

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
        self.killers[:] = 0
        self.last_move = 0
        self.last_completed_depth = 0

    # --- search --------------------------------------------------------------------------------
    def search(self, fen: str, soft_ms: float, hard_ms: float, max_depth: int = 64, verbose: bool = False) -> tuple[str, int, int]:
        self.last_completed_depth = 0
        t0 = time.perf_counter()
        if self.abort:
            raise RuntimeError("aborted")
        self.set_position(fen)
        generation = (int(self.sinfo[zs.SI_GEN]) + 1) & 0xFF
        self.sinfo[:] = 0
        self.sinfo[zs.SI_GEN] = generation
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
            self.last_completed_depth = 1
            self.last_move = legal[0]
            return zb.move_to_uci(legal[0]), 0, 1
        best_move = legal[0]
        prev_best = 0
        best_score = 0
        depth = 1
        prev_score = 0
        stable_depths = 0
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
                                   self.mode, self.tbl_mg, self.tbl_eg, self.acc, self.w1, self.w2, self.b2, True)
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
                # Retain the supplied engine's useful completed root branch
                # when a deeper iteration runs out of time.
                if rb != 0 and rb in legal and int(self.sinfo[zs.SI_ROOT_DEPTH]) >= 1:
                    best_move = rb
                    best_score = int(self.sinfo[zs.SI_ROOT_SCORE])
                break
            if rb != 0 and rb in legal:
                best_move = rb
                best_score = int(score)
            score_drop = max(0, prev_score - int(score)) if last_depth else 0
            prev_score = int(score)
            last_depth = depth
            elapsed = (time.perf_counter() - t0) * 1000.0
            if verbose:
                print(f"d{depth} score={score} nodes={int(self.sinfo[zs.SI_NODES])} ms={elapsed:.0f}", flush=True)
            if abs(score) > zs.MATE_BOUND and depth >= 4:
                break
            changed = best_move != prev_best
            stable_depths = 0 if changed else stable_depths + 1
            prev_best = best_move
            factor = 1.20 if changed and depth >= 4 else (0.75 if stable_depths >= 3 else 1.0)
            if score_drop >= 60:
                factor = max(factor, 1.5)
            limit = min(soft_ms * factor, hard_ms * 0.8)
            if elapsed >= limit or elapsed >= hard_ms * 0.75:
                break
            depth += 1
        self.st[zb.ST_PLY] = 0
        self.last_completed_depth = last_depth
        self.last_move = best_move
        return zb.move_to_uci(best_move), best_score, last_depth



def budget(time_left_ms: int, increment_ms: int = 500, ply: int = 0) -> tuple[float, float]:
    """Spend on useful search, with a clock reserve and bounded tactical extensions.

    Increment arrives AFTER returning; it is never treated as spendable cash.
    Pondering is unavailable. Budgets include search setup; API overhead gets
    a separate reserve. Frequent deadline checks bound ordinary overrun.
    """
    remaining = max(0.0, float(time_left_ms))
    reserve = min(180.0, max(25.0, remaining * 0.025))
    usable = max(0.0, remaining - reserve)
    if usable <= 0:
        return 0.0, 0.0
    moves_left = max(22, 38 - ply // 8)
    soft = usable / moves_left + 0.72 * increment_ms
    soft = min(soft, usable * 0.16)
    hard = min(usable * 0.32, soft * 2.6)
    if remaining < 3000:
        soft = min(soft, usable * 0.10, 220.0)
        hard = min(hard, usable * 0.20, 380.0)
    return max(0.0, soft), max(0.0, hard)
