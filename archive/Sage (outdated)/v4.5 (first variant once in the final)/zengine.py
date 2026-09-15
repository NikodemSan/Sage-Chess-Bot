"""Engine driver: owns the arrays, runs iterative deepening under a time budget, keeps game history."""

import os
import threading
import time

import numpy as np

import zboard as zb
import zeval as ze
import zsearch as zs

HERE = os.path.dirname(os.path.abspath(__file__))

ANTI_REPEAT = os.environ.get("SAGE_STAGE3ER2_ANTI_REPEAT", "1") == "1"
ANTI_REPEAT_MIN_SCORE = 150
ANTI_REPEAT_RESCUE_FLOOR = 100
ANTI_REPEAT_MIN_DEPTH = 6
ANTI_REPEAT_MAX_DEPTH = 8
ANTI_REPEAT_MAX_MS = 80.0


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
        self.sinfo = np.zeros(11, dtype=np.int64)
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
        self._ponder: threading.Thread | None = None
        self.pondering = False
        self.abort = False
        self.last_info = {}

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

    def _move_revisits_history(self, m: int) -> bool:
        """Check one legal root move against already-seen game positions."""
        self.st[zb.ST_PLY] = 0
        if not zb.make_move(self.bd, self.st, self.undo, self.hs, int(m)):
            return False
        child_hash = int(self.hs[int(self.st[zb.ST_HISTLEN]) + 1])
        half = int(self.st[zb.ST_HALF])
        zb.unmake_move(self.bd, self.st, self.undo, int(m))
        if half <= 0:
            return False
        # Equal Zobrist hashes include side/castling/EP state. Restricting to the
        # reversible window mirrors the recursive repetition detector.
        start = max(0, len(self.game_hashes) - half - 1)
        for h in self.game_hashes[start:-1]:
            if int(h) == child_hash:
                return True
        return False

    def _has_fresh_root_move(self, legal: list[int]) -> bool:
        for m in legal:
            if not self._move_revisits_history(int(m)):
                return True
        return False

    def new_game(self) -> None:
        self.game_hashes = []
        self.tt_key[:] = 0
        self.tt_data[:] = 0
        self.history[:] = 0

    # --- search --------------------------------------------------------------------------------
    def search(self, fen: str, soft_ms: float, hard_ms: float, max_depth: int = 64, verbose: bool = False,
               avoid_revisit: bool = False, allow_rescue: bool = True) -> tuple[str, int, int]:
        t0 = time.perf_counter()
        if self.abort:
            raise RuntimeError("aborted")
        self.set_position(fen)
        generation = (int(self.sinfo[zs.SI_GEN]) + 1) & 0xFF
        self.sinfo[:] = 0
        self.sinfo[zs.SI_GEN] = generation
        self.sinfo[zs.SI_AVOID_REVISIT] = 1 if avoid_revisit else 0
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
            self.last_info={'depth':0,'score_cp':None,'forced_move':True,'fallback':False,
                            'iterations':[],'stats':{'nodes':int(self.sinfo[0]),'stop':0}}
            return zb.move_to_uci(legal[0]), 0, 1
        best_move = legal[0]
        prev_best = 0
        best_score = 0
        depth = 1
        prev_score = 0
        last_depth = 0
        iterations=[]
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
                # a root move that completed at this depth with a better score than the fallback is safe
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
            iterations.append({'depth':depth,'move':zb.move_to_uci(best_move),'score_cp':best_score,
                               'nodes':int(self.sinfo[0]),'ms':round(elapsed,3)})
            if verbose:
                pv = zs.find_pv(self.bd, self.st, self.undo, self.hs, self.tt_key, self.tt_data)
                print(f"d{depth} s={score} n={int(self.sinfo[zs.SI_NODES])} t={elapsed:.0f}ms nps={int(self.sinfo[zs.SI_NODES]) / max(elapsed, 1) * 1000:.0f} pv={' '.join(pv)}")
            if abs(score) > zs.MATE_BOUND and depth >= 4 and not self.pondering:
                break
            changed = best_move != prev_best
            prev_best = best_move
            limit = soft_ms * (0.9 if changed else 0.5)
            if elapsed > limit:
                break
            depth += 1
        self.st[zb.ST_PLY] = 0
        self.last_move = best_move
        normal_nodes = int(self.sinfo[zs.SI_NODES])
        normal_info={'depth':last_depth,'score_cp':best_score,'forced_move':False,
                     'fallback':last_depth==0,'iterations':iterations,
                     'partial_iteration_used':bool(self.sinfo[zs.SI_STOP] and rb!=0 and rb in legal),
                     'stats':{'nodes':normal_nodes,'stop':int(self.sinfo[zs.SI_STOP]),
                              'seldepth':int(self.sinfo[zs.SI_SELDEPTH])}}
        self.last_info = normal_info

        # Normal path is untouched. Only spend extra work when the completed
        # search actually wants to revisit history while it believes we are
        # clearly ahead. Immediate third repetitions already score as draws in
        # negamax, so this targets the earlier (second-occurrence) cycle.
        if (ANTI_REPEAT and allow_rescue and not avoid_revisit and
                last_depth >= ANTI_REPEAT_MIN_DEPTH and best_score >= ANTI_REPEAT_MIN_SCORE and
                int(self.st[zb.ST_HALF]) >= 4 and
                self._move_revisits_history(best_move) and self._has_fresh_root_move(legal)):
            remaining_ms = max(0.0, (self.tinfo[0] - time.perf_counter()) * 1000.0)
            rescue_hard = min(ANTI_REPEAT_MAX_MS, remaining_ms * 0.8)
            if rescue_hard >= 12.0:
                rescue_soft = max(8.0, rescue_hard * 0.55)
                rescue_depth = min(ANTI_REPEAT_MAX_DEPTH, last_depth)
                normal_move, normal_score, normal_depth = best_move, best_score, last_depth
                rt0 = time.perf_counter()
                rmove_uci, rscore, rdepth = self.search(
                    fen, rescue_soft, rescue_hard, max_depth=rescue_depth,
                    verbose=False, avoid_revisit=True, allow_rescue=False)
                rescue_ms = (time.perf_counter() - rt0) * 1000.0
                rescue_nodes = int(self.sinfo[zs.SI_NODES])
                rmove = self.last_move
                accepted = (rdepth >= ANTI_REPEAT_MIN_DEPTH and rscore >= ANTI_REPEAT_RESCUE_FLOOR)
                selected_move = rmove if accepted else normal_move
                selected_score = int(rscore) if accepted else int(normal_score)
                selected_depth = int(rdepth) if accepted else int(normal_depth)
                self.last_move = selected_move
                self.sinfo[zs.SI_NODES] = normal_nodes + rescue_nodes
                self.last_info = dict(normal_info)
                self.last_info['anti_repeat']={
                    'triggered':True,'accepted':bool(accepted),
                    'normal_move':zb.move_to_uci(normal_move),'normal_score_cp':int(normal_score),
                    'normal_depth':int(normal_depth),'rescue_move':rmove_uci,
                    'rescue_score_cp':int(rscore),'rescue_depth':int(rdepth),
                    'rescue_nodes':rescue_nodes,'rescue_ms':round(rescue_ms,3)}
                self.last_info['stats']=dict(normal_info['stats'])
                self.last_info['stats']['nodes']=normal_nodes + rescue_nodes
                return zb.move_to_uci(selected_move), selected_score, selected_depth

        self.last_info['anti_repeat']={'triggered':False,'accepted':False}
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
