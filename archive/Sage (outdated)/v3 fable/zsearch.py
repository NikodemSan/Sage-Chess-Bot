"""Search: iterative deepening, aspiration windows, PVS alpha-beta with a transposition table,
null-move pruning, reverse futility, futility, late move reductions and pruning, killers,
history, check extension and a capture quiescence search with delta pruning.

Time is checked against a wall-clock deadline every 2048 nodes through numba's objmode.
"""

import math
import time

import numpy as np
from numba import int8, int16, int32, int64, float64, boolean, njit, objmode

from zboard import (
    BK, BP, EMPTY, F_CAPTURE, F_EP, M_FROM, M_PROMO, M_TO, PIECE_TYPE, PIECE_VAL, SQ120, ST_HALF,
    ST_HISTLEN, ST_KSQ_B, ST_KSQ_W, ST_PLY, ST_SIDE, WHITE, WK, WP, WN, WB, WR, WQ, OFF, N_OFF, K_OFF,
    attacked, gen_moves, make_move, make_null, unmake_move, unmake_null,
)
from zeval import evaluate, nn_push

MATE = 30000
MATE_BOUND = 29000
INF = 32000
DRAW = 0

TT_BITS = 24
TT_SIZE = 1 << TT_BITS
TT_MASK = TT_SIZE - 1
TT_EXACT, TT_LOWER, TT_UPPER = 1, 2, 3

# sinfo layout
SI_NODES, SI_STOP, SI_CHECK, SI_ROOT_BEST, SI_ROOT_SCORE, SI_GEN, SI_SELDEPTH, SI_ROOT_DEPTH, SI_ROOT_SIDE, SI_CONTEMPT, SI_PLY_CAP = range(11)
SINFO_LEN = 16

LMR = np.zeros((64, 64), dtype=np.int32)
for _d in range(1, 64):
    for _m in range(1, 64):
        LMR[_d, _m] = int(0.75 + math.log(_d) * math.log(_m) / 2.25)

MVV_LVA = np.zeros((13, 13), dtype=np.int32)
for _v in range(1, 13):
    for _a in range(1, 13):
        MVV_LVA[_v, _a] = PIECE_VAL[_v] * 10 - PIECE_TYPE[_a]


@njit(cache=False, nogil=True, error_model="numpy")
def tt_pack(move, score, depth, flag, gen):
    return (np.int64(move) & 0x1FFFFF) | (np.int64(score + 32768) << 21) | (np.int64(depth) << 37) | (np.int64(flag) << 45) | (np.int64(gen) << 47)


@njit(cache=False, nogil=True, error_model="numpy")
def tt_store(tt_key, tt_data, key, move, score, depth, flag, ply, gen):
    idx = key & TT_MASK
    if score > MATE_BOUND:
        score += ply
    elif score < -MATE_BOUND:
        score -= ply
    old = tt_data[idx]
    if depth == 0 and tt_key[idx] != 0 and ((old >> 47) & 0xFF) == gen and ((old >> 37) & 0xFF) >= 3:
        return  # a quiescence bound never evicts a deep entry of the current search
    if tt_key[idx] == key:
        # same position: keep the deeper entry unless it is from an older search or we have exact
        old_depth = (old >> 37) & 0xFF
        old_gen = (old >> 47) & 0xFF
        if old_gen == gen and old_depth > depth + 2 and flag != TT_EXACT:
            return
        if move == 0:
            move = int(old & 0x1FFFFF)
    tt_key[idx] = key
    tt_data[idx] = tt_pack(move, score, depth, flag, gen)


@njit(cache=False, nogil=True, error_model="numpy")
def draw_score(sinfo, side):
    """Draw from the point of view of `side`: the root side dislikes draws by the contempt amount."""
    c = sinfo[SI_CONTEMPT]
    return -c if side == sinfo[SI_ROOT_SIDE] else c


@njit(cache=False, nogil=True, error_model="numpy")
def is_repetition(hs, hl, ply, half):
    """Draw if the position already occurred inside the search tree (twofold), or twice in the
    game history (the referee claims the actual third occurrence). Null branches are approximate."""
    matches = 0
    cur = hs[hl + ply]
    i = ply - 2
    lim = ply - half
    if lim < -hl:
        lim = -hl
    while i >= lim:
        if hs[hl + i] == cur:
            if i > 0:
                return True
            matches += 1
            if matches >= 2:
                return True
        i -= 2
    return False


@njit(cache=False, nogil=True, error_model="numpy")
def least_valuable_attacker(bd, sq, side):
    """Square of the least valuable piece of `side` attacking sq on the current board, or -1.
    Order: pawn, knight, bishop, rook, queen, king. Sliders see through squares emptied by see()."""
    if side == WHITE:
        if bd[sq - 11] == WP:
            return sq - 11
        if bd[sq - 9] == WP:
            return sq - 9
        off = 0
    else:
        if bd[sq + 11] == BP:
            return sq + 11
        if bd[sq + 9] == BP:
            return sq + 9
        off = 6
    kn = WN + off
    for i in range(8):
        if bd[sq + N_OFF[i]] == kn:
            return sq + N_OFF[i]
    bi = WB + off
    ro = WR + off
    qu = WQ + off
    kg = WK + off
    qsq = -1
    rsq = -1
    for i in range(8):
        d = K_OFF[i]
        diag = (d == -11) or (d == -9) or (d == 9) or (d == 11)
        t = sq + d
        while True:
            p = bd[t]
            if p == OFF:
                break
            if p != EMPTY:
                if diag:
                    if p == bi:
                        return t
                elif p == ro and rsq < 0:
                    rsq = t
                if p == qu and qsq < 0:
                    qsq = t
                break
            t += d
    if rsq >= 0:
        return rsq
    if qsq >= 0:
        return qsq
    for i in range(8):
        if bd[sq + K_OFF[i]] == kg:
            return sq + K_OFF[i]
    return -1


@njit(cache=False, nogil=True, error_model="numpy")
def see(bd, st, m):
    """Static exchange evaluation of move m (captures, promotions or quiets) in centipawns from the
    mover's view. Uses the swap algorithm with x-rays revealed by removing pieces from the board;
    the board is restored before returning."""
    fr = m & M_FROM
    to = (m & M_TO) >> 7
    side = st[ST_SIDE]
    att = bd[fr]
    promo = (m & M_PROMO) >> 14
    gain = np.zeros(32, dtype=np.int32)
    removed = np.zeros(32, dtype=np.int32)
    removed_p = np.zeros(32, dtype=np.int8)
    nrem = 0
    d = 0
    if m & F_EP:
        capsq = to - 10 if side == WHITE else to + 10
        gain[0] = 100
        removed[nrem] = capsq
        removed_p[nrem] = bd[capsq]
        bd[capsq] = EMPTY
        nrem += 1
    else:
        vic = bd[to]
        gain[0] = PIECE_VAL[vic] if vic != EMPTY else 0
    attacker_val = PIECE_VAL[att]
    if promo:
        pv = PIECE_VAL[promo]
        gain[0] += pv - 100
        attacker_val = pv
    removed[nrem] = fr
    removed_p[nrem] = att
    bd[fr] = EMPTY
    nrem += 1
    stm = side ^ 1
    while d < 30:
        d += 1
        gain[d] = attacker_val - gain[d - 1]
        if max(-gain[d - 1], gain[d]) < 0:
            break
        sq = least_valuable_attacker(bd, to, stm)
        if sq < 0:
            break
        p = bd[sq]
        attacker_val = PIECE_VAL[p]
        if (p == WP or p == BP) and (to >= 91 or to <= 28):
            attacker_val = 900
            gain[d] += 800  # a promoting pawn recapture also gains the queen upgrade
        removed[nrem] = sq
        removed_p[nrem] = p
        bd[sq] = EMPTY
        nrem += 1
        stm ^= 1
    while d > 1:
        d -= 1
        gain[d - 1] = -max(-gain[d - 1], gain[d])
    for i in range(nrem):
        bd[removed[i]] = removed_p[i]
    return gain[0]


@njit(cache=False, nogil=True, error_model="numpy")
def has_non_pawn(bd, side):
    if side == WHITE:
        for s in range(64):
            p = bd[SQ120[s]]
            if p >= 2 and p <= 5:
                return True
    else:
        for s in range(64):
            p = bd[SQ120[s]]
            if p >= 8 and p <= 11:
                return True
    return False


@njit(cache=False, nogil=True, error_model="numpy")
def score_moves(bd, st, mv, sc, n, tt_move, killers, history, ply, counter):
    for i in range(n):
        m = mv[i]
        if m == tt_move:
            sc[i] = 20000000
        elif m & F_CAPTURE:
            fr = m & M_FROM
            to = (m & M_TO) >> 7
            vic = bd[to]
            if vic == EMPTY:
                vic = WP  # en passant
            att = bd[fr]
            s = MVV_LVA[vic, att]
            # losing captures (attacker worth more than victim and the exchange loses) go below quiets
            if PIECE_VAL[att] > PIECE_VAL[vic] and see(bd, st, m) < 0:
                sc[i] = 1000000 + s
            else:
                sc[i] = 10000000 + s
            if m & M_PROMO:
                sc[i] += 5000000 if ((m & M_PROMO) >> 14) == 5 else -2000000
        elif m & M_PROMO:
            sc[i] = 9000000 if ((m & M_PROMO) >> 14) == 5 else 500000
        elif m == killers[ply, 0]:
            sc[i] = 8000000
        elif m == killers[ply, 1]:
            sc[i] = 7000000
        elif m == counter:
            sc[i] = 6500000
        else:
            sc[i] = 2000000 + history[bd[m & M_FROM], (m & M_TO) >> 7]


@njit(cache=False, nogil=True, error_model="numpy")
def pick_next(mv, sc, n, i):
    best = i
    bs = sc[i]
    for j in range(i + 1, n):
        if sc[j] > bs:
            bs = sc[j]
            best = j
    if best != i:
        tm = mv[i]
        mv[i] = mv[best]
        mv[best] = tm
        ts = sc[i]
        sc[i] = sc[best]
        sc[best] = ts
    return mv[i]


@njit(cache=False, nogil=True, error_model="numpy")
def check_time(sinfo, tinfo):
    sinfo[SI_CHECK] += 1
    if (sinfo[SI_CHECK] & 255) == 0:
        with objmode(now="f8"):
            now = time.perf_counter()
        if now >= tinfo[0]:
            sinfo[SI_STOP] = 1



@njit(cache=False, nogil=True)
def has_legal_move(bd, st, undo, hs, buf):
    n = gen_moves(bd, st, buf, False)
    for i in range(n):
        m = buf[i]
        if make_move(bd, st, undo, hs, m):
            unmake_move(bd, st, undo, m)
            return True
    return False

@njit(int64(int8[::1], int32[::1], int32[:, ::1], int64[::1], int32[:, ::1], int32[:, ::1], int64, int64, int64[::1], float64[::1], int64[::1], int64[::1], int64, int32[:, ::1], int32[:, ::1], int32[:, :, ::1], int16[:, ::1], int16[::1], int32, int64), cache=False, nogil=True, error_model="numpy")
def qsearch(bd, st, undo, hs, mvbuf, scbuf, alpha, beta, sinfo, tinfo, tt_key, tt_data, mode, tbl_mg, tbl_eg, acc, w1, w2, b2, qply):
    sinfo[SI_NODES] += 1
    check_time(sinfo, tinfo)
    if sinfo[SI_STOP]:
        return 0
    ply = st[ST_PLY]
    if ply > sinfo[SI_SELDEPTH]:
        sinfo[SI_SELDEPTH] = ply
    side = st[ST_SIDE]
    if ply >= 120:
        return evaluate(bd, st, mode, tbl_mg, tbl_eg, acc, ply, w1, w2, b2)
    if ply >= sinfo[SI_PLY_CAP]:
        return draw_score(sinfo, side)
    ksq = st[ST_KSQ_W] if side == WHITE else st[ST_KSQ_B]
    incheck = attacked(bd, ksq, side ^ 1)
    if not has_legal_move(bd, st, undo, hs, mvbuf[ply]):
        return -MATE + ply if incheck else draw_score(sinfo, side)
    if st[ST_HALF] >= 100 or is_repetition(hs, st[ST_HISTLEN], ply, st[ST_HALF]):
        return draw_score(sinfo, side)
    pv_node = beta - alpha > 1

    # transposition table: any stored bound is at least as deep as a quiescence node
    key = hs[st[ST_HISTLEN] + ply]
    idx = key & TT_MASK
    tt_hit = False
    tt_score = 0
    tt_flag = 0
    if tt_key[idx] == key:
        data = tt_data[idx]
        tt_score = int((data >> 21) & 0xFFFF) - 32768
        tt_flag = int((data >> 45) & 0x3)
        tt_hit = True
        if tt_score > MATE_BOUND:
            tt_score -= ply
        elif tt_score < -MATE_BOUND:
            tt_score += ply
        if not pv_node:
            if tt_flag == TT_EXACT:
                return tt_score
            if tt_flag == TT_LOWER and tt_score >= beta:
                return tt_score
            if tt_flag == TT_UPPER and tt_score <= alpha:
                return tt_score

    if incheck:
        stand = -INF
        n = gen_moves(bd, st, mvbuf[ply], False)
    else:
        stand = evaluate(bd, st, mode, tbl_mg, tbl_eg, acc, ply, w1, w2, b2)
        if tt_hit:
            if (tt_flag == TT_LOWER and tt_score > stand) or (tt_flag == TT_UPPER and tt_score < stand) or tt_flag == TT_EXACT:
                stand = tt_score
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
        n = gen_moves(bd, st, mvbuf[ply], True)
    mv = mvbuf[ply]
    sc = scbuf[ply]
    for i in range(n):
        m = mv[i]
        if m & F_CAPTURE:
            vic = bd[(m & M_TO) >> 7]
            if vic == EMPTY:
                vic = WP
            sc[i] = 1000000 + MVV_LVA[vic, bd[m & M_FROM]]
        elif m & M_PROMO:
            sc[i] = 900000 if ((m & M_PROMO) >> 14) == 5 else 1000
        else:
            sc[i] = 0
    best = stand
    best_move = 0
    alpha_orig = alpha
    legal = 0
    for i in range(n):
        m = pick_next(mv, sc, n, i)
        if stand > -INF:
            if m & F_CAPTURE:
                vic = bd[(m & M_TO) >> 7]
                gain = PIECE_VAL[vic] if vic != EMPTY else 100
                if m & M_PROMO:
                    gain += 800
                # delta pruning
                if stand + gain + 150 < alpha:
                    continue
                # losing captures are not worth resolving
                if see(bd, st, m) < 0:
                    continue
            elif m & M_PROMO:
                if ((m & M_PROMO) >> 14) != 5:
                    continue
        if mode >= 1:
            nn_push(bd, st, m, acc, ply, w1)
        if not make_move(bd, st, undo, hs, m):
            continue
        legal += 1
        score = -qsearch(bd, st, undo, hs, mvbuf, scbuf, -beta, -alpha, sinfo, tinfo, tt_key, tt_data, mode, tbl_mg, tbl_eg, acc, w1, w2, b2, qply + 1)
        unmake_move(bd, st, undo, m)
        if sinfo[SI_STOP]:
            return 0
        if score > best:
            best = score
            best_move = m
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
    if stand == -INF and legal == 0:
        return -MATE + ply
    if best >= beta:
        flag = TT_LOWER
    elif best > alpha_orig:
        flag = TT_EXACT
    else:
        flag = TT_UPPER
    tt_store(tt_key, tt_data, key, best_move, best, 0, flag, ply, sinfo[SI_GEN])
    return best


@njit(int64(int8[::1], int32[::1], int32[:, ::1], int64[::1], int32[:, ::1], int32[:, ::1], int64, int64, int64, int64[::1], float64[::1], int64[::1], int64[::1], int32[:, ::1], int32[:, ::1], int32[:, ::1], int32[:, ::1], int64, int32[:, ::1], int32[:, ::1], int32[:, :, ::1], int16[:, ::1], int16[::1], int32, boolean), cache=False, nogil=True, error_model="numpy")
def negamax(bd, st, undo, hs, mvbuf, scbuf, depth, alpha, beta, sinfo, tinfo, tt_key, tt_data, killers, history,
            counters, stk, mode, tbl_mg, tbl_eg, acc, w1, w2, b2, can_null):
    ply = st[ST_PLY]
    hl = st[ST_HISTLEN]
    is_root = ply == 0
    pv_node = beta - alpha > 1

    side = st[ST_SIDE]
    if not is_root:
        if st[ST_HALF] >= 100 or is_repetition(hs, hl, ply, st[ST_HALF]):
            k = st[ST_KSQ_W] if side == WHITE else st[ST_KSQ_B]
            if attacked(bd, k, side ^ 1) and not has_legal_move(bd, st, undo, hs, mvbuf[ply]):
                return -MATE + ply
            return draw_score(sinfo, side)
        if ply >= 120:
            return evaluate(bd, st, mode, tbl_mg, tbl_eg, acc, ply, w1, w2, b2)
        if ply >= sinfo[SI_PLY_CAP]:
            return draw_score(sinfo, side)
        # mate distance pruning
        a = -MATE + ply
        if a > alpha:
            alpha = a
        b = MATE - ply - 1
        if b < beta:
            beta = b
        if alpha >= beta:
            return alpha

    ksq = st[ST_KSQ_W] if side == WHITE else st[ST_KSQ_B]
    incheck = attacked(bd, ksq, side ^ 1)
    if incheck:
        depth += 1

    if depth <= 0:
        return qsearch(bd, st, undo, hs, mvbuf, scbuf, alpha, beta, sinfo, tinfo, tt_key, tt_data, mode, tbl_mg, tbl_eg, acc, w1, w2, b2, 0)

    sinfo[SI_NODES] += 1
    check_time(sinfo, tinfo)
    if sinfo[SI_STOP]:
        return 0

    # transposition table
    key = hs[hl + ply]
    idx = key & TT_MASK
    tt_move = 0
    tt_hit = False
    tt_score = 0
    tt_flag = 0
    tt_depth = -1
    if tt_key[idx] == key:
        data = tt_data[idx]
        tt_move = int(data & 0x1FFFFF)
        tt_score = int((data >> 21) & 0xFFFF) - 32768
        tt_depth = int((data >> 37) & 0xFF)
        tt_flag = int((data >> 45) & 0x3)
        tt_hit = True
        if tt_score > MATE_BOUND:
            tt_score -= ply
        elif tt_score < -MATE_BOUND:
            tt_score += ply
        if not pv_node and tt_depth >= depth:
            if tt_flag == TT_EXACT:
                return tt_score
            if tt_flag == TT_LOWER and tt_score >= beta:
                return tt_score
            if tt_flag == TT_UPPER and tt_score <= alpha:
                return tt_score

    static_eval = 0
    improving = False
    if not incheck:
        static_eval = evaluate(bd, st, mode, tbl_mg, tbl_eg, acc, ply, w1, w2, b2)
        stk[ply, 1] = static_eval
        if ply >= 2:
            prev2 = stk[ply - 2, 1]
            improving = prev2 == -INF or static_eval > prev2
        if tt_hit:
            # use the table score as a better static estimate when its bound allows it
            if (tt_flag == TT_LOWER and tt_score > static_eval) or (tt_flag == TT_UPPER and tt_score < static_eval) or tt_flag == TT_EXACT:
                static_eval = tt_score
    else:
        stk[ply, 1] = -INF

    if not pv_node and not incheck:
        # reverse futility pruning
        rfp = 80 * depth - (20 * depth if improving else 0)
        if depth <= 7 and static_eval - rfp >= beta and static_eval < MATE_BOUND:
            return static_eval if has_legal_move(bd, st, undo, hs, mvbuf[ply]) else draw_score(sinfo, side)
        # null move pruning
        if can_null and depth >= 3 and static_eval >= beta and has_non_pawn(bd, side) and has_legal_move(bd, st, undo, hs, mvbuf[ply]):
            r = 3 + depth // 3
            if static_eval - beta > 200:
                r += 1
            stk[ply, 0] = 0
            make_null(bd, st, undo, hs)
            if mode >= 1:
                h = acc.shape[2]
                for k in range(h):
                    acc[ply + 1, 0, k] = acc[ply, 0, k]
                    acc[ply + 1, 1, k] = acc[ply, 1, k]
            score = -negamax(bd, st, undo, hs, mvbuf, scbuf, depth - 1 - r, -beta, -beta + 1, sinfo, tinfo, tt_key, tt_data,
                             killers, history, counters, stk, mode, tbl_mg, tbl_eg, acc, w1, w2, b2, False)
            unmake_null(st, undo)
            if sinfo[SI_STOP]:
                return 0
            if score >= beta:
                if score > MATE_BOUND:
                    score = beta
                return score

    # internal iterative reduction: no hash move on a deep node -> reduce
    if tt_move == 0 and depth >= 5 and pv_node:
        depth -= 1

    # countermove: the reply that refuted the opponent's last move elsewhere in the tree
    counter = 0
    if ply > 0:
        pm = stk[ply - 1, 0]
        if pm != 0:
            pto = (pm & M_TO) >> 7
            counter = counters[bd[pto], pto]

    mv = mvbuf[ply]
    sc = scbuf[ply]
    n = gen_moves(bd, st, mv, False)
    score_moves(bd, st, mv, sc, n, tt_move, killers, history, ply, counter)

    best = -INF
    best_move = 0
    alpha_orig = alpha
    legal = 0
    futile = (not pv_node) and (not incheck) and depth <= 3 and static_eval + 100 + 120 * depth <= alpha
    lmp_limit = 3 + depth * depth
    if not improving:
        lmp_limit = lmp_limit * 2 // 3 + 1

    for i in range(n):
        m = pick_next(mv, sc, n, i)
        is_quiet = (m & F_CAPTURE) == 0 and (m & M_PROMO) == 0
        if legal > 0 and not pv_node and not incheck:
            if is_quiet:
                if depth <= 3 and legal >= lmp_limit:
                    continue
                if futile:
                    continue
            elif (m & F_CAPTURE) and depth <= 6 and see(bd, st, m) < -100 * depth:
                continue
        if mode >= 1:
            nn_push(bd, st, m, acc, ply, w1)
        stk[ply, 0] = m
        if not make_move(bd, st, undo, hs, m):
            continue
        legal += 1
        if is_root:
            sinfo[SI_ROOT_DEPTH] = legal
        nksq = st[ST_KSQ_W] if st[ST_SIDE] == WHITE else st[ST_KSQ_B]
        gives_check = attacked(bd, nksq, side)
        score = 0
        if legal == 1:
            score = -negamax(bd, st, undo, hs, mvbuf, scbuf, depth - 1, -beta, -alpha, sinfo, tinfo, tt_key, tt_data,
                             killers, history, counters, stk, mode, tbl_mg, tbl_eg, acc, w1, w2, b2, True)
        else:
            red = 0
            if depth >= 3 and is_quiet and not incheck and not gives_check:
                red = LMR[min(depth, 63), min(legal, 63)]
                if pv_node and red > 0:
                    red -= 1
                if m == killers[ply, 0] or m == killers[ply, 1] or m == counter:
                    red -= 1
                if not improving:
                    red += 1
                hv = history[bd[m & M_FROM], (m & M_TO) >> 7]
                if hv > 6000:
                    red -= 1
                elif hv < -6000:
                    red += 1
                if red < 0:
                    red = 0
                if red > depth - 2:
                    red = depth - 2
            score = -negamax(bd, st, undo, hs, mvbuf, scbuf, depth - 1 - red, -alpha - 1, -alpha, sinfo, tinfo, tt_key, tt_data,
                             killers, history, counters, stk, mode, tbl_mg, tbl_eg, acc, w1, w2, b2, True)
            if score > alpha and red > 0:
                score = -negamax(bd, st, undo, hs, mvbuf, scbuf, depth - 1, -alpha - 1, -alpha, sinfo, tinfo, tt_key, tt_data,
                                 killers, history, counters, stk, mode, tbl_mg, tbl_eg, acc, w1, w2, b2, True)
            if score > alpha and score < beta:
                score = -negamax(bd, st, undo, hs, mvbuf, scbuf, depth - 1, -beta, -alpha, sinfo, tinfo, tt_key, tt_data,
                                 killers, history, counters, stk, mode, tbl_mg, tbl_eg, acc, w1, w2, b2, True)
        unmake_move(bd, st, undo, m)
        if sinfo[SI_STOP]:
            return 0
        if score > best:
            best = score
            best_move = m
            if score > alpha:
                alpha = score
                if is_root:
                    sinfo[SI_ROOT_BEST] = m
                    sinfo[SI_ROOT_SCORE] = score
                if alpha >= beta:
                    if is_quiet:
                        if killers[ply, 0] != m:
                            killers[ply, 1] = killers[ply, 0]
                            killers[ply, 0] = m
                        if ply > 0:
                            pm = stk[ply - 1, 0]
                            if pm != 0:
                                pto = (pm & M_TO) >> 7
                                counters[bd[pto], pto] = m
                        pc = bd[m & M_FROM]
                        to = (m & M_TO) >> 7
                        history[pc, to] += depth * depth
                        if history[pc, to] > 400000:
                            for a1 in range(13):
                                for a2 in range(120):
                                    history[a1, a2] //= 2
                        # penalise the quiets tried before this one
                        for j in range(i):
                            pm2 = mv[j]
                            if (pm2 & F_CAPTURE) == 0 and (pm2 & M_PROMO) == 0:
                                history[bd[pm2 & M_FROM], (pm2 & M_TO) >> 7] -= depth * depth // 2
                    break

    if legal == 0:
        if incheck:
            return -MATE + ply
        return draw_score(sinfo, side)

    flag = TT_EXACT if best > alpha_orig and best < beta else (TT_LOWER if best >= beta else TT_UPPER)
    tt_store(tt_key, tt_data, key, best_move, best, depth, flag, ply, sinfo[SI_GEN])
    return best


def find_pv(bd, st, undo, hs, tt_key, tt_data, max_len=12):
    """Walk the TT to extract the principal variation (for logging)."""
    from zboard import move_to_uci
    pv = []
    mvbuf = np.zeros(256, dtype=np.int32)
    made = []
    for _ in range(max_len):
        key = hs[st[ST_HISTLEN] + st[ST_PLY]]
        idx = key & TT_MASK
        if tt_key[idx] != key:
            break
        m = int(tt_data[idx] & 0x1FFFFF)
        if m == 0:
            break
        n = gen_moves(bd, st, mvbuf, False)
        if m not in set(int(x) for x in mvbuf[:n]):
            break
        if not make_move(bd, st, undo, hs, m):
            break
        made.append(m)
        pv.append(move_to_uci(m))
    for m in reversed(made):
        unmake_move(bd, st, undo, m)
    return pv
