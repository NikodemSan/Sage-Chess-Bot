"""Search: iterative deepening, aspiration windows, PVS alpha-beta with a transposition table,
null-move pruning, reverse futility, futility, late move reductions and pruning, killers,
history, check extension and a capture quiescence search with delta pruning.

Time is checked against a wall-clock deadline at an adaptive node interval through numba's objmode.
"""

import math
import time

import numpy as np
from numba import int8, int16, int32, int64, float64, boolean, njit, objmode

from zboard import (
    BB, BK, BN, BP, BQ, BR, B_OFF, EMPTY, F_CAPTURE, F_EP, K_OFF, M_FROM, M_PROMO, M_TO, N_OFF, OFF, R_OFF,
    PIECE_TYPE, PIECE_VAL, SQ120, SQ64, ST_HALF, ST_HISTLEN, ST_KSQ_B, ST_KSQ_W, ST_PLY, ST_SIDE,
    WB, WHITE, WK, WN, WP, WQ, WR, attacked, gen_moves,
    make_move, make_null, unmake_move, unmake_null,
)
from zeval import evaluate, nn_push

MATE = 30000
MATE_BOUND = 29000
INF = 32000
DRAW = 0

TT_BITS = 23
TT_SIZE = 1 << TT_BITS
TT_MASK = TT_SIZE - 1
TT_EXACT, TT_LOWER, TT_UPPER = 1, 2, 3

EVAL_BITS = 22
EVAL_SIZE = 1 << EVAL_BITS
EVAL_MASK = EVAL_SIZE - 1

# sinfo layout
SI_NODES, SI_STOP, SI_CHECK, SI_ROOT_BEST, SI_ROOT_SCORE, SI_GEN, SI_SELDEPTH, SI_ROOT_DEPTH, SI_ROOT_SIDE, SI_CONTEMPT, SI_TIME_MASK = range(11)

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
def is_repetition(hs, hl, ply, half):
    """Detect a real threefold or a cycle created inside the current search.

    Negative relative indices are actual game history before the search root.  A single historical
    match is only the *second* occurrence and is not yet claimable.  A match at a non-negative
    index is a cycle made by the current principal search path; engines conventionally score that
    as a draw immediately to avoid endlessly re-searching reversible loops.
    """
    cur = hs[hl + ply]
    i = ply - 2
    lim = ply - half
    if lim < -hl:
        lim = -hl
    historical_matches = 0
    while i >= lim:
        if hs[hl + i] == cur:
            if i >= 0:
                return True
            historical_matches += 1
            if historical_matches >= 2:
                return True
        i -= 2
    return False


@njit(cache=False, nogil=True, error_model="numpy")
def null_move_material_ok(bd, side):
    """Allow null-move pruning only with rook-equivalent non-pawn material.

    Zugzwang is concentrated in sparse pawn/minor-piece endings.  The old any-non-pawn guard
    still allowed NMP in K+N+P type endings, where a false fail-high is especially dangerous.
    Requiring at least 500cp of non-pawn material keeps normal middlegame NMP unchanged while
    conservatively disabling it in the classic zugzwang-prone minor-only endings.
    """
    total = 0
    if side == WHITE:
        lo, hi = 2, 5
    else:
        lo, hi = 8, 11
    for s in range(64):
        p = bd[SQ120[s]]
        if p >= lo and p <= hi:
            total += PIECE_VAL[p]
            if total >= 500:
                return True
    return False


@njit(cache=False, nogil=True, error_model="numpy")
def sparse_endgame(bd):
    """True for <=7 pieces, where exact terminal handling matters more than delta pruning."""
    count = 0
    for s in range(64):
        if bd[SQ120[s]] != EMPTY:
            count += 1
            if count > 7:
                return False
    return True


@njit(inline="always", cache=False, nogil=True, error_model="numpy")
def _see_bit(sq):
    """64-bit occupancy bit for an on-board mailbox square."""
    return np.uint64(1) << np.uint64(SQ64[sq])


@njit(inline="always", cache=False, nogil=True, error_model="numpy")
def _see_piece_at(bd, sq, removed, to, to_piece):
    """Piece lookup in SEE's virtual position without touching the real board.

    `removed` contains source squares vacated by the exchange (and an EP victim square).
    The target square is special: every capture replaces the previous trophy there.
    """
    p = bd[sq]
    if p == OFF:
        return OFF
    if sq == to:
        return to_piece
    if (removed & _see_bit(sq)) != 0:
        return EMPTY
    return p


@njit(cache=False, nogil=True, error_model="numpy")
def _see_attacked(bd, sq, by, removed, to, to_piece):
    """Attack test on SEE's virtual occupancy.

    This deliberately mirrors zboard.attacked().  Keeping it mailbox-native avoids converting the
    whole position to bitboards for every SEE call, while the virtual occupancy makes pins, x-rays,
    en-passant discoveries and king captures legal rather than merely pseudo-legal.
    """
    if by == WHITE:
        if _see_piece_at(bd, sq - 11, removed, to, to_piece) == WP or _see_piece_at(bd, sq - 9, removed, to, to_piece) == WP:
            return True
        kn, bi, ro, qu, kg = WN, WB, WR, WQ, WK
    else:
        if _see_piece_at(bd, sq + 11, removed, to, to_piece) == BP or _see_piece_at(bd, sq + 9, removed, to, to_piece) == BP:
            return True
        kn, bi, ro, qu, kg = BN, BB, BR, BQ, BK

    for i in range(8):
        if _see_piece_at(bd, sq + N_OFF[i], removed, to, to_piece) == kn:
            return True
    for i in range(8):
        if _see_piece_at(bd, sq + K_OFF[i], removed, to, to_piece) == kg:
            return True
    for i in range(8):
        d = K_OFF[i]
        diag = d == -11 or d == -9 or d == 9 or d == 11
        slider = bi if diag else ro
        t = sq + d
        while True:
            p = _see_piece_at(bd, t, removed, to, to_piece)
            if p == OFF:
                break
            if p != EMPTY:
                if p == slider or p == qu:
                    return True
                break
            t += d
    return False


@njit(inline="always", cache=False, nogil=True, error_model="numpy")
def _see_legal_attacker(bd, st, to, side, removed, fr, piece_after):
    """Whether `fr x to` is legal in the virtual exchange position.

    Full king-safety simulation is intentional here.  A cheaper pin approximation is tempting, but
    SEE is used for forward pruning: a false "losing capture" is much more harmful than spending a
    few extra instructions on the small subset of captures that reach SEE.
    """
    removed2 = removed | _see_bit(fr)
    if PIECE_TYPE[piece_after] == 6:
        ksq = to
    else:
        ksq = st[ST_KSQ_W] if side == WHITE else st[ST_KSQ_B]
    return not _see_attacked(bd, ksq, side ^ 1, removed2, to, piece_after)


@njit(cache=False, nogil=True, error_model="numpy")
def _see_lva(bd, st, to, side, removed, to_piece):
    """Return the least valuable *legal* recapturing attacker.

    Result is (from_square, piece_left_on_target, promotion_bonus).  Rescanning the eight rays after
    each virtual capture naturally discovers x-rays.  Equal-valued candidates are all checked for
    legality before moving to the next piece class.
    """
    if side == WHITE:
        pawn, knight, bishop, rook, queen, king = WP, WN, WB, WR, WQ, WK
        ps1, ps2 = to - 11, to - 9
        promotes = (SQ64[to] >> 3) == 7
    else:
        pawn, knight, bishop, rook, queen, king = BP, BN, BB, BR, BQ, BK
        ps1, ps2 = to + 11, to + 9
        promotes = (SQ64[to] >> 3) == 0

    # Pawn
    for fr in (ps1, ps2):
        if _see_piece_at(bd, fr, removed, to, to_piece) == pawn:
            piece_after = queen if promotes else pawn
            bonus = 800 if promotes else 0
            if _see_legal_attacker(bd, st, to, side, removed, fr, piece_after):
                return fr, piece_after, bonus

    # Knight
    for i in range(8):
        fr = to + N_OFF[i]
        if _see_piece_at(bd, fr, removed, to, to_piece) == knight:
            if _see_legal_attacker(bd, st, to, side, removed, fr, knight):
                return fr, knight, 0

    # Bishop
    for i in range(4):
        d = B_OFF[i]
        fr = to + d
        while True:
            p = _see_piece_at(bd, fr, removed, to, to_piece)
            if p == OFF:
                break
            if p != EMPTY:
                if p == bishop and _see_legal_attacker(bd, st, to, side, removed, fr, bishop):
                    return fr, bishop, 0
                break
            fr += d

    # Rook
    for i in range(4):
        d = R_OFF[i]
        fr = to + d
        while True:
            p = _see_piece_at(bd, fr, removed, to, to_piece)
            if p == OFF:
                break
            if p != EMPTY:
                if p == rook and _see_legal_attacker(bd, st, to, side, removed, fr, rook):
                    return fr, rook, 0
                break
            fr += d

    # Queen (both diagonal and orthogonal rays)
    for i in range(8):
        d = K_OFF[i]
        fr = to + d
        while True:
            p = _see_piece_at(bd, fr, removed, to, to_piece)
            if p == OFF:
                break
            if p != EMPTY:
                if p == queen and _see_legal_attacker(bd, st, to, side, removed, fr, queen):
                    return fr, queen, 0
                break
            fr += d

    # King.  Full legality above guarantees that if this capture is legal the exchange ends: the
    # opponent cannot have a legal attacker on a square occupied by the king.
    for i in range(8):
        fr = to + K_OFF[i]
        if _see_piece_at(bd, fr, removed, to, to_piece) == king:
            if _see_legal_attacker(bd, st, to, side, removed, fr, king):
                return fr, king, 0

    return 0, 0, 0


@njit(cache=False, nogil=True, error_model="numpy")
def see_capture(bd, st, m):
    """Legal LVA swap-off Static Exchange Evaluation for capture `m` (centipawns).

    The initial move is forced. Thereafter both sides may stop the exchange, while legal
    least-valuable recaptures are followed on the target square. The virtual occupancy discovers
    x-rays after every removal and handles pins, king legality, en passant, and promotion material
    without modifying the real board/state.
    """
    fr = m & M_FROM
    to = (m & M_TO) >> 7
    side = st[ST_SIDE]
    mover = bd[fr]

    captured = bd[to]
    removed = _see_bit(fr)
    if m & F_EP:
        cap_sq = to - 10 if side == WHITE else to + 10
        captured = BP if side == WHITE else WP
        removed |= _see_bit(cap_sq)

    promo = (m & M_PROMO) >> 14
    promo_bonus = 0
    if promo:
        current_piece = promo if side == WHITE else promo + 6
        promo_bonus = PIECE_VAL[current_piece] - 100
    else:
        current_piece = mover

    # A legal chess position has at most 32 pieces, hence at most 31 recaptures after the forced
    # first move. Numba lowers this small fixed array efficiently and avoids recursion entirely.
    gain = np.empty(32, dtype=np.int32)
    gain[0] = (PIECE_VAL[captured] if captured != EMPTY else 0) + promo_bonus
    depth = 0
    stm = side ^ 1

    while depth < 31:
        afrom, piece_after, bonus = _see_lva(bd, st, to, stm, removed, current_piece)
        if afrom == 0:
            break
        depth += 1
        # The trophy captured at this ply is the piece left on `to` by the previous capture.
        gain[depth] = PIECE_VAL[current_piece] + bonus - gain[depth - 1]
        removed |= _see_bit(afrom)
        current_piece = piece_after
        if PIECE_TYPE[piece_after] == 6:
            break
        stm ^= 1

    # Negamax the swap list backwards. At every recapture after the forced initial move the side
    # to move may decline the exchange, which is the stand-pat alternative encoded by max().
    while depth > 0:
        gain[depth - 1] = -max(-gain[depth - 1], gain[depth])
        depth -= 1
    return int(gain[0])


@njit(inline="always", cache=False, nogil=True, error_model="numpy")
def see_ge(bd, st, m, threshold):
    """Threshold SEE used by search pruning/order decisions."""
    return see_capture(bd, st, m) >= threshold


@njit(cache=False, nogil=True, error_model="numpy")
def _see_capture_gives_check(bd, st, m):
    """Whether capture `m` gives check, using the same virtual occupancy as SEE.

    This is only used as a safety valve before qsearch SEE-prunes a materially losing capture.
    Checking sacrifices can be positionally/tactically decisive even when their local swap-off is bad,
    so qsearch keeps them.  Computing the check virtually avoids a make/unmake just to decide whether
    the pruning heuristic is allowed.
    """
    fr = m & M_FROM
    to = (m & M_TO) >> 7
    side = st[ST_SIDE]
    removed = _see_bit(fr)
    if m & F_EP:
        cap_sq = to - 10 if side == WHITE else to + 10
        removed |= _see_bit(cap_sq)

    promo = (m & M_PROMO) >> 14
    if promo:
        to_piece = promo if side == WHITE else promo + 6
    else:
        to_piece = bd[fr]

    opp_king = st[ST_KSQ_B] if side == WHITE else st[ST_KSQ_W]
    return _see_attacked(bd, opp_king, side, removed, to, to_piece)


@njit(cache=False, nogil=True, error_model="numpy")
def score_moves(bd, st, mv, sc, n, tt_move, killers, history, ply):
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
            # If attacker <= victim, the capture is provably SEE >= 0 even if immediately
            # recaptured, so avoid the SEE call.  Only ambiguous high-attacker/low-victim captures
            # pay for the full legal swap-off.
            if PIECE_VAL[att] > PIECE_VAL[vic] and not see_ge(bd, st, m, 0):
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
    if (sinfo[SI_CHECK] & sinfo[SI_TIME_MASK]) == 0:
        with objmode(now="f8"):
            now = time.perf_counter()
        if now >= tinfo[0]:
            sinfo[SI_STOP] = 1


@njit(cache=False, nogil=True, error_model="numpy")
def has_legal_move(bd, st, undo, hs, mv):
    """Cheap legality probe used only when a draw rule collides with an in-check node."""
    n = gen_moves(bd, st, mv, False)
    for i in range(n):
        m = mv[i]
        if make_move(bd, st, undo, hs, m):
            unmake_move(bd, st, undo, m)
            return True
    return False


@njit(cache=False, nogil=True, error_model="numpy")
def cached_eval(key, bd, st, mode, tbl_mg, tbl_eg, acc, ply, w2, b2, eval_key, eval_val):
    if key != 0:
        idx = key & EVAL_MASK
        if eval_key[idx] == key:
            return int(eval_val[idx])
    v = evaluate(bd, st, mode, tbl_mg, tbl_eg, acc, ply, w2, b2)
    if key != 0:
        eval_key[idx] = key
        eval_val[idx] = v
    return v


@njit(int64(int8[::1], int32[::1], int32[:, ::1], int64[::1], int32[:, ::1], int32[:, ::1], int64, int64, int64[::1], float64[::1], int64[::1], int32[::1], int64, int32[:, ::1], int32[:, ::1], int16[:, :, ::1], int16[:, ::1], int16[::1], int32, int64), cache=False, nogil=True, error_model="numpy")
def qsearch(bd, st, undo, hs, mvbuf, scbuf, alpha, beta, sinfo, tinfo, eval_key, eval_val, mode, tbl_mg, tbl_eg, acc, w1, w2, b2, qply):
    sinfo[SI_NODES] += 1
    check_time(sinfo, tinfo)
    if sinfo[SI_STOP]:
        return 0
    ply = st[ST_PLY]
    if ply > sinfo[SI_SELDEPTH]:
        sinfo[SI_SELDEPTH] = ply
    if ply >= 120:
        return cached_eval(hs[st[ST_HISTLEN] + ply], bd, st, mode, tbl_mg, tbl_eg, acc, ply, w2, b2, eval_key, eval_val)
    side = st[ST_SIDE]
    hl = st[ST_HISTLEN]
    ksq = st[ST_KSQ_W] if side == WHITE else st[ST_KSQ_B]
    incheck = attacked(bd, ksq, side ^ 1)
    draw_rule = ply > 0 and (st[ST_HALF] >= 100 or is_repetition(hs, hl, ply, st[ST_HALF]))
    if draw_rule:
        # Checkmate terminates the game before a 50-move/repetition claim can matter.  If the king
        # is not in check, any terminal no-move position is stalemate anyway and the score is draw.
        if incheck and not has_legal_move(bd, st, undo, hs, mvbuf[ply]):
            return -MATE + ply
        return -sinfo[SI_CONTEMPT] if side == sinfo[SI_ROOT_SIDE] else sinfo[SI_CONTEMPT]
    # Stand-pat is never legal while in check.  The old qply<2 guard could silently discard
    # quiet check evasions and even miss mates once a checking sequence reached qsearch depth 2.
    if incheck:
        stand = -INF
        n = gen_moves(bd, st, mvbuf[ply], False)
    else:
        stand = cached_eval(hs[st[ST_HISTLEN] + ply], bd, st, mode, tbl_mg, tbl_eg, acc, ply, w2, b2, eval_key, eval_val)
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
        n = gen_moves(bd, st, mvbuf[ply], True)
    # In very sparse endings, delta/SEE forward pruning can discard the only transition into a
    # tablebase-like win/draw, and a capture-only qsearch otherwise mistakes stalemate for a
    # static position.  Pay the tiny exactness cost only at <=7 pieces.
    sparse = sparse_endgame(bd) if stand > -INF else False
    if stand > -INF and sparse and n == 0 and not has_legal_move(bd, st, undo, hs, mvbuf[ply]):
        return DRAW
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
    legal = 0
    for i in range(n):
        m = pick_next(mv, sc, n, i)
        if stand > -INF and not sparse:
            # Delta/SEE pruning is deliberately disabled in <=7-piece endings.
            if m & F_CAPTURE:
                vic = bd[(m & M_TO) >> 7]
                gain = PIECE_VAL[vic] if vic != EMPTY else 100
                if m & M_PROMO:
                    gain += 800
                if stand + gain + 150 < alpha:
                    continue
                # SEE prune only when the cheap material bound cannot already prove >= -50cp.
                # Keep promotions and checking sacrifices: SEE deliberately models only the local
                # exchange on one square, while qsearch must not throw away a forcing check merely
                # because that sacrifice loses material locally.
                fr = m & M_FROM
                if PIECE_VAL[bd[fr]] > PIECE_VAL[vic] + 50 and not (m & M_PROMO):
                    if not see_ge(bd, st, m, -50) and not _see_capture_gives_check(bd, st, m):
                        continue
        if mode >= 1:
            nn_push(bd, st, m, acc, ply, w1)
        if not make_move(bd, st, undo, hs, m):
            continue
        legal += 1
        score = -qsearch(bd, st, undo, hs, mvbuf, scbuf, -beta, -alpha, sinfo, tinfo, eval_key, eval_val, mode, tbl_mg, tbl_eg, acc, w1, w2, b2, qply + 1)
        unmake_move(bd, st, undo, m)
        if sinfo[SI_STOP]:
            return 0
        if score > best:
            best = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    return best
    if stand == -INF and legal == 0:
        return -MATE + ply
    return best


@njit(int64(int8[::1], int32[::1], int32[:, ::1], int64[::1], int32[:, ::1], int32[:, ::1], int64, int64, int64, int64[::1], float64[::1], int64[::1], int32[::1], int64[::1], int64[::1], int32[:, ::1], int32[:, ::1], int64, int32[:, ::1], int32[:, ::1], int16[:, :, ::1], int16[:, ::1], int16[::1], int32, boolean), cache=False, nogil=True, error_model="numpy")
def negamax(bd, st, undo, hs, mvbuf, scbuf, depth, alpha, beta, sinfo, tinfo, eval_key, eval_val, tt_key, tt_data, killers, history,
            mode, tbl_mg, tbl_eg, acc, w1, w2, b2, can_null):
    ply = st[ST_PLY]
    hl = st[ST_HISTLEN]
    is_root = ply == 0
    pv_node = beta - alpha > 1

    side = st[ST_SIDE]
    ksq = st[ST_KSQ_W] if side == WHITE else st[ST_KSQ_B]
    incheck = attacked(bd, ksq, side ^ 1)
    if not is_root:
        draw_rule = st[ST_HALF] >= 100 or is_repetition(hs, hl, ply, st[ST_HALF])
        if draw_rule:
            if incheck and not has_legal_move(bd, st, undo, hs, mvbuf[ply]):
                return -MATE + ply
            # draws are slightly bad for the root side (contempt), good for its opponent
            return -sinfo[SI_CONTEMPT] if side == sinfo[SI_ROOT_SIDE] else sinfo[SI_CONTEMPT]
        if ply >= 120:
            return evaluate(bd, st, mode, tbl_mg, tbl_eg, acc, ply, w2, b2)
        # mate distance pruning
        a = -MATE + ply
        if a > alpha:
            alpha = a
        b = MATE - ply - 1
        if b < beta:
            beta = b
        if alpha >= beta:
            return alpha
    if incheck:
        depth += 1

    if depth <= 0:
        return qsearch(bd, st, undo, hs, mvbuf, scbuf, alpha, beta, sinfo, tinfo, eval_key, eval_val, mode, tbl_mg, tbl_eg, acc, w1, w2, b2, 0)

    sinfo[SI_NODES] += 1
    check_time(sinfo, tinfo)
    if sinfo[SI_STOP]:
        return 0

    # transposition table
    key = hs[st[ST_HISTLEN] + ply]
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
    if not incheck:
        # For an exact TT entry the old code evaluated first and then overwrote the result with
        # tt_score.  Skipping that redundant NNUE+PSQT call is semantically identical and saves
        # one of the hottest operations at transposed PV/shallow nodes.
        if tt_hit and tt_flag == TT_EXACT:
            static_eval = tt_score
        else:
            static_eval = cached_eval(key, bd, st, mode, tbl_mg, tbl_eg, acc, ply, w2, b2, eval_key, eval_val)
            if tt_hit:
                # use a bound as a better static estimate only in the direction it proves
                if (tt_flag == TT_LOWER and tt_score > static_eval) or (tt_flag == TT_UPPER and tt_score < static_eval):
                    static_eval = tt_score

    if not pv_node and not incheck:
        # reverse futility pruning
        if depth <= 7 and static_eval - 75 * depth >= beta and static_eval < MATE_BOUND:
            return static_eval
        # null move pruning
        if can_null and depth >= 3 and static_eval >= beta and null_move_material_ok(bd, side):
            r = 3 + depth // 3
            if static_eval - beta > 200:
                r += 1
            make_null(bd, st, undo, hs)
            if mode >= 1:
                h = w1.shape[1]
                for k in range(h):
                    acc[ply + 1, 0, k] = acc[ply, 0, k]
                    acc[ply + 1, 1, k] = acc[ply, 1, k]
            score = -negamax(bd, st, undo, hs, mvbuf, scbuf, depth - 1 - r, -beta, -beta + 1, sinfo, tinfo, eval_key, eval_val, tt_key, tt_data,
                             killers, history, mode, tbl_mg, tbl_eg, acc, w1, w2, b2, False)
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

    mv = mvbuf[ply]
    sc = scbuf[ply]
    n = gen_moves(bd, st, mv, False)
    score_moves(bd, st, mv, sc, n, tt_move, killers, history, ply)

    best = -INF
    best_move = 0
    alpha_orig = alpha
    legal = 0
    futile = (not pv_node) and (not incheck) and depth <= 3 and static_eval + 100 + 120 * depth <= alpha
    lmp_limit = 3 + depth * depth

    for i in range(n):
        m = pick_next(mv, sc, n, i)
        is_quiet = (m & F_CAPTURE) == 0 and (m & M_PROMO) == 0
        if legal > 0 and is_quiet and not pv_node and not incheck:
            if depth <= 3 and legal >= lmp_limit:
                continue
            if futile:
                continue
        if mode >= 1:
            nn_push(bd, st, m, acc, ply, w1)
        if not make_move(bd, st, undo, hs, m):
            continue
        legal += 1
        if is_root:
            sinfo[SI_ROOT_DEPTH] = legal
        nksq = st[ST_KSQ_W] if st[ST_SIDE] == WHITE else st[ST_KSQ_B]
        gives_check = attacked(bd, nksq, side)
        score = 0
        if legal == 1:
            score = -negamax(bd, st, undo, hs, mvbuf, scbuf, depth - 1, -beta, -alpha, sinfo, tinfo, eval_key, eval_val, tt_key, tt_data,
                             killers, history, mode, tbl_mg, tbl_eg, acc, w1, w2, b2, True)
        else:
            red = 0
            if depth >= 3 and is_quiet and not incheck and not gives_check:
                red = LMR[min(depth, 63), min(legal, 63)]
                if pv_node and red > 0:
                    red -= 1
                if m == killers[ply, 0] or m == killers[ply, 1]:
                    red -= 1
                if red < 0:
                    red = 0
                if red > depth - 2:
                    red = depth - 2
            score = -negamax(bd, st, undo, hs, mvbuf, scbuf, depth - 1 - red, -alpha - 1, -alpha, sinfo, tinfo, eval_key, eval_val, tt_key, tt_data,
                             killers, history, mode, tbl_mg, tbl_eg, acc, w1, w2, b2, True)
            if score > alpha and red > 0:
                score = -negamax(bd, st, undo, hs, mvbuf, scbuf, depth - 1, -alpha - 1, -alpha, sinfo, tinfo, eval_key, eval_val, tt_key, tt_data,
                                 killers, history, mode, tbl_mg, tbl_eg, acc, w1, w2, b2, True)
            if score > alpha and score < beta:
                score = -negamax(bd, st, undo, hs, mvbuf, scbuf, depth - 1, -beta, -alpha, sinfo, tinfo, eval_key, eval_val, tt_key, tt_data,
                                 killers, history, mode, tbl_mg, tbl_eg, acc, w1, w2, b2, True)
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
                        pc = bd[m & M_FROM]
                        to = (m & M_TO) >> 7
                        history[pc, to] += depth * depth
                        if history[pc, to] > 400000:
                            for a1 in range(13):
                                for a2 in range(120):
                                    history[a1, a2] //= 2
                        # penalise the quiets tried before this one
                        for j in range(i):
                            pm = mv[j]
                            if (pm & F_CAPTURE) == 0 and (pm & M_PROMO) == 0:
                                history[bd[pm & M_FROM], (pm & M_TO) >> 7] -= depth * depth // 2
                    break

    if legal == 0:
        if incheck:
            return -MATE + ply
        return -sinfo[SI_CONTEMPT] if side == sinfo[SI_ROOT_SIDE] else sinfo[SI_CONTEMPT]

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
