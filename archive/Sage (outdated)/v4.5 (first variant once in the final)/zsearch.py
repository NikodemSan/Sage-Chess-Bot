"""Search: iterative deepening, aspiration windows, PVS alpha-beta with a transposition table,
null-move pruning, reverse futility, futility, late move reductions and pruning, killers,
history, check extension and a capture quiescence search with delta pruning.

Time is checked against a wall-clock deadline every 2048 nodes through numba's objmode.
"""

import math
import os
import time

import numpy as np
from numba import int8, int16, int32, int64, float64, boolean, njit, objmode

from zboard import (
    BK, BP, EMPTY, F_CAPTURE, F_EP, M_FROM, M_PROMO, M_TO, PIECE_TYPE, PIECE_VAL, SQ120, ST_HALF,
    ST_HISTLEN, ST_KSQ_B, ST_KSQ_W, ST_PLY, ST_SIDE, WHITE, WK, WP, attacked, gen_moves,
    make_move, make_null, unmake_move, unmake_null, K_OFF, PIECE_COLOR, SQ64,
)
from zeval import evaluate, nn_push

MATE = 30000
MATE_BOUND = 29000
INF = 32000
DRAW = 0
# Staged ordering remains a disabled experiment: it regressed the loss-position
# benchmark. Production preserves the original move-ordering/search policy.
STAGED_TT = os.environ.get('SAGE_STAGE3_STAGED_TT','0') == '1'
FAST_LEGAL = os.environ.get('SAGE_STAGE3_FAST_LEGAL','1') == '1'
CORRECTIONS = os.environ.get('SAGE_STAGE3_CORRECTIONS','1') == '1'

TT_BITS = 21
TT_SIZE = 1 << TT_BITS
TT_MASK = TT_SIZE - 1
TT_EXACT, TT_LOWER, TT_UPPER = 1, 2, 3

# sinfo layout
SI_NODES, SI_STOP, SI_CHECK, SI_ROOT_BEST, SI_ROOT_SCORE, SI_GEN, SI_SELDEPTH, SI_ROOT_DEPTH, SI_ROOT_SIDE, SI_CONTEMPT, SI_AVOID_REVISIT = range(11)

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
    """Three occurrences including the current position; null branches are approximate."""
    matches = 0
    cur = hs[hl + ply]
    i = ply - 2
    lim = ply - half
    if lim < -hl:
        lim = -hl
    while i >= lim:
        if hs[hl + i] == cur:
            matches += 1
            if matches >= 2:
                return True
        i -= 2
    return False


@njit(cache=False, nogil=True, error_model="numpy")
def is_revisit(hs, hl, ply, half):
    """True when the current position has occurred previously in the reversible history."""
    cur = hs[hl + ply]
    i = ply - 2
    lim = ply - half
    if lim < -hl:
        lim = -hl
    while i >= lim:
        if hs[hl + i] == cur:
            return True
        i -= 2
    return False


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
def see_capture(bd, st, m):
    """Cheap static exchange approximation for capture m from the mover's view (centipawns)."""
    fr = m & M_FROM
    to = (m & M_TO) >> 7
    att = bd[fr]
    vic = bd[to]
    gain = PIECE_VAL[vic] if vic != EMPTY else 100  # ep
    if m & M_PROMO:
        gain += 800
    side = st[ST_SIDE]
    # if the destination is defended, assume we lose the attacker
    bd[fr] = EMPTY
    defended = attacked(bd, to, side ^ 1)
    bd[fr] = att
    if defended:
        gain -= PIECE_VAL[att]
    return gain


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
            # losing captures (attacker worth more than victim, target defended) go below quiets
            if PIECE_VAL[att] > PIECE_VAL[vic] and see_capture(bd, st, m) < 0:
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
    if (sinfo[SI_CHECK] & 255) == 0:
        with objmode(now="f8"):
            now = time.perf_counter()
        if now >= tinfo[0]:
            sinfo[SI_STOP] = 1



@njit(cache=False, nogil=True)
def has_legal_move(bd, st, undo, hs, buf):
    if FAST_LEGAL:
        side=st[ST_SIDE]
        fr=st[ST_KSQ_W] if side==WHITE else st[ST_KSQ_B]
        for k in range(8):
            to=fr+K_OFF[k]; victim=bd[to]
            if victim<0 or PIECE_COLOR[victim]==side or victim==WK or victim==BK: continue
            m=fr|(to<<7)|(F_CAPTURE if victim else 0)
            if make_move(bd,st,undo,hs,m):
                unmake_move(bd,st,undo,m)
                return True
    n = gen_moves(bd, st, buf, False)
    for i in range(n):
        m = buf[i]
        if make_move(bd, st, undo, hs, m):
            unmake_move(bd, st, undo, m)
            return True
    return False

@njit(cache=False, nogil=True)
def common_dead_material(bd,acc,ply,w1):
    # Cheap gate from the unchanged evaluator's incremental piece count.
    # Recognize K/K, K+minor/K and same-colour bishop-only four-man positions.
    count=acc[ply,0,w1.shape[1]+15]
    if count>4: return False
    minors=0; knights=0; colour=-1
    for i in range(64):
        p=bd[SQ120[i]]; typ=PIECE_TYPE[p]
        if typ==0 or typ==6: continue
        if typ!=2 and typ!=3: return False
        minors+=1
        if typ==2: knights+=1
        else:
            c=((i//8)+(i%8))&1
            if colour>=0 and colour!=c: return False
            colour=c
    return minors<=1 or knights==0

@njit(cache=False, nogil=True)
def legal_checking_move(bd,st,undo,hs,m):
    if not make_move(bd,st,undo,hs,m): return False
    side=st[ST_SIDE]; king=st[ST_KSQ_W] if side==WHITE else st[ST_KSQ_B]
    check=attacked(bd,king,side^1)
    unmake_move(bd,st,undo,m)
    return check

@njit(int64(int8[::1], int32[::1], int32[:, ::1], int64[::1], int32[:, ::1], int32[:, ::1], int64, int64, int64[::1], float64[::1], int64, int32[:, ::1], int32[:, ::1], int32[:, :, ::1], int16[:, ::1], int16[::1], int32, int64), cache=False, nogil=True, error_model="numpy")
def qsearch(bd, st, undo, hs, mvbuf, scbuf, alpha, beta, sinfo, tinfo, mode, tbl_mg, tbl_eg, acc, w1, w2, b2, qply):
    sinfo[SI_NODES] += 1
    check_time(sinfo, tinfo)
    if sinfo[SI_STOP]:
        return 0
    ply = st[ST_PLY]
    if CORRECTIONS and common_dead_material(bd,acc,ply,w1): return 0
    if ply > sinfo[SI_SELDEPTH]:
        sinfo[SI_SELDEPTH] = ply
    if ply >= 120:
        return evaluate(bd, st, mode, tbl_mg, tbl_eg, acc, ply, w1, w2, b2)
    side = st[ST_SIDE]
    ksq = st[ST_KSQ_W] if side == WHITE else st[ST_KSQ_B]
    incheck = attacked(bd, ksq, side ^ 1)
    if not has_legal_move(bd, st, undo, hs, mvbuf[ply]):
        return -MATE + ply if incheck else 0
    if st[ST_HALF] >= 100 or is_repetition(hs, st[ST_HISTLEN], ply, st[ST_HALF]):
        return 0
    if incheck:
        stand = -INF
        n = gen_moves(bd, st, mvbuf[ply], False)
    else:
        stand = evaluate(bd, st, mode, tbl_mg, tbl_eg, acc, ply, w1, w2, b2)
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
    legal = 0
    for i in range(n):
        m = pick_next(mv, sc, n, i)
        if stand > -INF:
            # delta pruning
            if m & F_CAPTURE:
                vic = bd[(m & M_TO) >> 7]
                gain = PIECE_VAL[vic] if vic != EMPTY else 100
                if m & M_PROMO:
                    gain += 800
                discard = stand + gain + 150 < alpha
                # skip clearly losing captures
                fr = m & M_FROM
                if not discard and PIECE_VAL[bd[fr]] > PIECE_VAL[vic] + 50 and see_capture(bd, st, m) < -50 and not (m & M_PROMO):
                    discard=True
                if discard:
                    # The exchange approximation may count an illegal king
                    # recapture. Checking captures/promotions must survive.
                    if not CORRECTIONS or (not (m&M_PROMO) and not legal_checking_move(bd,st,undo,hs,m)):
                        continue
        if mode >= 1:
            nn_push(bd, st, m, acc, ply, w1)
        if not make_move(bd, st, undo, hs, m):
            continue
        legal += 1
        score = -qsearch(bd, st, undo, hs, mvbuf, scbuf, -beta, -alpha, sinfo, tinfo, mode, tbl_mg, tbl_eg, acc, w1, w2, b2, qply + 1)
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


@njit(int64(int8[::1], int32[::1], int32[:, ::1], int64[::1], int32[:, ::1], int32[:, ::1], int64, int64, int64, int64[::1], float64[::1], int64[::1], int64[::1], int32[:, ::1], int32[:, ::1], int64, int32[:, ::1], int32[:, ::1], int32[:, :, ::1], int16[:, ::1], int16[::1], int32, boolean), cache=False, nogil=True, error_model="numpy")
def negamax(bd, st, undo, hs, mvbuf, scbuf, depth, alpha, beta, sinfo, tinfo, tt_key, tt_data, killers, history,
            mode, tbl_mg, tbl_eg, acc, w1, w2, b2, can_null):
    ply = st[ST_PLY]
    if CORRECTIONS and common_dead_material(bd,acc,ply,w1): return 0
    hl = st[ST_HISTLEN]
    is_root = ply == 0
    pv_node = beta - alpha > 1

    side = st[ST_SIDE]
    if not is_root:
        if st[ST_HALF] >= 100 or is_repetition(hs, hl, ply, st[ST_HALF]):
            k = st[ST_KSQ_W] if side == WHITE else st[ST_KSQ_B]
            if attacked(bd, k, side ^ 1) and not has_legal_move(bd, st, undo, hs, mvbuf[ply]):
                return -MATE + ply
            return 0
        if ply >= 120:
            return evaluate(bd, st, mode, tbl_mg, tbl_eg, acc, ply, w1, w2, b2)
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
        return qsearch(bd, st, undo, hs, mvbuf, scbuf, alpha, beta, sinfo, tinfo, mode, tbl_mg, tbl_eg, acc, w1, w2, b2, 0)

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
    if not incheck:
        static_eval = evaluate(bd, st, mode, tbl_mg, tbl_eg, acc, ply, w1, w2, b2)
        if tt_hit:
            # use the table score as a better static estimate when its bound allows it
            if (tt_flag == TT_LOWER and tt_score > static_eval) or (tt_flag == TT_UPPER and tt_score < static_eval) or tt_flag == TT_EXACT:
                static_eval = tt_score

    if not pv_node and not incheck:
        # reverse futility pruning
        if depth <= 7 and static_eval - 75 * depth >= beta and static_eval < MATE_BOUND:
            return static_eval if has_legal_move(bd, st, undo, hs, mvbuf[ply]) else 0
        # null move pruning
        if can_null and depth >= 3 and static_eval >= beta and has_non_pawn(bd, side) and has_legal_move(bd, st, undo, hs, mvbuf[ply]):
            r = 3 + depth // 3
            if static_eval - beta > 200:
                r += 1
            make_null(bd, st, undo, hs)
            if mode >= 1:
                h = acc.shape[2]
                for k in range(h):
                    acc[ply + 1, 0, k] = acc[ply, 0, k]
                    acc[ply + 1, 1, k] = acc[ply, 1, k]
            score = -negamax(bd, st, undo, hs, mvbuf, scbuf, depth - 1 - r, -beta, -beta + 1, sinfo, tinfo, tt_key, tt_data,
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
    delayed = STAGED_TT and tt_move != 0
    if delayed:
        # A matching full position key supplies a move previously generated by
        # this engine. make_move still rejects king exposure. Do not generate
        # and score the remaining moves unless this move fails to cut off.
        mv[0]=tt_move; n=1
    else:
        n = gen_moves(bd, st, mv, False)
        score_moves(bd, st, mv, sc, n, tt_move, killers, history, ply)

    best = -INF
    best_move = 0
    alpha_orig = alpha
    legal = 0
    futile = (not pv_node) and (not incheck) and depth <= 3 and static_eval + 100 + 120 * depth <= alpha
    lmp_limit = 3 + depth * depth

    for i in range(256):
        if delayed and i==1:
            n=gen_moves(bd,st,mv,False)
            found=-1
            for j in range(n):
                if mv[j]==tt_move: found=j; break
            # A generated hash move remains in slot zero, already searched.
            # An inconsistent TT move should never occur without hash damage;
            # fail visibly rather than silently skipping an unrelated move.
            if found<0: raise RuntimeError('Transposition move absent from move list')
            mv[0],mv[found]=mv[found],mv[0]
            score_moves(bd,st,mv,sc,n,tt_move,killers,history,ply)
            delayed=False
        if i>=n: break
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
        if is_root:
            if sinfo[SI_AVOID_REVISIT] and is_revisit(hs, st[ST_HISTLEN], ply + 1, st[ST_HALF]):
                unmake_move(bd, st, undo, m)
                continue
            sinfo[SI_ROOT_DEPTH] = legal + 1
        legal += 1
        nksq = st[ST_KSQ_W] if st[ST_SIDE] == WHITE else st[ST_KSQ_B]
        gives_check = attacked(bd, nksq, side)
        score = 0
        if legal == 1:
            score = -negamax(bd, st, undo, hs, mvbuf, scbuf, depth - 1, -beta, -alpha, sinfo, tinfo, tt_key, tt_data,
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
            score = -negamax(bd, st, undo, hs, mvbuf, scbuf, depth - 1 - red, -alpha - 1, -alpha, sinfo, tinfo, tt_key, tt_data,
                             killers, history, mode, tbl_mg, tbl_eg, acc, w1, w2, b2, True)
            if score > alpha and red > 0:
                score = -negamax(bd, st, undo, hs, mvbuf, scbuf, depth - 1, -alpha - 1, -alpha, sinfo, tinfo, tt_key, tt_data,
                                 killers, history, mode, tbl_mg, tbl_eg, acc, w1, w2, b2, True)
            if score > alpha and score < beta:
                score = -negamax(bd, st, undo, hs, mvbuf, scbuf, depth - 1, -beta, -alpha, sinfo, tinfo, tt_key, tt_data,
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
