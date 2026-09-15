"""Evaluation: a tapered material + piece-square baseline, and an NNUE (768 -> H x2 -> 1).

The NNUE is a perspective net: two accumulators (white view, black view) over 768 binary
features (12 piece kinds x 64 squares), updated incrementally on make. Output is SCReLU
activations dotted with the output layer, side to move first. Quantisation: QA=255 for the
first layer / accumulators (int16), QB=64 for the output layer, eval scale 400 cp.
"""

import numpy as np
from numba import njit

from zboard import (
    BK, BP, BR, EMPTY, F_CASTLE, F_EP, M_FROM, M_PROMO, M_TO, PIECE_TYPE, SQ64, SQ120, ST_KSQ_B,
    ST_KSQ_W, ST_SIDE, WHITE, WK, WP, WR,
)

QA = 255
QB = 64
NN_SCALE = 400

# --- classical baseline ----------------------------------------------------------------------------
MAT_MG = np.array([0, 82, 337, 365, 477, 1025, 0], dtype=np.int32)
MAT_EG = np.array([0, 94, 281, 297, 512, 936, 0], dtype=np.int32)
PHASE_INC = np.array([0, 0, 1, 1, 2, 4, 0], dtype=np.int32)


def _tbl(rows: list[list[int]]) -> np.ndarray:
    # rows are given rank 8 first (as on a diagram); convert to a1=0 ordering
    a = np.array(rows, dtype=np.int32)
    return a[::-1].reshape(64)


# Hand-written starting tables (own values); refined later by regression on labelled data.
PST_MG = np.zeros((7, 64), dtype=np.int32)
PST_EG = np.zeros((7, 64), dtype=np.int32)
PST_MG[1] = _tbl([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [60, 80, 60, 70, 65, 70, 30, 0],
    [0, 10, 25, 35, 40, 50, 25, -10],
    [-10, 5, 5, 20, 25, 10, 15, -20],
    [-20, -5, -5, 15, 20, 5, 5, -25],
    [-25, -5, -10, -5, 0, 0, 25, -15],
    [-30, 0, -15, -25, -20, 15, 30, -20],
    [0, 0, 0, 0, 0, 0, 0, 0]])
PST_EG[1] = _tbl([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [150, 150, 140, 120, 130, 120, 150, 160],
    [80, 90, 70, 55, 45, 45, 70, 75],
    [25, 20, 10, 0, -5, 0, 10, 15],
    [10, 5, -5, -10, -10, -10, 0, 0],
    [0, 5, -5, 0, 0, -5, 0, -5],
    [10, 5, 5, 10, 10, 0, 0, -5],
    [0, 0, 0, 0, 0, 0, 0, 0]])
PST_MG[2] = _tbl([
    [-160, -80, -30, -45, 55, -90, -15, -100],
    [-70, -35, 65, 30, 20, 55, 5, -15],
    [-45, 55, 35, 60, 80, 120, 70, 40],
    [-5, 15, 15, 50, 35, 65, 15, 20],
    [-10, 5, 15, 10, 25, 15, 20, -5],
    [-20, -5, 10, 10, 15, 15, 20, -15],
    [-25, -50, -10, 0, 0, 15, -10, -15],
    [-100, -20, -55, -30, -15, -25, -15, -20]])
PST_EG[2] = _tbl([
    [-55, -35, -10, -25, -30, -25, -60, -95],
    [-25, -5, -25, 0, -10, -25, -25, -50],
    [-25, -20, 10, 10, 0, -10, -20, -40],
    [-15, 5, 20, 20, 20, 10, 5, -15],
    [-15, -5, 15, 25, 15, 15, 5, -15],
    [-25, -5, 0, 15, 10, -5, -20, -20],
    [-40, -20, -10, -5, 0, -20, -25, -45],
    [-30, -50, -25, -15, -20, -20, -50, -65]])
PST_MG[3] = _tbl([
    [-30, 5, -80, -35, -25, -40, 5, -10],
    [-25, 15, -20, -15, 30, 60, 20, -45],
    [-15, 35, 45, 40, 35, 50, 35, 0],
    [-5, 5, 20, 50, 35, 35, 5, 0],
    [-5, 15, 15, 25, 35, 10, 10, 5],
    [0, 15, 15, 15, 15, 25, 20, 10],
    [5, 15, 15, 0, 5, 20, 35, 0],
    [-35, 0, -15, -20, -15, -10, -40, -20]])
PST_EG[3] = _tbl([
    [-15, -20, -10, -10, -5, -10, -15, -25],
    [-10, -5, 5, -10, -5, -15, -5, -15],
    [0, -10, 0, 0, 0, 5, 0, 5],
    [-5, 10, 10, 10, 15, 10, 5, 0],
    [-5, 5, 15, 20, 5, 10, -5, -10],
    [-10, -5, 10, 10, 15, 5, -5, -15],
    [-15, -20, -5, 0, 5, -10, -15, -25],
    [-25, -10, -25, -5, -10, -15, -5, -15]])
PST_MG[4] = _tbl([
    [30, 40, 30, 50, 60, 10, 30, 45],
    [25, 30, 60, 60, 80, 65, 25, 45],
    [-5, 20, 25, 35, 15, 45, 60, 15],
    [-25, -10, 5, 25, 25, 35, -10, -20],
    [-35, -25, -10, 0, 10, -5, 5, -25],
    [-45, -25, -15, -15, 5, 0, -5, -35],
    [-45, -15, -20, -10, 0, 10, -5, -70],
    [-20, -15, 0, 15, 15, 5, -35, -25]])
PST_EG[4] = _tbl([
    [15, 10, 20, 15, 10, 10, 10, 5],
    [10, 15, 15, 10, -5, 5, 10, 5],
    [5, 5, 5, 5, 5, -5, -5, -5],
    [5, 5, 15, 0, 0, 0, 0, 0],
    [5, 5, 10, 5, -5, -5, -10, -10],
    [-5, 0, -5, 0, -5, -10, -10, -15],
    [-5, -5, 0, 0, -10, -10, -10, -5],
    [-10, 0, 5, 0, -5, -15, 5, -20]])
PST_MG[5] = _tbl([
    [-30, 0, 30, 10, 60, 45, 45, 45],
    [-25, -40, -5, 0, -15, 55, 30, 55],
    [-15, -15, 5, 10, 30, 55, 45, 55],
    [-25, -25, -15, -10, 0, 15, 0, 0],
    [-10, -25, -10, -10, 0, -5, 5, 0],
    [-15, 0, -10, 0, -5, 0, 15, 5],
    [-35, -10, 10, 0, 10, 15, 0, 0],
    [0, -20, -10, 10, -15, -25, -30, -50]])
PST_EG[5] = _tbl([
    [-10, 20, 20, 25, 25, 20, 10, 20],
    [-15, 20, 30, 40, 60, 25, 30, 0],
    [-20, 5, 10, 50, 45, 35, 20, 10],
    [5, 20, 25, 45, 55, 40, 55, 35],
    [-20, 30, 20, 45, 30, 35, 40, 25],
    [-15, -25, 15, 5, 10, 15, 10, 5],
    [-20, -25, -30, -15, -15, -25, -35, -30],
    [-35, -30, -20, -45, -5, -30, -20, -40]])
PST_MG[6] = _tbl([
    [-65, 25, 15, -15, -55, -35, 0, 15],
    [30, 0, -20, -5, -10, -5, -40, -30],
    [-10, 25, 0, -15, -20, 5, 20, -20],
    [-15, -20, -10, -25, -30, -25, -15, -35],
    [-50, 0, -25, -40, -45, -45, -35, -50],
    [-15, -15, -20, -45, -45, -30, -15, -25],
    [0, 5, -10, -65, -45, -15, 10, 10],
    [-15, 35, 10, -55, 10, -30, 25, 15]])
PST_EG[6] = _tbl([
    [-75, -35, -20, -20, -10, 15, 5, -15],
    [-10, 15, 15, 15, 15, 40, 25, 10],
    [10, 15, 25, 15, 20, 45, 45, 15],
    [-10, 20, 25, 25, 25, 35, 25, 5],
    [-20, -5, 20, 25, 25, 25, 10, -10],
    [-20, -5, 10, 20, 25, 15, 5, -10],
    [-25, -10, 5, 15, 15, 5, -5, -15],
    [-55, -35, -20, -10, -30, -15, -25, -45]])

# combined per-square tables indexed by piece code (1..12) and sq64, white-relative sign applied
TBL_MG = np.zeros((13, 64), dtype=np.int32)
TBL_EG = np.zeros((13, 64), dtype=np.int32)
for _p in range(1, 7):
    for _s in range(64):
        TBL_MG[_p, _s] = MAT_MG[_p] + PST_MG[_p, _s]
        TBL_EG[_p, _s] = MAT_EG[_p] + PST_EG[_p, _s]
        TBL_MG[_p + 6, _s ^ 56] = -(MAT_MG[_p] + PST_MG[_p, _s])
        TBL_EG[_p + 6, _s ^ 56] = -(MAT_EG[_p] + PST_EG[_p, _s])


@njit(cache=False, nogil=True, error_model="numpy")
def eval_classical(bd, st, tbl_mg, tbl_eg):
    mg = 0
    eg = 0
    phase = 0
    wnp = 0
    bnp = 0
    wp = 0
    bp = 0
    for s in range(64):
        p = bd[SQ120[s]]
        if p == EMPTY:
            continue
        mg += tbl_mg[p, s]
        eg += tbl_eg[p, s]
        t = PIECE_TYPE[p]
        phase += PHASE_INC[t]
        if p == WP:
            wp += 1
        elif p == BP:
            bp += 1
        elif t != 6:
            if p < 7:
                wnp += MAT_EG[t]
            else:
                bnp += MAT_EG[t]
    if wp == 0 and bp == 0:
        # insufficient material: bare minor pieces cannot win
        if wnp <= 300 and bnp <= 300:
            return 0
        # lone minor(s) vs nothing when the stronger side has less than a rook: dampen
        if (bnp == 0 and wnp < 500) or (wnp == 0 and bnp < 500):
            return 0
    if phase > 24:
        phase = 24
    score = (mg * phase + eg * (24 - phase)) // 24
    return score if st[ST_SIDE] == WHITE else -score


# --- NNUE ---------------------------------------------------------------------------------------------
@njit(cache=False, nogil=True, error_model="numpy")
def feat_index(p, s64, persp):
    """Feature index of piece code p on square s64 from perspective persp (0 white, 1 black)."""
    if persp == WHITE:
        return (p - 1) * 64 + s64
    q = p - 6 if p >= 7 else p + 6
    return (q - 1) * 64 + (s64 ^ 56)


@njit(cache=False, nogil=True, error_model="numpy")
def nn_refresh(bd, acc, ply, w1, b1):
    """Recompute both accumulators for the current board into acc[ply]."""
    h = w1.shape[1]
    for k in range(h):
        acc[ply, 0, k] = b1[k]
        acc[ply, 1, k] = b1[k]
    for s in range(64):
        p = bd[SQ120[s]]
        if p == EMPTY:
            continue
        iw = feat_index(p, s, 0)
        ib = feat_index(p, s, 1)
        for k in range(h):
            acc[ply, 0, k] += w1[iw, k]
            acc[ply, 1, k] += w1[ib, k]


@njit(cache=False, nogil=True, error_model="numpy")
def nn_push(bd, st, m, acc, ply, w1):
    """Before make_move: write acc[ply+1] = acc[ply] updated for move m (reads the pre-move board)."""
    h = w1.shape[1]
    np1 = ply + 1
    for k in range(h):
        acc[np1, 0, k] = acc[ply, 0, k]
        acc[np1, 1, k] = acc[ply, 1, k]
    fr = m & M_FROM
    to = (m & M_TO) >> 7
    promo = (m & M_PROMO) >> 14
    side = st[ST_SIDE]
    p = bd[fr]
    cap = bd[to]
    f64 = SQ64[fr]
    t64 = SQ64[to]
    # up to 4 feature changes: (piece, square, sign)
    cp0 = p
    cs0 = f64
    cg0 = -1
    cp1 = (promo if side == WHITE else promo + 6) if promo else p
    cs1 = t64
    cg1 = 1
    cp2 = 0
    cs2 = 0
    cg2 = 0
    cp3 = 0
    cs3 = 0
    cg3 = 0
    if m & F_EP:
        cp2 = BP if side == WHITE else WP
        cs2 = SQ64[to - 10] if side == WHITE else SQ64[to + 10]
        cg2 = -1
    elif cap != EMPTY:
        cp2 = cap
        cs2 = t64
        cg2 = -1
    if m & F_CASTLE:
        if to == 27:
            cp2 = WR
            cs2 = 7
            cg2 = -1
            cp3 = WR
            cs3 = 5
            cg3 = 1
        elif to == 23:
            cp2 = WR
            cs2 = 0
            cg2 = -1
            cp3 = WR
            cs3 = 3
            cg3 = 1
        elif to == 97:
            cp2 = BR
            cs2 = 63
            cg2 = -1
            cp3 = BR
            cs3 = 61
            cg3 = 1
        else:
            cp2 = BR
            cs2 = 56
            cg2 = -1
            cp3 = BR
            cs3 = 59
            cg3 = 1
    for j in range(4):
        if j == 0:
            cp = cp0
            cs = cs0
            cg = cg0
        elif j == 1:
            cp = cp1
            cs = cs1
            cg = cg1
        elif j == 2:
            cp = cp2
            cs = cs2
            cg = cg2
        else:
            cp = cp3
            cs = cs3
            cg = cg3
        if cg == 0:
            continue
        iw = (cp - 1) * 64 + cs
        q = cp - 6 if cp >= 7 else cp + 6
        ib = (q - 1) * 64 + (cs ^ 56)
        if cg > 0:
            for k in range(h):
                acc[np1, 0, k] += w1[iw, k]
                acc[np1, 1, k] += w1[ib, k]
        else:
            for k in range(h):
                acc[np1, 0, k] -= w1[iw, k]
                acc[np1, 1, k] -= w1[ib, k]


@njit(cache=False, nogil=True, error_model="numpy")
def nn_eval(acc, ply, side, w2, b2):
    """SCReLU output: side to move perspective first. Returns centipawns for side to move.

    v <= 255 and |w2| < 128 so each product fits in int32; chunks of 64 are summed in int32
    (worst case 64 * 255^2 * 127 < 2^31) and folded into an int64 total.
    """
    h = acc.shape[2]
    total = np.int64(0)
    them = side ^ 1
    for chunk in range(0, h, 64):
        part = np.int32(0)
        for k in range(chunk, chunk + 64):
            v = np.int32(acc[ply, side, k])
            if v < 0:
                v = 0
            elif v > QA:
                v = QA
            part += v * v * np.int32(w2[k])
        total += part
        part = np.int32(0)
        for k in range(chunk, chunk + 64):
            v = np.int32(acc[ply, them, k])
            if v < 0:
                v = 0
            elif v > QA:
                v = QA
            part += v * v * np.int32(w2[h + k])
        total += part
    total = total // QA + np.int64(b2)
    return (total * NN_SCALE) // (QA * QB)


CENTER_DIST = np.zeros(64, dtype=np.int32)
for _s in range(64):
    _f = _s & 7
    _r = _s >> 3
    CENTER_DIST[_s] = max(abs(2 * _f - 7), abs(2 * _r - 7)) // 2  # 0 centre .. 3 edge


@njit(cache=False, nogil=True, error_model="numpy")
def evaluate(bd, st, mode, tbl_mg, tbl_eg, acc, ply, w2, b2):
    """mode 0: classical tables; 1: NNUE; 2: blend of both. Includes draw and mop-up knowledge."""
    wp = 0
    bp = 0
    wnp = 0
    bnp = 0
    for s in range(64):
        p = bd[SQ120[s]]
        if p == EMPTY:
            continue
        if p == WP:
            wp += 1
        elif p == BP:
            bp += 1
        elif p != WK and p != BK:
            if p < 7:
                wnp += MAT_EG[p]
            else:
                bnp += MAT_EG[p - 6]
    if wp == 0 and bp == 0 and ((wnp <= 300 and bnp <= 300) or (bnp == 0 and wnp < 500) or (wnp == 0 and bnp < 500)):
        return 0
    if mode == 0:
        v = eval_classical(bd, st, tbl_mg, tbl_eg)
    elif mode == 1:
        v = nn_eval(acc, ply, st[ST_SIDE], w2, b2)
    else:
        v = (5 * nn_eval(acc, ply, st[ST_SIDE], w2, b2) + 3 * eval_classical(bd, st, tbl_mg, tbl_eg)) // 8
    # mop-up: the side with a decisive material edge and the opponent without pawns drives the
    # enemy king to the edge and brings its own king closer (white view, then side to move)
    if (bp == 0 and wnp >= bnp + 450) or (wp == 0 and bnp >= wnp + 450):
        wk = SQ64[st[ST_KSQ_W]]
        bk = SQ64[st[ST_KSQ_B]]
        kd = max(abs((wk & 7) - (bk & 7)), abs((wk >> 3) - (bk >> 3)))
        if bp == 0 and wnp >= bnp + 450:
            mop = 10 * CENTER_DIST[bk] + 4 * (7 - kd)
        else:
            mop = -(10 * CENTER_DIST[wk] + 4 * (7 - kd))
        v += mop if st[ST_SIDE] == WHITE else -mop
    return v


def dummy_nn(h: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w1 = np.zeros((768, h), dtype=np.int16)
    b1 = np.zeros(h, dtype=np.int16)
    w2 = np.zeros(2 * h, dtype=np.int16)
    b2 = np.int32(0)
    return w1, b1, w2, b2
