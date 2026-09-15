"""Board core for the agent: 10x12 mailbox, pseudo-legal movegen, make/unmake, Zobrist hashing.

Everything hot is jitted with numba. Layout:
  bd    int8[120]   piece codes, 0 empty, 1..6 = white PNBRQK, 7..12 = black PNBRQK, -1 off board
  st    int32[16]   scalar state, see ST_* below
  undo  int32[MAX_PLY, 4]  captured piece, castle rights, ep square, halfmove counter
  hs    int64[HIST]  zobrist hash per ply (game history first, then search plies)
"""

import numpy as np
from numba import njit

# --- constants -------------------------------------------------------------------------------
EMPTY, OFF = 0, -1
WP, WN, WB, WR, WQ, WK = 1, 2, 3, 4, 5, 6
BP, BN, BB, BR, BQ, BK = 7, 8, 9, 10, 11, 12
WHITE, BLACK = 0, 1

ST_SIDE, ST_EP, ST_CASTLE, ST_HALF, ST_KSQ_W, ST_KSQ_B, ST_PLY, ST_HISTLEN = 0, 1, 2, 3, 4, 5, 6, 7
MAX_PLY = 128
HIST = 1024
ST_NULL = 8  # Active artificial null moves: not part of legal game history.

CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ = 1, 2, 4, 8

# move encoding
M_FROM, M_TO, M_PROMO = 0x7F, 0x7F << 7, 0x7 << 14
F_CAPTURE, F_EP, F_CASTLE, F_DOUBLE = 1 << 17, 1 << 18, 1 << 19, 1 << 20

SQ120 = np.array([21 + (s & 7) + 10 * (s >> 3) for s in range(64)], dtype=np.int32)
SQ64 = np.full(120, -1, dtype=np.int32)
for _s in range(64):
    SQ64[SQ120[_s]] = _s

N_OFF = np.array([-21, -19, -12, -8, 8, 12, 19, 21], dtype=np.int32)
B_OFF = np.array([-11, -9, 9, 11], dtype=np.int32)
R_OFF = np.array([-10, -1, 1, 10], dtype=np.int32)
K_OFF = np.array([-11, -10, -9, -1, 1, 9, 10, 11], dtype=np.int32)

CASTLE_MASK = np.full(120, 15, dtype=np.int32)
CASTLE_MASK[25] = 15 & ~(CASTLE_WK | CASTLE_WQ)  # e1
CASTLE_MASK[21] = 15 & ~CASTLE_WQ  # a1
CASTLE_MASK[28] = 15 & ~CASTLE_WK  # h1
CASTLE_MASK[95] = 15 & ~(CASTLE_BK | CASTLE_BQ)  # e8
CASTLE_MASK[91] = 15 & ~CASTLE_BQ  # a8
CASTLE_MASK[98] = 15 & ~CASTLE_BK  # h8

PIECE_VAL = np.array([0, 100, 320, 330, 500, 900, 20000, 100, 320, 330, 500, 900, 20000], dtype=np.int32)
PIECE_TYPE = np.array([0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6], dtype=np.int32)
PIECE_COLOR = np.array([-1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int32)

_rng = np.random.default_rng(20260904)
Z_PIECE = _rng.integers(-(2**63), 2**63 - 1, size=(13, 120), dtype=np.int64)
Z_CASTLE = _rng.integers(-(2**63), 2**63 - 1, size=16, dtype=np.int64)
Z_EP = _rng.integers(-(2**63), 2**63 - 1, size=120, dtype=np.int64)
Z_SIDE = np.int64(_rng.integers(-(2**63), 2**63 - 1, dtype=np.int64))


def new_state() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bd = np.full(120, OFF, dtype=np.int8)
    for s in range(64):
        bd[SQ120[s]] = EMPTY
    st = np.zeros(16, dtype=np.int32)
    undo = np.zeros((MAX_PLY + 2, 4), dtype=np.int32)
    hs = np.zeros(HIST * 2, dtype=np.int64)
    return bd, st, undo, hs


CHAR_TO_PIECE = {"P": WP, "N": WN, "B": WB, "R": WR, "Q": WQ, "K": WK,
                 "p": BP, "n": BN, "b": BB, "r": BR, "q": BQ, "k": BK}
PIECE_TO_CHAR = ".PNBRQKpnbrqk"


def set_fen(bd: np.ndarray, st: np.ndarray, fen: str) -> None:
    parts = fen.split()
    rows = parts[0].split("/")
    for s in range(64):
        bd[SQ120[s]] = EMPTY
    for r, row in enumerate(rows):
        rank = 7 - r
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                p = CHAR_TO_PIECE[ch]
                bd[SQ120[rank * 8 + f]] = p
                if p == WK:
                    st[ST_KSQ_W] = SQ120[rank * 8 + f]
                elif p == BK:
                    st[ST_KSQ_B] = SQ120[rank * 8 + f]
                f += 1
    st[ST_SIDE] = WHITE if parts[1] == "w" else BLACK
    c = 0
    if "K" in parts[2]:
        c |= CASTLE_WK
    if "Q" in parts[2]:
        c |= CASTLE_WQ
    if "k" in parts[2]:
        c |= CASTLE_BK
    if "q" in parts[2]:
        c |= CASTLE_BQ
    st[ST_CASTLE] = c
    st[ST_EP] = 0
    if parts[3] != "-":
        st[ST_EP] = SQ120[(ord(parts[3][0]) - 97) + 8 * (int(parts[3][1]) - 1)]
    st[ST_HALF] = int(parts[4]) if len(parts) > 4 else 0
    st[ST_PLY] = 0
    st[ST_NULL] = 0


def move_to_uci(m: int) -> str:
    fr = SQ64[m & M_FROM]
    to = SQ64[(m & M_TO) >> 7]
    s = "abcdefgh"[fr & 7] + str((fr >> 3) + 1) + "abcdefgh"[to & 7] + str((to >> 3) + 1)
    promo = (m & M_PROMO) >> 14
    if promo:
        s += "nbrq"[promo - 2]
    return s


# --- attack detection --------------------------------------------------------------------------
@njit(cache=False, nogil=True, error_model="numpy")
def attacked(bd, sq, by):
    """Is square sq attacked by side `by`?"""
    if by == WHITE:
        if bd[sq - 11] == WP or bd[sq - 9] == WP:
            return True
        off = 0
    else:
        if bd[sq + 11] == BP or bd[sq + 9] == BP:
            return True
        off = 6
    kn = WN + off
    kg = WK + off
    bi = WB + off
    ro = WR + off
    qu = WQ + off
    for i in range(8):
        if bd[sq + N_OFF[i]] == kn:
            return True
    for i in range(8):
        if bd[sq + K_OFF[i]] == kg:
            return True
    for i in range(8):
        d = K_OFF[i]
        diag = (d == -11) or (d == -9) or (d == 9) or (d == 11)
        slider = bi if diag else ro
        t = sq + d
        while True:
            p = bd[t]
            if p == OFF:
                break
            if p != EMPTY:
                if p == slider or p == qu:
                    return True
                break
            t += d
    return False


@njit(cache=False, nogil=True, error_model="numpy")
def in_check(bd, st):
    side = st[ST_SIDE]
    ksq = st[ST_KSQ_W] if side == WHITE else st[ST_KSQ_B]
    return attacked(bd, ksq, side ^ 1)


# --- hashing -------------------------------------------------------------------------------------
@njit(cache=False, nogil=True, error_model="numpy")
def legal_ep_exists(bd,st,ep,side):
    if ep==0:return False
    pawn=WP if side==WHITE else BP
    step=10 if side==WHITE else -10
    cap=ep-step
    if bd[ep]!=EMPTY or bd[cap]!=(BP if side==WHITE else WP):return False
    king=st[ST_KSQ_W] if side==WHITE else st[ST_KSQ_B]
    for fr in (cap-1,cap+1):
        if bd[fr]!=pawn:continue
        victim=bd[cap];bd[fr]=EMPTY;bd[cap]=EMPTY;bd[ep]=pawn
        legal=not attacked(bd,king,side^1)
        bd[fr]=pawn;bd[cap]=victim;bd[ep]=EMPTY
        if legal:return True
    return False


@njit(cache=False, nogil=True, error_model="numpy")
def compute_hash(bd, st):
    h = np.int64(0)
    for s in range(64):
        sq = SQ120[s]
        p = bd[sq]
        if p != EMPTY:
            h ^= Z_PIECE[p, sq]
    h ^= Z_CASTLE[st[ST_CASTLE]]
    ep = st[ST_EP]
    if legal_ep_exists(bd,st,ep,st[ST_SIDE]):h ^= Z_EP[ep]
    if st[ST_SIDE] == BLACK:
        h ^= Z_SIDE
    return h


# --- move generation ------------------------------------------------------------------------------
@njit(cache=False, nogil=True, error_model="numpy")
def gen_moves(bd, st, mv, captures_only):
    """Pseudo-legal moves into mv[0:n]; returns n. Captures-only also yields promotions."""
    n = 0
    side = st[ST_SIDE]
    ep = st[ST_EP]
    if side == WHITE:
        lo = 1
        hi = 6
        elo = 7
        ehi = 12
        fwd = 10
        prom_lo = 81
        prom_hi = 88
        dbl_lo = 31
        dbl_hi = 38
        c1 = CASTLE_WK
        c2 = CASTLE_WQ
        ksq0 = 25
        rook = WR
    else:
        lo = 7
        hi = 12
        elo = 1
        ehi = 6
        fwd = -10
        prom_lo = 31
        prom_hi = 38
        dbl_lo = 81
        dbl_hi = 88
        c1 = CASTLE_BK
        c2 = CASTLE_BQ
        ksq0 = 95
        rook = BR
    for s in range(64):
        sq = SQ120[s]
        p = bd[sq]
        if p < lo or p > hi:
            continue
        pt = p - lo + 1
        if pt == 1:
            t = sq + fwd
            is_prom = sq >= prom_lo and sq <= prom_hi
            if bd[t] == EMPTY:
                if is_prom:
                    for pr in range(5, 1, -1):
                        mv[n] = sq | (t << 7) | (pr << 14)
                        n += 1
                elif not captures_only:
                    mv[n] = sq | (t << 7)
                    n += 1
                    if sq >= dbl_lo and sq <= dbl_hi and bd[t + fwd] == EMPTY:
                        mv[n] = sq | ((t + fwd) << 7) | F_DOUBLE
                        n += 1
            for k in range(2):
                t = sq + fwd + (1 if k == 0 else -1)
                q = bd[t]
                if q >= elo and q <= ehi:
                    if is_prom:
                        for pr in range(5, 1, -1):
                            mv[n] = sq | (t << 7) | (pr << 14) | F_CAPTURE
                            n += 1
                    else:
                        mv[n] = sq | (t << 7) | F_CAPTURE
                        n += 1
                elif t == ep and ep != 0:
                    mv[n] = sq | (t << 7) | F_CAPTURE | F_EP
                    n += 1
        elif pt == 2 or pt == 6:
            for i in range(8):
                t = sq + (N_OFF[i] if pt == 2 else K_OFF[i])
                q = bd[t]
                if q == EMPTY:
                    if not captures_only:
                        mv[n] = sq | (t << 7)
                        n += 1
                elif q >= elo and q <= ehi:
                    mv[n] = sq | (t << 7) | F_CAPTURE
                    n += 1
            if pt == 6 and not captures_only and sq == ksq0:
                c = st[ST_CASTLE]
                if (c & c1) and bd[sq + 1] == EMPTY and bd[sq + 2] == EMPTY and bd[sq + 3] == rook:
                    if not attacked(bd, sq, side ^ 1) and not attacked(bd, sq + 1, side ^ 1) and not attacked(bd, sq + 2, side ^ 1):
                        mv[n] = sq | ((sq + 2) << 7) | F_CASTLE
                        n += 1
                if (c & c2) and bd[sq - 1] == EMPTY and bd[sq - 2] == EMPTY and bd[sq - 3] == EMPTY and bd[sq - 4] == rook:
                    if not attacked(bd, sq, side ^ 1) and not attacked(bd, sq - 1, side ^ 1) and not attacked(bd, sq - 2, side ^ 1):
                        mv[n] = sq | ((sq - 2) << 7) | F_CASTLE
                        n += 1
        else:
            d0 = 0 if pt == 3 else (4 if pt == 4 else 0)
            d1 = 4 if pt == 3 else 8
            for i in range(d0, d1):
                d = B_OFF[i] if i < 4 else R_OFF[i - 4]
                t = sq + d
                while True:
                    q = bd[t]
                    if q == EMPTY:
                        if not captures_only:
                            mv[n] = sq | (t << 7)
                            n += 1
                    else:
                        if q >= elo and q <= ehi:
                            mv[n] = sq | (t << 7) | F_CAPTURE
                            n += 1
                        break
                    t += d
    return n


# --- make / unmake ---------------------------------------------------------------------------------
@njit(cache=False, nogil=True, error_model="numpy")
def make_move(bd, st, undo, hs, m):
    """Apply move m. Returns False (and leaves position restored) if it leaves own king in check."""
    ply = st[ST_PLY]
    fr = m & M_FROM
    to = (m & M_TO) >> 7
    promo = (m & M_PROMO) >> 14
    side = st[ST_SIDE]
    p = bd[fr]
    cap = bd[to]
    h = hs[st[ST_HISTLEN] + ply]

    undo[ply, 0] = cap
    undo[ply, 1] = st[ST_CASTLE]
    undo[ply, 2] = st[ST_EP]
    undo[ply, 3] = st[ST_HALF]

    h ^= Z_CASTLE[st[ST_CASTLE]]
    ep = st[ST_EP]
    if legal_ep_exists(bd,st,ep,side):h ^= Z_EP[ep]

    st[ST_HALF] += 1
    if m & F_EP:
        csq = to - 10 if side == WHITE else to + 10
        h ^= Z_PIECE[bd[csq], csq]
        bd[csq] = EMPTY
        undo[ply, 0] = BP if side == WHITE else WP
    elif cap != EMPTY:
        h ^= Z_PIECE[cap, to]
        st[ST_HALF] = 0
    if p == WP or p == BP:
        st[ST_HALF] = 0

    bd[fr] = EMPTY
    h ^= Z_PIECE[p, fr]
    if promo:
        np_ = promo if side == WHITE else promo + 6
        bd[to] = np_
        h ^= Z_PIECE[np_, to]
    else:
        bd[to] = p
        h ^= Z_PIECE[p, to]

    if m & F_CASTLE:
        if to == 27:
            bd[28] = EMPTY
            bd[26] = WR
            h ^= Z_PIECE[WR, 28] ^ Z_PIECE[WR, 26]
        elif to == 23:
            bd[21] = EMPTY
            bd[24] = WR
            h ^= Z_PIECE[WR, 21] ^ Z_PIECE[WR, 24]
        elif to == 97:
            bd[98] = EMPTY
            bd[96] = BR
            h ^= Z_PIECE[BR, 98] ^ Z_PIECE[BR, 96]
        else:
            bd[91] = EMPTY
            bd[94] = BR
            h ^= Z_PIECE[BR, 91] ^ Z_PIECE[BR, 94]

    if p == WK:
        st[ST_KSQ_W] = to
    elif p == BK:
        st[ST_KSQ_B] = to

    st[ST_CASTLE] &= CASTLE_MASK[fr] & CASTLE_MASK[to]
    h ^= Z_CASTLE[st[ST_CASTLE]]

    if m & F_DOUBLE:
        st[ST_EP] = (fr + to) >> 1
        nep = st[ST_EP]
        if legal_ep_exists(bd,st,nep,side^1):h ^= Z_EP[nep]
    else:
        st[ST_EP] = 0

    st[ST_SIDE] = side ^ 1
    h ^= Z_SIDE
    st[ST_PLY] = ply + 1
    hs[st[ST_HISTLEN] + ply + 1] = h
    idx = st[ST_HISTLEN] + ply
    # Commutative fingerprint of prior reversible positions. Used only to
    # validate TT scores; ordinary board hashes still order transpositions.
    hs[HIST + idx + 1] = 0 if st[ST_HALF] == 0 else hs[HIST + idx] + hs[idx]

    ksq = st[ST_KSQ_W] if side == WHITE else st[ST_KSQ_B]
    if attacked(bd, ksq, side ^ 1):
        unmake_move(bd, st, undo, m)
        return False
    return True


@njit(cache=False, nogil=True, error_model="numpy")
def unmake_move(bd, st, undo, m):
    ply = st[ST_PLY] - 1
    st[ST_PLY] = ply
    fr = m & M_FROM
    to = (m & M_TO) >> 7
    promo = (m & M_PROMO) >> 14
    side = st[ST_SIDE] ^ 1
    st[ST_SIDE] = side
    cap = undo[ply, 0]
    st[ST_CASTLE] = undo[ply, 1]
    st[ST_EP] = undo[ply, 2]
    st[ST_HALF] = undo[ply, 3]

    p = bd[to]
    if promo:
        p = WP if side == WHITE else BP
    bd[fr] = p
    bd[to] = EMPTY
    if m & F_EP:
        csq = to - 10 if side == WHITE else to + 10
        bd[csq] = cap
    elif cap != EMPTY:
        bd[to] = cap

    if m & F_CASTLE:
        if to == 27:
            bd[28] = WR
            bd[26] = EMPTY
        elif to == 23:
            bd[21] = WR
            bd[24] = EMPTY
        elif to == 97:
            bd[98] = BR
            bd[96] = EMPTY
        else:
            bd[91] = BR
            bd[94] = EMPTY
    if p == WK:
        st[ST_KSQ_W] = fr
    elif p == BK:
        st[ST_KSQ_B] = fr


@njit(cache=False, nogil=True, error_model="numpy")
def make_null(bd, st, undo, hs):
    ply = st[ST_PLY]
    st[ST_NULL] += 1
    undo[ply, 2] = st[ST_EP]
    undo[ply, 3] = st[ST_HALF]
    h = hs[st[ST_HISTLEN] + ply]
    ep = st[ST_EP]
    if legal_ep_exists(bd,st,ep,st[ST_SIDE]):h ^= Z_EP[ep]
    st[ST_EP] = 0
    st[ST_SIDE] ^= 1
    st[ST_PLY] = ply + 1
    hs[st[ST_HISTLEN] + ply + 1] = h ^ Z_SIDE
    hs[HIST + st[ST_HISTLEN] + ply + 1] = 0


@njit(cache=False, nogil=True, error_model="numpy")
def unmake_null(st, undo):
    st[ST_NULL] -= 1
    ply = st[ST_PLY] - 1
    st[ST_PLY] = ply
    st[ST_EP] = undo[ply, 2]
    st[ST_HALF] = undo[ply, 3]
    st[ST_SIDE] ^= 1


@njit(cache=False, nogil=True, error_model="numpy")
def perft(bd, st, undo, hs, depth, mvbuf):
    if depth == 0:
        return 1
    ply = st[ST_PLY]
    n = gen_moves(bd, st, mvbuf[ply], False)
    total = 0
    for i in range(n):
        m = mvbuf[ply, i]
        if make_move(bd, st, undo, hs, m):
            total += perft(bd, st, undo, hs, depth - 1, mvbuf)
            unmake_move(bd, st, undo, m)
    return total
