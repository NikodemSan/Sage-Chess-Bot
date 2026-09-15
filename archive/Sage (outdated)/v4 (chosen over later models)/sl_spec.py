"""Frozen, original feature and numeric contract. No published network weights."""
import hashlib
import json
import numpy as np

VERSION = 'sl-v1'
PAD = 768
QA, QB, CP_SCALE = 255, 4096, 400
SPECS = {
    'sage': dict(name='sage', inputs=6144, hidden=256, king_buckets=8, output_buckets=8),
    'lemma': dict(name='lemma', inputs=768, hidden=512, king_buckets=1, output_buckets=8),
}
PHASE = np.array([0, 1, 1, 2, 4, 0] * 2 + [0], dtype=np.int32)

def baseline_tables():
    """Small fixed analytical baseline; all values newly chosen for this lab."""
    mg, eg = np.zeros(769, np.int32), np.zeros(769, np.int32)
    material = [100, 320, 330, 500, 900, 0]
    for t in range(6):
        for sq in range(64):
            f, r = sq % 8, sq // 8
            center = 7 - abs(2*f-7) - abs(2*r-7)
            if t == 0:
                a, b = 4*r + 2*(7-abs(2*f-7)), 9*r
            elif t == 1:
                a, b = 5*center, 3*center
            elif t == 2:
                a, b = 3*center, 2*center
            elif t == 3:
                a, b = 2*r, center
            elif t == 4:
                a, b = center, 2*center
            else:
                a, b = -4*center - 5*r, 5*center
            mg[t*64+sq], eg[t*64+sq] = material[t]+a, material[t]+b
            mg[(t+6)*64+(sq^56)] = -mg[t*64+sq]
            eg[(t+6)*64+(sq^56)] = -eg[t*64+sq]
    return mg, eg

MG, EG = baseline_tables()
BASELINE_HASH = hashlib.sha256(MG.tobytes()+EG.tobytes()).hexdigest()

def contract(name):
    return dict(SPECS[name], version=VERSION, qa=QA, qb=QB, cp_scale=CP_SCALE,
                baseline_sha256=BASELINE_HASH, orientation='black: colour-swap, rank-flip XOR56; no file mirror',
                king_region='2*(rank//2)+(file//4)', output_region='min(7,max(0,(pieces-2)//4))',
                output='B_stm + 400 * (dot(SCReLU(acc_stm,acc_other), head[bucket])+bias[bucket])')

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def features(pieces, stm, kings, name):
    """pieces Bx32: absolute colour/type/square indices, padded with 768."""
    p = np.asarray(pieces, dtype=np.int64)
    valid = p != PAD
    black = ((p//64+6)%12)*64 + ((p%64)^56)
    kb = SPECS[name]['king_buckets']
    if kb == 8:
        wk = np.asarray(kings, dtype=np.int64)[:, 0]
        bk = np.asarray(kings, dtype=np.int64)[:, 1] ^ 56
        wr = 2*(wk//8//2)+(wk%8//4)
        br = 2*(bk//8//2)+(bk%8//4)
    else:
        wr = br = np.zeros(len(p), dtype=np.int64)
    # Training padding index is the final embedding row (inputs).
    w = np.where(valid, wr[:, None]*768+p, SPECS[name]['inputs'])
    b = np.where(valid, br[:, None]*768+black, SPECS[name]['inputs'])
    side = np.asarray(stm, dtype=bool)[:, None]
    first, second = np.where(side, b, w), np.where(side, w, b)
    phase = np.minimum(24, PHASE[p//64].sum(1))
    baseline = (MG[p].sum(1)*phase + EG[p].sum(1)*(24-phase)) // 24
    baseline = np.where(np.asarray(stm)==0, baseline, -baseline).astype(np.float32)
    out = np.minimum(7, np.maximum(0, (valid.sum(1)-2)//4)).astype(np.int64)
    return first, second, baseline, out
