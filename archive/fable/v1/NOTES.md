# Engine notes

Everything the agent plays with is in this directory; nothing else is shipped.

| File | What it is |
|---|---|
| `agent.py` | Entry point. Clock management, legality guard, pondering on the opponent's clock. |
| `zboard.py` | 10x12 mailbox board, pseudo-legal move generation, make/unmake, incremental Zobrist hashing. Verified by perft on the six standard test positions. |
| `zsearch.py` | Iterative deepening, aspiration windows, PVS alpha-beta, 2M-entry transposition table, null-move pruning, reverse futility, futility, late move reductions/pruning, killers + history (with malus), check extension, quiescence with delta pruning, repetition and 50-move detection, draw contempt. |
| `zeval.py` | Evaluation: NNUE inference (incremental accumulators), tapered material + piece-square tables, blend, mop-up. |
| `zengine.py` | Driver: iterative deepening loop, time budget, game-history tracking for repetition, ponder thread. |
| `weights/nnue.npz` | The trained network (int16 / int32 arrays, numpy `savez`). |
| `weights/psqt.npz` | Texel-tuned piece-square tables (int32 arrays). |

All hot code is `numba.njit` (`nogil=True`); the whole thing compiles at import in about 24 s on a 2.1 GHz Xeon core, well inside the init budget. Speed: ~700k nodes/s with the classical eval, ~500k with the network, depth 12-14 in a few seconds from a middlegame position.

## Evaluation

The network is a perspective NNUE, `768 -> 256 x2 -> 1`:

* Inputs: 768 binary features (12 piece kinds x 64 squares) seen from White's side and, mirrored, from Black's side — two accumulators of 256 int16 each, updated incrementally on every move (at most four feature changes per move).
* Activation: SCReLU (clipped ReLU, squared). Output layer over the concatenated `[side to move, other side]` accumulators.
* Quantisation: first layer and accumulators x255 (int16), output layer x64, output scale 400 cp. Products accumulate in int32 chunks so LLVM vectorises the output dot product.

The shipped eval is a blend: `(5 * nnue + 3 * psqt) / 8`, plus a mop-up term when one side has a decisive material edge against a pawnless opponent, plus insufficient-material draws. See "Measurements" for why the blend.

## Data and training (not shipped)

Training data was generated with Stockfish 18 as a labelling tool, which the rules allow ("training data is unrestricted, including positions annotated by an existing engine"). No engine, book, or lookup table of engine moves ships with the agent.

* `tools/gen_data.py`: gensfen-style self-play. Random opening plies (2-10), then Stockfish plays both sides at depth 6-8 with occasional random moves for material-imbalance diversity; every position is recorded with the search score (side-to-move relative), quietness flags, and the game result. ~1.3M positions.
* `tools/gen_endgame.py`: Stockfish play-outs from random legal endgame positions (depth 8-10), so the net sees conversions and mates. ~140k positions.
* Filtering follows the NNUE-dataset literature (arXiv 2412.17948): quiet positions only (not in check, best move not a capture/promotion), mate scores kept and clipped to +-3000, decisive positions subsampled so ~1/3 of the data is within +-100 cp.
* `tools/train_nnue.py`: NumPy trainer (no PyTorch available in the build sandbox). Adam, batch 4096, loss `MSE(sigmoid(out), sigmoid(cp/400))`, 40 epochs with exponential LR decay. ~13 s per epoch on one core for 800k positions.
* `tools/tune_psqt.py`: Texel tuning of the tapered tables with the same loss.

## Measurements

In-process A/B matches, alternating colours, near-level openings, 2s + 0.05s per game unless stated (`tools/ab.py`; error bars are ~+-81 Elo at 24 games).

| Configuration | vs hand-written tapered PSQT |
|---|---|
| NNUE v3 (454k positions) | -29 |
| NNUE v4 (737k positions) | -44 |
| NNUE v5 (H=128) | -120 |
| Texel-tuned PSQT | -29 |
| NNUE v4, **fixed depth 7 both sides** | **+89** |
| **Blend 5/8 NNUE v4 + 3/8 tuned PSQT, mop-up** | **+211** (+18 =1 -5) |

The fixed-depth row shows the network evaluates better than the tables inside a search; its time-control parity was the ~1.7x cost per ply of the accumulator updates. Blending recovers the gain: averaging two roughly independent estimates cancels evaluation noise (which alpha-beta amplifies, since it maximises over many noisy estimates), the tables keep material strictly monotone, and the net supplies positional knowledge.

## Clock

`zengine.budget`: soft budget `time_left / max(18, 40 - ply/4) + 0.8 * increment`, hard cap 4x soft and 40% of the remaining clock; the iterative deepening loop does not start a new depth after half the soft budget is spent, and the search aborts at the hard limit (checked every 2048 nodes). While the opponent thinks, a background thread searches the position after our move (`nogil` numba code), so the transposition table is warm when our turn comes; the thread is stopped before the next search starts.
