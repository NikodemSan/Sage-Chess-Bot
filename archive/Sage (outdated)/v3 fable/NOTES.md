# Sage — AI Chessathon entry

Everything that runs is Python from this zip plus the preinstalled stack (numpy, numba,
python-chess). No third-party engine source, wrapper, port or network is included.

## Layout
- `agent.py`     the `get_move(fen, time_left_ms)` contract, clock management, warm-up at import,
                 game-history tracking for repetition detection, and an engine-free fallback.
- `zboard.py`    10x12 mailbox board, pseudo-legal move generation, make/unmake, Zobrist hashing.
- `zsearch.py`   iterative deepening, aspiration windows, PVS alpha-beta, transposition table,
                 null move, reverse futility, futility, late move reductions/pruning, static
                 exchange evaluation (swap algorithm), killers, history, countermoves, check
                 extension, quiescence with delta and SEE pruning, contempt-aware draw scoring,
                 the 600-ply cap modelled in search.
- `zengine.py`   owns the arrays, runs the search under soft/hard time limits, keeps game history.
- `zeval.py`     NNUE adapter: lazily updated accumulators, integer SCReLU head.
- `sl_spec.py`   the frozen feature/numeric contract the network was trained against.
- `weights.npz`  the network (int16), trained from scratch by the team; its `metadata` field
                 records the contract, dataset hash and training step count.

## Evaluation
Residual NNUE: a small analytical material + piece-square baseline plus a network correction.
Inputs are 8 king-bucketed piece-square features per perspective (6144), one hidden layer of 256
per side with SCReLU, and 8 output buckets by piece count. Quantised to int16 (QA=255, QB=4096).
The network was trained from scratch on positions labelled with a search-based evaluation; only
the trained weights ship.

## Resources
One thread. Transposition table of 2^24 entries (256 MB), pre-touched at import so no page
faults land on the clock. About 550 MB resident. JIT compilation happens at import, inside the
init budget (~30 s on one core), and every jitted path is warmed before `ready`.
