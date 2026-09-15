# Fable v4 — maxed current-weights tournament build

## Current competition assumptions

Built for the live AI Chessathon participant specification on 5 September 2026: Python 3.12,
one AMD EPYC 9V74 core, 2 GB RAM, no network/GPU, 90 s import/init budget, 120 s + 0.5 s per
move, 50 MB uncompressed submission cap, FIDE draw rules, and a 600-ply referee draw cap.
The process is suspended while the opponent moves, so this build contains no pondering thread or
background-opponent-time work.

## Evaluator preservation

The evaluator is intentionally unchanged. `weights/nnue.npz` and `weights/psqt.npz` are byte-for-
byte the trained files supplied by the team, and the runtime score remains the same 5:3 NNUE/PSQT
blend. Search, cache, book, tablebase and clock work are wrappers around that evaluator rather than
replacement weights.

## Search / memory

- 128 MiB direct transposition table with persistent generations across our moves.
- ~48 MiB exact static-evaluation cache keyed by Fable's 64-bit Zobrist hash. Values are int32 and
  are exactly the existing NNUE+PSQT score, never an approximation.
- Iterative deepening, aspiration windows, PVS alpha-beta, LMR/LMP, null-move pruning, reverse
  futility/futility pruning, killer/history ordering, qsearch and mate-distance pruning.
- Conservative null-move guard disables NMP in classic minor/pawn zugzwang-prone endings.
- Sparse (<=7 piece) qsearch disables risky delta/SEE pruning and recognises stalemate exactly.
- Wall-clock aborts use adaptive checking and return the last safe completed root result.

## Rules / correctness fixes retained

- Legal en-passant Zobrist semantics.
- Threefold logic distinguishes one previous historical occurrence from a real third occurrence,
  while treating a cycle created inside the current search path as draw.
- 50-move/repetition handling cannot mask checkmate.
- Forced single-move searches correctly update `last_move`.
- TT mate scores are ply-normalised on store/probe.

## Chessathon-specific preparation

`zopenings.py` contains 33 public candidate starting FENs whose opening families recur on public
Chessathon game pages. During free initialization Fable analyses these positions with its own
trained evaluator/search and keeps the TT entries. The initialization work has an absolute
55-second-from-import cutoff. Each seed is capped at 250 ms; a cold-cache development import completed
in about 25 seconds, leaving a large safety margin under the 90-second referee limit.

`zbook.py` is a genuine opening book generated offline by this exact Fable engine around those
candidate starts. It contains moves only — no third-party evaluations, no Stockfish lookup table.
The first layer covers all 33 candidate starts; the second covers every legal first opponent reply;
a selective third layer covers the two strongest first replies according to Fable's own analysis.
Book use is restricted to the first three Fable moves and every move is live-legality checked.
Because the clock increment arrives after a move, prepared book moves also preserve/bank almost all
of the +0.5 s increment.

## Endgames

The `syzygy/` directory ships the complete orthodox 3-piece WDL classes K+P/R/Q/B/N vs K. The
current rules explicitly permit tablebases. Fable probes these only as an exact W/D/L guard: it
briefly searches with its own evaluator, accepts that move if it preserves optimal WDL, and vetoes
it only if it throws away the theoretical result. WDL-only is deliberate; unverified DTZ files are
not shipped.

## SEE hardening

The old one-recapture/"is destination defended" approximation has been replaced by a mailbox-native,
iterative legal swap-off SEE. It follows least-valuable legal recaptures on the target square, updates
virtual occupancy after every exchange so slider x-rays appear naturally, rejects pinned/illegal king
recaptures, and accounts for en-passant removal and promotion material. It does not mutate the live
board or state.

Search keeps the existing cheap material bounds before calling SEE: an attacker no more valuable than
its victim is already provably non-losing at a zero threshold, and the qsearch equivalent uses a -50cp
margin. This avoids paying for SEE on obvious captures. Qsearch also preserves checking sacrifices even
when local SEE is negative, because a one-square material model must not forward-prune a forcing check.
