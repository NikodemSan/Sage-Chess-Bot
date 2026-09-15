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

- 2^21-entry direct transposition table (about 32 MiB across key+data arrays) with persistent
  generations across our moves. The earlier 2^23 table was reverted after target-EPYC A/B tests.
- 2^20-entry exact static-evaluation cache (about 12 MiB across key+value arrays) keyed by Fable's
  64-bit Zobrist hash. Values are int32 and are exactly the existing NNUE+PSQT score.
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


## v5 search-selectivity patch

- Keeps v4 rule/repetition/endgame/Syzygy/opening infrastructure.
- Replaces unlimited recursive in-check qsearch with a bounded legal-evasion frontier after two qsearch checking plies. Quiet legal evasions become static leaves; tactical capture/promotion evasions continue normally.
- Slightly increases time spent on PV/score instability while keeping stable-position stopping close to the older proven engine.
- No neural or PSQT weights were changed.
- Restores the transposition table to 2^21 entries (the older engine's size) after fixed-depth A/B tests on the target EPYC showed materially smaller/faster trees than v4's 2^23 table on several curated starts.
- Reduces the static-eval cache to 2^20 entries; fixed-depth node counts were unchanged versus 2^22, with lower memory/cache pressure.

## v6 competition capture-history patch

Capture history is deliberately small and used only for **main-search move ordering**. It is not a
replacement for SEE, is never consulted by qsearch, and never directly prunes a move. The table is
`int16[13,120,7]` (about 22 KiB), indexed by moved piece, destination and captured piece type.

- MVV-LVA remains dominant. Capture history is only a weak tie/near-tie term inside the existing
  cheap-SEE good/bad buckets, so noisy history cannot promote a tactically losing capture over a
  good capture.
- A capture causing a beta cutoff receives a bounded depth-squared bonus; previously searched
  captures receive half-strength negative evidence. Updates use integer history gravity and are
  clamped to +/-4096, avoiding whole-table rescales in the hot path.
- The table is halved before each root search and cleared on `new_game()`. This preserves useful
  within-game ordering information while quickly forgetting context from earlier positions.
- On the 33 curated-seed fixed-depth-11 suite on the target EPYC, this variant searched 5,116,191
  nodes versus 5,323,564 for the SEE-rescue v5 baseline (-3.9%) and completed the suite in 8.163 s
  versus 8.383 s (-2.6%). Per-node speed fell about 1.3%, so the gain comes from better ordering,
  not a benchmark trick. This is promising search-efficiency evidence, not a claim of proven Elo.
