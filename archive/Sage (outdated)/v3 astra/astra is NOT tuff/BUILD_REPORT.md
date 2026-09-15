# Build report — Sage Fusion

Built 7 September 2026 from the two user-supplied archives.

## Result and scope

The final default is the supplied 150M-sample network, the supplied selective search with targeted safeguards, a larger cache, and complete 3/4-piece tablebases. The 50M checkpoint and optional 75/25 ensemble are retained for local experiments. No new training was performed.

No third-party engine source, compiled extension, executable, hosted inference, training-position lookup database, or published neural weights were introduced. All runtime Python derives from the supplied team code and changes made with Codex assistance. Endgame DATA is Syzygy, explicitly permitted by the event.

## Resource and API verification

- Actual entry-point import, on one local CPU: **22.187 seconds**, below the 90-second budget.
- Measured peak resident memory: **818.5 MiB**, including a **512 MiB** transposition table and mapped endgame tables.
- The cold first-use move-history compile was found and moved into import warmup.
- Legal outputs returned within every tested clock: 79, 100, 200, 500, 1,000, 3,000 and 120,000 ms.
- Injected search exception and injected illegal proposal both produced legal fallback moves.
- Persistent API smoke sequences passed as both colours.
- Python 3.12; NumPy 2.5.2; Numba 0.67.0; chess 1.11.2. Torch/ONNX Runtime are not imported.
- Tests ran on local hardware with CPU affinity limited to one logical CPU; it is not a certification on the organiser’s AMD EPYC. Peak RSS is observed usage, not an enforced organiser cgroup test.

## Chess correctness checks

- Five perft positions: **410,227 leaf nodes** with exact expected counts.
- **1,660 random positions** compared with python-chess for legal move sets; make/unmake, incremental hashes and deferred neural accumulators checked.
- 100 neural evaluations matched an independent NumPy implementation of the supplied training feature contract exactly.
- Legal en-passant hashing, null/repetition horizons, budget ordering, and two mate-in-one regressions passed.
- All **70 tablebase files** matched the mirror’s published SHA-256 values; **311** random endgames preserved WDL.
- Complete tablebase-only conversions: queen mate in 11 plies; rook in 19; bishop+knight in 53; two bishops in 25, from the test positions. These are DTZ-driven conversions, not minimum-distance-to-mate claims.
- The endgame policy returns to search for missing data, a probe deadline, or an uncertain fifty-move boundary; it also accounts for observed repetitions.

## Match experiments

All scores below are from the candidate’s perspective. These are small local tests, not Elo measurements. Checkpoint/ensemble tests used an exploratory search, so they do not prove the same ordering for every final-engine time control. Short tests used identical permitted per-move ceilings, though iterative deepening makes actual time usage differ.

| Comparison | Wins | Draws | Losses |
|---|---:|---:|---:|
| Checkpoint test: 150M vs 50M, experimental search | 2 | 3 | 1 |
| 75/25 ensemble vs 150M, experimental search | 2 | 2 | 2 |
| Rejected search experiment 1 vs original 150M | 1 | 4 | 3 |
| Rejected search experiment 2 vs original 150M | 0 | 5 | 3 |
| Rejected search experiment 3 vs original 150M | 0 | 3 | 5 |
| Retained search vs original 150M | 1 | 7 | 0 |
| Final API at 120s + 0.5s vs original 150M | 1 | 1 | 0 |

The more expensive exchange/pruning/cache experiments regressed and were rejected. The original move-ordering approximation remains a heuristic; its speed was more useful in these tests. Simply adding more search features did not improve the measured results.

Final API match settings: one worker CPU; 120 seconds per side plus 0.5 seconds per move; paired colours; a fresh worker per game; automatic claimable draws; 600 total plies. These are local opening tests, not the organiser’s private opening suite. The small sample cannot establish a championship ranking or ensure the best move in every position.

## Reproducibility and limitations

Raw correctness/resource summaries and the compact match results are under `reports/`. The final-clock PGN is also included for review and is never imported or read by the agent. Test sources are under `tests/`.

Both model files are byte-identical to the user’s inputs. Their metadata reports training from scratch; the datasets, training programs and original logs were not attached and their claims were not independently recreated here. Preserve those original records for the judges. `PROVENANCE.json` records this distinction and the local-only origin notice of the 150M archive.

No test suite proves zero bugs, universal best moves, organiser approval, or first place. This package is a tested entrant build, not a guarantee of any of those outcomes.
