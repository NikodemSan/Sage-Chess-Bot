# Sage Fusion — AI Chessathon submission

Upload the ZIP itself. `agent.py` is already at its root. No training, downloads,
configuration, installation, or repacking is needed on the competition server.

This revision uses your **150M-sample Sage checkpoint** by default. Your original
50M checkpoint is also included unchanged, with an optional two-network ensemble.
The numbers describe training samples, not parameter counts. Both source builds
had the same board, search and evaluation code.

## What is added

- A **512 MiB transposition table** (32 million entries), versus 32 MiB in the
  supplied engine, with room for Python, Numba, weights and mapped tablebases.
- **All 3- and 4-piece Syzygy endgames**, both WDL and DTZ: 70 verified data files,
  4,346,080 bytes. Covered endings use result-preserving moves and DTZ progress.
  Near uncertain fifty-move boundaries or a probe deadline, normal search resumes.
- Startup warmup includes the otherwise hidden first-use compilation cost of
  recording our move. Compile caches are disabled, so a read-only submission works.
- Adaptive search budgets, a clock reserve, and deadline checks every 64 nodes.
- Public API legal-move validation, low-clock fallback, exception fallback, and
  persistent history for both our positions and the positions after our moves.
- Legal en-passant hashing, including pinned-pawn cases; a null move resets the
  speculative reversible-history horizon. Cached score cutoffs/static bounds
  are disabled near the fifty-move boundary (80+ halfmoves).
- Weight-shape and training-contract checks, model and tablebase provenance,
  reproducible differential tests and a local comparison harness.

The default retains the supplied engine's selective search: PVS/alpha-beta,
iterative deepening, aspiration windows, quiescence, null move, futility,
late-move reductions, killers and history. More expensive exchange analysis
and other search experiments lost short local matches and were removed.
The ensemble did not establish an advantage, so it is not the default.

## Resource choices

Only one CPU thread is used. There is no pondering, network access, GPU use,
external engine, compiler invocation, or runtime installation. The neural
weights are the two files you supplied, not a published chess network.

The 2 GB allowance is a ceiling, not a target to fill. A 512 MiB table already
holds substantially more entries than the original. Allocating unused memory
would not guarantee stronger chess and would reduce startup/runtime headroom.
See `BUILD_REPORT.md` for measured startup, peak memory and the local matches.

## Optional local checks on your Mac

Use Python 3.12 in an environment with these packages:

```sh
python3 -m pip install numpy==2.5.2 numba==0.67.0 chess==1.11.2
python3 tests/verify.py
python3 tests/endgames_test.py
python3 tests/integration.py
```

Startup includes compilation, so allow roughly tens of seconds before output.
The tests print progress. The test scripts are not imported by the entry point.

Compare against an extracted original build (its directory must contain
`zengine.py` and its original `weights.npz`):

```sh
python3 tests/match.py --opponent /path/to/original --games 8
python3 tests/match.py --opponent /path/to/original --games 2 --seconds 120 --increment 0.5 --api --fresh-workers
```

The latter uses each build's actual `get_move` API and the event clock; local
hardware is still different. `--api` uses `agent.py`, or `reference_agent.py`
if that is how the original was packaged. JSON results and a PGN are written
locally. The harness does not estimate Elo from a tiny sample.

## Optional model configuration

`config.json` selects `150m`, `50m`, or `ensemble`; its default is `150m`.
The ensemble concatenates the independent hidden units and weights their
outputs 75% toward 150M / 25% toward 50M. It does not average unrelated hidden
weight coordinates, retrain either model, or claim to be a new trained checkpoint.
Changing configuration makes a different build and needs fresh match testing.

For local experiments, `SAGE_MODEL` and `SAGE_TT_BITS` override configuration.
Do not change these to increase memory blindly.

## Explaining this entry to judges

`PROVENANCE.json` preserves both checkpoints' supplied training metadata and
hashes. Keep your original training scripts, dataset records and training logs:
they were not in these attachments, and this package cannot recreate proof of
training from weights alone. No new model training was performed here.
`models/provenance150m.json` is the original supplied provenance file.

Tablebase hashes were checked against the Lichess mirror's published SHA-256
list. These are the expressly permitted endgame-tablebase data, accessed through
the preinstalled `chess.syzygy` library. No third-party engine code was included.

This build is intended for the agent competition, not the Daily Five.
Testing reduces risk; it does not certify zero bugs, perfect moves, an Elo,
first place, or approval by the organisers.

Rules and interface checked on 7 September 2026:
https://aichessathon.com/docs and https://aichessathon.com/terms .
