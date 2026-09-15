# Sage
![alt text](https://media.licdn.com/dms/image/v2/D4E22AQGgXqpJbxjGug/feedshare-shrink_800/B4EaCfADRmG4Ac-/0/1789373986192?e=1790812800&v=beta&t=RjHRw5Rjjxho3mBh70A2JXYfVSmu6yEA1cO-HltwNKg)

Sage is a Python/Numba chess engine built for the AI Chessathon. This repository contains the final competition engine as well as most of the models and major experimental builds we made during the hackathon.

The version of Sage at the root of this repository is the final submitted competition build. Earlier engines, intermediate versions, and experiments are preserved under the `archive/` folder.

The project did **not** begin on GitHub. Most of the development happened locally using frozen `.zip` submissions, experiment folders, hashes, benchmarks, match logs, and test packages. This repository was created afterwards, so the Git commit history is not the original development chronology. This README is intended to preserve that chronology.

The overall path was roughly:

```text
First submission
      ↓
    Fable
      ↓
   Parable
      ↓
Sage architecture reset
      ↓
search / correctness iterations
      ↓
Stage 3 breakthrough
      ↓
endgame conversion work
      ↓
anti-repetition work
      ↓
qualification build
      ↓
R2
      ↓
Final Sage build
```

The most important change in the whole project was not one search tweak or one extra training run. It was the decision to stop making the evaluator larger and instead build a much cheaper architecture that let the search do more work.

---

# The first submission

The oldest preserved engine in the archive is simply `submission.zip`. It already used a real NNUE-style evaluator, incremental accumulators, iterative deepening, PVS / alpha-beta, a transposition table, null-move pruning, late move reductions, killers, history, quiescence search, Zobrist hashing, and Numba-compiled hot paths. Its evaluator was a large king-relative HalfKP network with roughly 40,960 sparse features feeding 256-wide accumulators for both perspectives and then a linear output:

```text
40,960 HalfKP features
          ↓
256 accumulator per perspective
          ↓
clipped activation
          ↓
512 combined values
          ↓
1
```

That first engine established the basic approach we kept for the rest of the project: use Python for the overall engine, but keep the expensive chess logic and NNUE inference in Numba-friendly code so that the bot could still search seriously under competition time controls.

---

# Fable

Fable was the first major redesign. Instead of using the very large king-relative HalfKP feature space, it used a much smaller neural evaluator based on ordinary piece-square inputs. Its network was approximately:

```text
768 piece-square inputs
       ↓
256 accumulator per perspective
       ↓
SCReLU
       ↓
512 combined activations
       ↓
1
```

The neural score was blended with a classical tapered material / PSQT evaluation, so Fable was deliberately a hybrid rather than a purely neural engine. The later Fable variants experimented with things like stronger draw handling, larger transposition tables, Static Exchange Evaluation, capture-history ordering, Syzygy support, opening books, and other search refinements. Those versions were useful, but they were not the central direction of the project. The main thing we learned from Fable was that **a smaller evaluator could search much faster and that cheap search-friendly evaluation mattered at least as much as raw network size**. Fable became an important comparison point when we later evaluated Parable.

---

# Parable

Parable was the opposite direction: we tried a much larger learned evaluator again, this time with mirrored king-relative HalfKP features and two extra hidden layers. The main architecture was:

```text
20,480 mirrored HalfKP inputs
          ↓
         256
          ↓
          32
          ↓
          32
          ↓
           1
```

This was roughly a 5.26 million parameter network. We iterated on Parable through several versions, including more training and multiple search refinements, but the basic architecture stayed similar. Parable did improve, but the key lesson was that the larger evaluator was expensive enough to noticeably reduce search throughput. In practice, we were paying a lot of CPU time per node without getting enough playing strength back from the extra network capacity. Comparisons against the smaller Fable line made this increasingly obvious. The most important outcome of Parable was therefore not a particular version number: it was the decision to **stop trying to rescue the large evaluator and completely redesign the architecture**.

---

# Sage — the architecture reset

Sage was the clean reset. Instead of full king-square HalfKP, it used only eight coarse king regions, which reduced the sparse input space to 6,144 features. It also removed Parable's two 32-wide hidden layers. The final Sage family used:

```text
6,144 king-bucketed inputs
          ↓
256 accumulator per perspective
          ↓
SCReLU
          ↓
512 combined activations
          ↓
1 of 8 piece-count output heads
          ↓
1
```

In compact form:

```text
6144 → 256×2 → SCReLU → 512 → bucketed linear → 1
```

Sage was also trained as a **residual evaluator**. Instead of asking the network to learn material and every basic positional fact from scratch, we kept a fixed material / PST baseline and trained the NNUE to learn the correction on top of it. The final Sage network was roughly 1.58 million parameters, much smaller than Parable. The 50M and 150M versions were essentially the same architecture; the important difference was training amount and the maturity of the search around them. The 150M checkpoint became the long-lived network used by almost every later Sage experiment, including the final bot.

---

# The Sage breakthrough

Once the 150M network was good enough, most of the development shifted away from neural architecture and into search. We experimented with faster and slower refactors, stricter draw correctness, history-aware transposition-table reuse, multiple TT contexts, different memory layouts, continuation-history ordering, move-ordering changes, and endgame policies. Some ideas were theoretically cleaner but made the engine slower. Stage 2C was the clearest example: it preserved the intended logic but roughly halved search throughput, so it was abandoned. Stage 4 improved repetition and fifty-move correctness but lost too much NPS. Stage 5 and Stage 6 explored history-aware TT layouts and cache locality. Stage 7D added continuation-history style move ordering. These experiments were useful, but the most important result was that **Stage 3 emerged as the strongest practical search core**. It was faster than the earlier Sage search, retained the exact 150M network, and gave a repeatable throughput gain of around 7% without an observed strength regression. From that point onward, Stage 3 became the stable base we kept returning to whenever a more complicated branch failed to justify itself.

---

# Endgame conversion

After search speed improved, a different weakness became obvious: Sage could evaluate a position as clearly winning without always converting it cleanly. It could shuffle, delay irreversible progress, or allow the fifty-move counter to become dangerous. Stage 8E attacked this with a very narrow handcrafted mop-up bonus for clearly winning low-material positions, especially bare-king endings. The bonus rewarded pushing the defending king toward the edge and bringing the winning king closer. It was deliberately small and heavily gated rather than being a general-purpose endgame evaluator.

Stage 8F then experimented with encouraging progress moves such as pawn pushes at the root when the fifty-move clock was rising. That showed an important distinction: changing move ordering can change where search effort goes, but it does not necessarily make the engine choose the right irreversible move. Rather than keep stacking new search ideas on top of the Stage 7 line, we went back to the strongest stable core and created **Stage 3E: exact Stage 3 search plus the useful Stage 8E mop-up evaluation**.

---

# Anti-repetition

The next weakness was repetition in positions the engine already believed were winning. We did not want to simply ban repetition, because repetition can be the correct result when the engine is worse or drawing. Instead, the later ER2 / R2 approach was conservative: run the normal Stage 3E search first, and only afterwards intervene if the selected root move revisits recent reversible history while Sage is already clearly ahead.

The rescue logic was deliberately narrow. It only activates when the normal search has reached sufficient depth, the score is clearly positive, the selected move repeats, and a fresh legal alternative exists. It then performs a small capped search excluding revisiting root moves. The alternative is accepted only if that rescue search still shows a clearly winning position. This gave us a way to avoid obvious winning-position shuffles without changing normal recursive search behaviour.

---

# Qualification build

By the qualification stage, the project had converged on a much simpler answer than the long list of experiments might suggest:

```text
Stage 3 search
+
Sage 150M residual NNUE
+
endgame mop-up
+
anti-repetition protection
```

That combination represented the strongest ideas we had actually validated. Instead of using every later experimental search feature, we kept the Stage 3 core and transplanted only the changes that addressed real observed weaknesses: conversion and avoidable repetition.

That was the major qualification-era breakthrough: **the final direction was not the newest search branch, but the strongest proven search core plus a few narrow fixes.**

---

# Finals: R2

After qualification, the engine was frozen into the R2 / Stage3ER2 line for the finals phase. The preserved R2 package combines:

```text
Stage 3 search
+
Sage150 NNUE
+
Stage 8E-derived mop-up
+
ER2 root-only anti-repeat rescue
```

The neural architecture was still exactly the same Sage architecture:

```text
6144 → 256×2 → SCReLU → 512 → 8 bucketed output heads → 1
```

At this point, the project was no longer trying to make the network larger. The improvements were about stability, conversion, repetition handling, runtime behaviour, and preserving the search strength we already had.

In the controlled R2 vs Stage 3E development arena:

```text
20 pairs / 40 games

R2:
8 wins
28 draws
4 losses

Score: 55%
```

The anti-repeat rescue itself was rare and cheap relative to the total search workload, which was exactly what we wanted from it.

---

# Final Sage build

The final Sage build is the last evolution of the same Stage3ER2 architecture rather than another neural redesign.

It still uses the exact Sage150 network and the same overall evaluator:

```text
6,144 sparse king-bucketed features
              ↓
256 accumulator per perspective
              ↓
SCReLU
              ↓
512 combined values
              ↓
piece-count bucketed linear head
              ↓
residual score
              +
fixed material / PST baseline
              +
narrow endgame mop-up terms
```

The final work focused on engineering and edge cases rather than changing the network. The preserved final Sage build includes faster Numba initialisation work, a larger transposition table, improved search-tree repetition handling, and additional endgame conversion logic. Some extra mop-up work was added for endings such as two bishops against king and bishop + knight against king. Not every experimental ending routine was equally reliable, but the main R2 / Stage 3 / Sage150 foundation stayed intact.

The final bot can therefore be thought of as:

> **A compact residual NNUE attached to a heavily optimised selective alpha-beta search, with narrow endgame and anti-repetition safeguards added only where testing showed a real conversion problem.**

---

# What the project taught us

The strongest lesson was that a chess engine is not just its evaluator.

Parable had more neural capacity than Sage, but Sage was cheap enough that the search could do substantially more work. Once Sage150 was strong and stable, most of the gains came from search speed, move ordering, draw handling, conversion and reliability rather than from making the network larger.

A second major lesson was that theoretically cleaner changes could still make the engine worse. History-aware TT correctness, larger tables, more complicated move ordering, extra verification searches, and larger integration packages all had costs. We ended up rejecting many ideas that looked attractive on paper because they either lost NPS, increased memory pressure, or failed to improve paired games.

The development rule gradually became:

> **A feature is not promoted because it sounds stronger. It is promoted because it survives correctness testing, fixed-depth comparison, performance measurement, and actual games.**

That is why the final engine is a mixture of old and new ideas. Stage 3 survived because it was fast and strong. Stage 8E survived because it improved conversion. ER2 survived because it addressed repetition without disturbing normal search. Everything else had to justify its cost.

---
