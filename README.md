# Sage
![alt text](https://media.licdn.com/dms/image/v2/D4E22AQGgXqpJbxjGug/feedshare-shrink_800/B4EaCfADRmG4Ac-/0/1789373986192?e=1790812800&v=beta&t=RjHRw5Rjjxho3mBh70A2JXYfVSmu6yEA1cO-HltwNKg)

Sage is a Python/Numba chess engine built for the AI Chessathon. This repository contains the final competition engine, along with most of the models and larger experiments we made during the hackathon.

The version of Sage in the root of the repository is the final submitted build. Older engines and experiments are in the `archive/` folder.

This project did not start on GitHub. Most of it was developed locally, usually as folders and frozen `.zip` submissions, so the Git history here does not match the real development timeline. I have tried to reconstruct that timeline below from the builds, hashes, benchmarks and match results we kept.

Roughly, it went like this:

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
Stage 3
      ↓
endgame work
      ↓
anti-repetition work
      ↓
qualification build
      ↓
R2
      ↓
Final Sage build
```

---

# The first submission

The oldest engine I still have is just called `submission.zip`. It was already using an NNUE-style evaluator with incremental accumulators, and the search had most of the things you would expect from a proper engine: iterative deepening, PVS / alpha-beta, a transposition table, null move, LMR, killers, history, quiescence search and Zobrist hashing.

The evaluator was a large king-relative HalfKP network:

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

A lot of the code was written around the fact that Python itself was too slow for the hot path. The board/search/eval code therefore leaned heavily on Numba. That general setup stayed with the project even when the network architecture changed several times.

---

# Fable

Fable was the first big change in direction. It used a much smaller evaluator based on normal piece-square inputs rather than the huge king-relative feature space:

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

The NN output was blended with a normal tapered material / PSQT evaluation, so Fable was more of a hybrid engine. We made several Fable versions after that. Some tried better draw handling, SEE, capture history, bigger TTs, Syzygy, opening books and other search changes.

I do not think the individual Fable versions are that important to the final story. What mattered was that Fable was cheap to evaluate and could search quickly. It gave us a useful reference point later, especially when Parable became much more expensive per node.

---

# Parable

Parable went back in the other direction. We built a much larger NNUE with mirrored king-relative HalfKP features and two extra hidden layers:

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

It was around 5.26 million parameters. We trained and tuned several Parable versions and changed the search around it, but the network shape stayed broadly the same.

Parable got better over time, but it was expensive. The engine was spending a lot more time evaluating each node, and the extra network capacity was not making up for the loss in search speed. That became hard to ignore when we compared it with smaller engines such as Fable.

Eventually we stopped trying to keep improving that architecture and started again with Sage.

---

# Sage — the architecture reset

Sage was the rebuild. Instead of using a separate king bucket for every king square, it used eight coarse king regions. That cut the sparse feature count down to 6,144. We also removed Parable's two 32-wide hidden layers.

The Sage network was:

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

Or more compactly:

```text
6144 → 256×2 → SCReLU → 512 → bucketed linear → 1
```

Sage also used a fixed material / PST evaluation underneath the network. The NN was trained as a residual on top of that baseline instead of being responsible for the whole evaluation by itself.

The network ended up at roughly 1.58 million parameters. We had 50M and 150M training versions, but they were the same basic architecture. The 150M weights became the ones we kept using for almost all of the later work.

---

# The Sage breakthrough

Once the 150M network was in place, we mostly stopped changing the NN and started spending our time on search.

There were a lot of branches. Stage 2C was a refactor that looked fine logically but was much slower in practice, so it was dropped. Stage 4 tried to make repetition and fifty-move handling more correct, but the history-aware TT checks cost too much search speed. Stage 5 and Stage 6 kept working on the TT/context problem and memory layout. Stage 7D added continuation-history style ordering.

The version that kept standing out was Stage 3. It used the same 150M network, searched about 7% faster than the earlier Sage search in our tests, and did not show a strength loss in the match testing we ran. After trying several newer branches, we kept ending up back at Stage 3 because it was fast and reliable.

That became the search base for the later engine.

---

# Endgame conversion

The next problem was not really search speed. It was converting positions that Sage already knew were winning.

In some low-material games the eval could be huge, but the engine would still shuffle around, make very slow progress, or let the fifty-move counter get uncomfortably high. Stage 8E added a small mop-up term for obvious winning endings, especially positions where one side was basically down to a bare king.

The bonus pushed the losing king toward the edge and encouraged the winning king to get closer. It was intentionally narrow rather than a replacement endgame evaluator.

Stage 8F tried something different: giving progress moves such as pawn pushes more preference at the root when the fifty-move clock was getting high. That could change the search effort, but it did not reliably make Sage choose the move we actually wanted.

So instead of keeping the whole Stage 7/8 search line, we copied the useful Stage 8E mop-up idea back onto Stage 3. That became Stage 3E.

---

# Anti-repetition

Even with better endgame evaluation, Sage could still repeat in positions where it was clearly ahead.

We did not want to make repetition illegal or always bad. If the engine is worse, repeating can obviously be the right move. The eventual ER2 / R2 solution only does anything after the normal Stage 3E search is finished.

If the chosen root move goes back to a recent reversible position, Sage is already clearly ahead, the search completed deeply enough, and there is another legal move available, the engine runs a small extra search without the repeating root moves. It only switches moves if that second search still says the fresh move is clearly winning.

So normal search still decides the move first. The anti-repeat code is more of a last check for obvious winning-position loops.

---

# Qualification build

By the time we were preparing the qualification build, the engine had more or less settled into:

```text
Stage 3 search
+
Sage 150M NNUE
+
endgame mop-up
+
anti-repetition protection
```

A lot of newer experiments existed by then, but we did not just use the newest branch. Stage 3 was still the search we trusted most, and the useful later changes were added back onto it.

That was the version of Sage we took into the qualification stage.

---

# Finals: R2

After qualifying, we kept working on that same line for the final. R2 / Stage3ER2 was:

```text
Stage 3 search
+
Sage150 NNUE
+
Stage 8E-derived mop-up
+
ER2 root-only anti-repeat rescue
```

The NN architecture itself had not changed:

```text
6144 → 256×2 → SCReLU → 512 → 8 bucketed output heads → 1
```

In our R2 vs Stage 3E match:

```text
20 pairs / 40 games

R2:
8 wins
28 draws
4 losses

Score: 55%
```

The anti-repeat code only triggered occasionally and used a tiny fraction of the total search time, which was what we wanted. It was there to catch a specific failure mode rather than reshape the whole search.

---

# Final Sage build

The final submitted Sage was still based on Stage3ER2. We did not change the NN again.

It still used:

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

The last changes were mostly engineering work and a few edge cases. We worked on faster Numba initialisation, increased the transposition table size, changed how repetition inside the current search tree was handled, and added some extra endgame conversion code.

There was also work on endings such as two bishops vs king and bishop + knight vs king. The two-bishop case worked better; the bishop + knight code was less reliable at normal game depth, so I would not describe that part as solved.

The final engine was basically the Sage150 evaluator we had been using for a long time, attached to the Stage 3 search, with the endgame and repetition fixes that had proved useful.

---

# What the project taught us

The biggest thing we changed our minds about was network size. Early on, making the NN stronger mostly meant making it bigger. Parable pushed that quite far. In the end, Sage worked better for us because it was much cheaper and gave the search more time.

We also had plenty of changes that looked good in isolation and then did not survive testing. Some lost too much NPS. Some used more memory. Some were more correct but made the engine weaker under the actual clock. Some just did not change the match results enough to be worth keeping.

By the end we were fairly conservative about promoting changes. We normally wanted a correctness check, a fixed-depth comparison, performance numbers and games before replacing the current build.

That is why the final Sage still contains a fairly old search core. Stage 3 kept winning its place back. The later additions were mostly small fixes for problems we had actually seen in games rather than a full rewrite every time.

---
