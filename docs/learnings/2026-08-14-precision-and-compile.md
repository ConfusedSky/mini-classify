## Precision, and what it actually flips (2026-08-14)

The question that started it: torch.compile was disqualified because its
embedding drift (max 9.8e-04) matched the closest observed top-1 margin
(9e-04, 128 renders, 0 flips) — but did anything ever actually flip? Three
harnesses later the answer is measured at every level, and the flag shipped.
Harnesses: `eval/score_precision.py`, `eval/compile_flips.py`, and a
compile-mode sweep added to `eval/siglip_bench.py`. Raw results in
`eval/out/`, gitignored — the figures here are the record.

### The system has three measured perturbation scales

| perturbation | size | where it acts |
|---|---|---|
| fp16 scoring cast | 6.2e-05 max per-view sim | every scored model, every run |
| torch.compile | median 7.3e-04, max 3.1e-03 | embeddings, if compiled |
| JPEG q92 vs in-memory | ~8e-03 mean cosine | saved renders only — off the scoring path by design |

For scale: production top-1 margins are median 0.015, p1 1.75e-04, min
8.6e-07 (softmax pool, 2943 models).

### The fp16 scoring cast: 1 flip in 2,943

Production scores by casting cached fp32 embeddings *down* to fp16 for the
GPU matmul (`classify_stls.py:1078,1104`) while the pose ensemble runs fp32
numpy. Scoring every cached model both ways (plus fp64): **one top-1 flip**,
at a 1.9e-05 margin between 'terrain or scenery piece' and 'building or
ruin'; fp32 agreed with fp64 on every model, so fp32 is converged and fp16 is
the only precision effect in the system. Mean pool: 0 flips; max pool: 3, all
sub-1e-04 near-synonym ties. Not worth "fixing": moving to fp32 would change
one arbitrary answer and perturb every fourth-decimal score in the CSV.

### torch.compile: 1 flip in 341, at a margin 200× below the drift

The direct experiment: the 241 worst-margin models plus 100 controls, each
model's 16 saved views preprocessed once and pushed through eager and
compiled SigLIP as identical tensors. **One flip** —
`Slathaai_Casting_Hand_Animancy_L`, 'demon or monster' → 'beast or animal',
eager margin 4.3e-06. Even among the 44 models whose margins sat *below* the
drift magnitude, only that one flipped: being inside the band means the
perturbation can reach you, but its direction has to line up too.

Two methodology notes that earned their keep:

* **The null-run canary.** The first run reported 0.0 drift after a 1 s
  "compile" — `torch.compile(model)` only intercepts `forward()`, and
  `get_image_features` on the wrapper silently stays eager. Compile the bound
  method (`siglip_bench` had it right all along). A drift of exactly zero
  from a supposedly-compiled tower is the tell.
* **JPEG dilution is real, not theoretical.** At-risk selection used cached
  (lossless-pixel) margins, but the jpg substrate perturbs embeddings ~8e-03
  — the flipped model's cached margin was 1.7e-03 and its jpg margin
  4.3e-06, a 400× move. The comparison itself is immune (identical pixels
  through both towers); the *targeting* has to be read against the margin
  the towers actually contested.

### Compile modes: max-autotune is gated off this card

`COMPILE_MODES=default,max-autotune,reduce-overhead siglip_bench.py compile`,
A/B interleaved per mode (the eager baseline slid 23.3 → 24.5 img/s across
blocks as the card warmed — the ratios are comparable, the raw img/s are not):

| mode | speedup | compile cost | max drift |
|---|---|---|---|
| default | 1.092× | 3.7 s (cache warm) | 9.77e-04 |
| max-autotune | 1.116× | 50.7 s | 7.32e-04 |
| reduce-overhead | 1.104× | 13.0 s | 9.77e-04 (byte-identical to default) |

Inductor gates Triton GEMM autotuning behind `is_big_gpu` (≥68 SMs); the
4060 Laptop's 24 SMs close it, so "max-autotune" here tunes only the
pointwise/reduction kernels around the matmuls — 2 points over default for
14× the compile time. reduce-overhead replays default's kernels through CUDA
graphs, hence identical numerics.

### The decision: `--compile` shipped, as a cache regime

The drift was never the hazard — it flips only coin tosses. The real hazard
was the permanent `.npy` cache mixing two numeric regimes ~7e-04 apart. So
the flag lands with the regime in the cache identity:

* `classify_stls.py --compile` (BooleanOptionalAction) compiles the image
  forward, default mode — the robust choice; the mode sweep's extra 1-2% is
  not worth CUDA-graph fragility across the two production batch shapes.
* `cache_key` appends `|compiled` **only when set**, the same trick `elev`
  uses: every pre-existing eager key stays byte-identical, and the two
  regimes can never share a `.npy`.
* `compile` joins `RUN_PARAMS_KEYS`, so the regime sticks to its cache and
  flows to `test_categories` like any other identity key. Flipping regimes
  on a populated cache is an explicit act (`--no-compile` to leave) and
  costs a re-embed, which is the correct price.
* Text embeddings stay eager — they are recomputed at startup, never cached
  per-file.

### Pass 2 addendum: the pose cache shares the tower (review R1/M3)

The review's second pass found the gap in the paragraph above: `--compile`
re-keys the *embedding* cache and not the *pose* cache, which consumes the
same compiled tower — `score_upright` reaches `get_image_features` through
`embed_images`, so under `--compile` the ensemble's tile embeddings carry the
~1e-03 drift into `combine_up`, which decides both the pose argmax and the
margin. The exposure is not a wrong category but the escalation gate: a
margin crossing `MARGIN_THRESHOLD` changes whether a *paid* arbiter call
happens, and `compile_flips.py` measured category flips, not pose flips.

**Now measured — twice, the second time bounded**
(`eval/compile_pose_flips.py`). The first round targeted by *cached* margin:
180 models, 1 up flip (a ramp plank at eager margin 2.2e-03, z to −z), 0
gate flips — but review T2 showed the gate result rested on n=4, because
cached margins are not live margins (54 cached-in-band collapsed to 4
live-in-band). The re-run selects on the margins the towers actually
contest: an eager census over all 2799 pose-cached models, then
compile-vs-eager on the 197 whose live margins sat within 0.022 of either
exposure point. Result: **0 gate flips in the 49 still in the gate band at
compare time** — a rule-of-three bound of ≈6% per in-band model. The exposed
population is a range, not a point (U3): 107 models sit within the harness
band of the gate, 133 within the median run-to-run noise, 385 within its
p90 — the margin moves more than the band, so "in-band" is not a fixed set.
And **4 up flips**, every one at an eager margin ≤ 4.1e-03: ties. Margin deltas median 1.5e-03, max 1.5e-02,
consistent with round one, and the ~20× pose-vs-category amplification
(`combine_up_scores`' min-max) stands. The acceptance below now rests on a
bounded measurement.

**The ratio that closes R1 (U1).** The compare rows hold two eager margins
per model — phase-1 census and phase-2 compare — so their difference is the
pipeline's own run-to-run noise, measured on the same rows as the compile
delta:

| \|Δ margin\| | median | p90 | max |
|---|---|---|---|
| run-to-run (eager vs eager) | 2.66e-02 | 8.93e-02 | 2.69e-01 |
| torch.compile (identical tensors) | 1.48e-03 | 4.70e-03 | 1.52e-02 |

**Compile perturbs the pose margin 18× less than re-running the pipeline
unchanged**, and for 123 of 197 models the run-to-run noise exceeds the
largest compile delta observed anywhere. That makes the acceptance
categorical, not statistical: enabling `--compile` moves a pose less than
re-resolving that same pose twice eagerly — which the pipeline already does
routinely on every upgrade path and already treats as sound. No flip count
can weaken this while the ratio holds.

**The noise's mechanism is the renderer, isolated (U2,
`eval/render_determinism.py`):** re-rendering the same loaded mesh's
candidate grid in one process changes ~43% of pixels by 2–28/255 on every
model tried, while the tower on byte-identical tensors is bit-deterministic
(max delta 0.0, twice). Filament is the sole source; fp16 kernel variance is
excluded. The spread reaches the up *pick* too: one tight-margin model
(`32mm_Pipe5`) chose a different up axis between two identical eager passes
— run-to-run pick instability is real at tie margins, not just margin
jitter. Geometry was seeded precisely to prevent this class of
irreproducibility; the SigLIP arm reintroduces it through the renderer, and
`combine_up`'s min-max amplifies it. A fix exists and is verified (review
V1): `set_post_processing(False)` — Filament's temporal dithering was the
noise — gives byte-identical renders and 0.00 margin spread on 7/7 models
after one throwaway frame. But the toggle also removes tone mapping: the
stable render is a visibly different, much darker image (max 187/255 over
100% of pixels vs production), and no `ColorGrading` combination in
Open3D's binding is byte-stable, so there is no keep-the-look knob.
Adoption is therefore gated on both cache version bumps *and* a real
accuracy re-read on the new-look pixels (margins move median 1.3e-01); the
open entry in `OPEN_QUESTIONS.md` carries the decision.

The census bought two more observations bigger than the bound it ran for:

* **Cached margins have drifted median 0.119 (p90 0.354, max 1.13) from
  live ones** under the deliberately unbumped `POSE_CACHE_VERSION` — the
  `UP_TILE_AZIMUTHS` 4→2 change plus state noise, now measured
  collection-wide. The non-bump's argument was that *decisions* are stable,
  and it said nothing about margins; T2's 54→4 collapse was the first
  symptom and this is the full-scale number. Any analysis that keys on
  cached margins must re-derive them live first.
* **Live-vs-live is noisy too**: between the census pass and the compare
  pass — same code, same config — the ±0.022 gate band kept only 49 of 107
  models, the up band 30 of 90. A same-config re-render moves margins
  across a 2e-02 boundary about half the time: the `parser_gate` A/A lesson
  at collection scale, and the honest reason "in band at compare time" is
  the denominator quoted above.

**Accepted, as a decision rather than an omission**: poses are resolved only
on cold and upgrade files, and a mixed pose cache sits inside the state noise
`parser_gate`'s A/A control showed is permanent anyway. It is one of three
instances of the same bug — an input that moves the pose but not its key —
beside `--render-size` (recorded in the 7-hour-run write-up) and the absence
of any version on the embedding's *derivation*. The third is now closed:
`EMBED_CACHE_VERSION` versions the `load_mesh → up_axis_scores` chain the way
`POSE_CACHE_VERSION` versions poses, appended to the key only when bumped so
its introduction moved nothing (the numpy-parser swap was the near-miss that
motivated it — it passed only because triangle counts and bounding boxes came
out exact).
