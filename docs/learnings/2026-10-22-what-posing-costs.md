# What posing costs, re-measured: as much as the views, and `pose-embed` under-reports by 10×

*2026-10-22. `classify_stls.py --instrument`, `--pose-vlm off`, 512 px,
`google/siglip2-so400m-patch16-512`, a fresh scratch cache per run (none of
these numbers touched `embed-cache512`). Samples of 40, seed 1022. One run per
arm. Nothing else was running on either GPU.*

The last per-stage numbers were from 2026-08-13 (`2026-08-13-instrumented.md`):
602 models at 384 px, the old parser, 24 pose tiles, with posing at 1926
ms/model against 1297 ms for the views. Since then the tiles have halved
(`UP_TILE_AZIMUTHS = 2`), the tower and render size have changed, and the
−20° ring has been dropped. The question was what posing costs now. A second
question came up along the way: why the 2026-09-01 one-ring run over 3,380
models took only about 47 minutes.

## The runs

| # | sample | views | profile | compile | wall |
|---|---|---|---|---|---|
| 1 | `test-models/miniatures/original` | 16 | balanced | off | 78.1 s |
| 2 | STLLibrary | 16 | balanced | off | 101.2 s |
| 3 | STLLibrary | 16 | performance | off | 96.5 s |
| 4 | STLLibrary | **8** | performance | off | **77.7 s** |
| 5 | STLLibrary | **8** | performance | on | 116.7 s |

The STLLibrary sample is 40 files drawn from the `embed-cache512` walk list of
2026-09-10 (3,070 files, part-filtered). That makes it a sample of what an index
actually processes, not of the raw volume. Run 1 used the test collection only
because the volume was not mounted. Its meshes are much lighter (16 ms mesh-load
against 132 ms), so it is a lower bound and nothing more. Runs 4 and 5 are the
production shape (`--elevations 20`, 8 views).

## Per-stage, ms/model, as the instrument reports them

| stage | where | 1 test 16v | 2 lib 16v | 3 lib 16v perf | 4 lib 8v | 5 lib 8v compile |
|---|---|---|---|---|---|---|
| mesh-load | child | 16 | 132 | 117 | 135 | 102 |
| pose-render (12 tiles) | child | 119 | 456 | 427 | 456 | 445 |
| pose-geometry | child | 4 | 17 | 16 | 17 | 17 |
| view-render | child | 241 | 893 | 859 | 635 | 605 |
| pose-embed — **launch only, see below** | parent | 122 | 124 | 126 | 143 | 572 (~100 steady) |
| `cache-save` (view forward) | parent | 780 | 783 | 769 | 372 | 349 |
| `embed` (preprocessing) | parent | 102 | 126 | 125 | 63 | 642 (~72 steady) |
| `results-wait` total | parent | 7.9 s | 30.5 s | 27.5 s | 26.3 s | 25.3 s |
| unattributed main-thread time | parent | 24.0 s | 24.4 s | 23.7 s | 23.3 s | 22.7 s |

## `pose-embed` times the launch, not the forward

The first pass over these numbers had `pose-embed` at 122–143 ms/model, 10×
below 2026-08-13's 1176 ms. Halving the tiles explains only 2×, and the bigger
tower should cost more, not less. The tiles were not 10× cheaper. The stage
stopped containing the work.

It is the 2026-08-13 lesson, "`embed` and `cache-save` are one number split by
a CUDA sync", in a new place. Torch launches asynchronously, so
`Embedder.embed_tiles` returns as soon as preprocessing is done and the forward
is queued. In `src/driver.py`'s `drain`, only that call is inside
`stage("pose-embed")`. The GPU wait lands at the next line,
`Poser.on_tile_embeds`'s `m.embeds.float().cpu().numpy()`, which is outside
every stage. The view path has the same split, but its pull is inside
`cache-save`, so the views were always counted.

Measured directly through `eval/rig.py`'s production `Embedder`: warm, 512 px
inputs, median of 10, with a `torch.cuda.synchronize()` before each call.

| | inside `embed_*` | at the `.cpu()` pull | total |
|---|---|---|---|
| 12 pose tiles | 58 ms | 560 ms | **618 ms** |
| 8 views | 38 ms | 376 ms | 414 ms |

The missing ~560 ms is the instrument's unattributed main-thread time: ~23–24 s
per 40-model run (≈ 580–600 ms/model), the same on both samples, as a forward
pass should be. The real tile cost, ~52 ms per tile, is in line with
2026-08-13's ~49 ms on the smaller tower. **The 10× improvement does not
exist.** Pose-embed is ~1176 → ~620 ms/model, which is the tile halving and
nothing more.

Every `--instrument` table from the actor refactor until this fix
under-reports `pose-embed` in the same way. For those tables, add the
unattributed remainder to the stage.

### Fixed, and re-measured

`drain` now calls `on_tile_embeds` inside `stage("pose-embed")`.
`tests/test_driver.py::test_pose_embed_times_the_poser_pull` checks that, and
it failed against the old code before passing against the new. The same 40
STLLibrary models were rerun at 8 views, performance profile, fresh scratch
caches:

| stage | where | fixed, compile off | fixed, compile on |
|---|---|---|---|
| wall | | 77.2 s | 82.3 s |
| mesh-load | child | 161 | 129 |
| pose-render | child | 419 | 442 |
| pose-geometry | child | 15 | 17 |
| view-render | child | 566 | 591 |
| **pose-embed** | parent | **676** | 742 |
| `cache-save` | parent | 371 | 358 |
| `embed` | parent | 65 | 144 |
| `results-wait` total | parent | 27.2 s | 25.5 s |
| unattributed main-thread time | parent | **1.4 s** | 1.1 s |

The unattributed time fell from ~23 s to ~1.4 s, and `pose-embed` now reads
676 ms, matching the direct measurement. Posing is 419 + 676 + 15 ≈ **1.11
s/model** against 566 + 371 + 65 ≈ **1.0 s** for the views, so the corrected
estimates below hold.

The compile arm's wall clock was 82.3 s this time against 116.7 s in run 5.
Most of the ~42 s of warmup did not recur, presumably because torch's
on-disk compile cache was warm from run 5 (not verified). It was still slower
than compile off, with no stage meaningfully faster.

## Posing costs as much as the views

Corrected, at the production shape (run 4): posing is 456 (render) + ~620
(embed) + 17 (geometry) ≈ **1.09 s/model**, against 635 + ~435 ≈ **1.07 s** for
the eight views. Posing is half of the per-model work for a cold file.

The two processes are close to balanced. The child works ~1.24 s per model
(mesh-load, pose-render, view-render). The parent works ~1.16 s (tile forward
~620 ms plus view forward ~435 ms plus preprocessing). Neither is clearly the
bottleneck. The ~26 s of `results-wait` are the stalls of a two-stage pipeline
whose stages trade places from model to model. Skipping posing would take load
off both sides, so a warm-pose run should be close to twice as fast as a cold
one. That matches the reconciliation below.

The test collection understates rendering by 3.7–3.8× (render stages scale with
mesh size; the forwards do not). **Measure on the volume, not the test set.**

Halving the views does not halve view-render (893 → 635 ms). There is a fixed
per-model cost under the views, including the ~275 ms geometry upload.

## The 47-minute run reconciles

The 2026-09-01 run (3,380 models, `--elevations 20`, `--compile`) found every
pose already cached from the ev2 rebuild. The child did mesh-load and eight
views only, ~0.77 s per model, and the parent did one 8-view forward, ~0.45 s.
That is ~43 minutes for 3,380, plus startup and drain, against the ~47 minutes
the cache's file mtimes show. For a **cold** index of the 3,070-file walk at 8
views with the arbiter off, run 4 projects ~1.83 s per model, about **1.6 h**.
The cold/warm ratio, ~2.2×, is posing's cost.

## `--compile` buys almost nothing

Run 5 pays roughly **42 s of one-off compile warmup**: ~19 s on the first
pose-embed and ~23 s on the first view embed, because the 12-tile and 8-view
batches are different shapes. That was read off the instrument's ~100 ms
sampling timeline, so the steady-state figures are approximate. After warmup the
forwards barely move: `cache-save` 372 → 349 ms, unattributed time 23.3 → 22.7
s. `results-wait` is unchanged (26.3 s vs 25.3 s). Without the warmup, run 5's
wall clock is about 75 s against run 4's 77.7 s. With the instrument fixed
and the compile cache warm, the compile arm was still 5 s slower (82.3 s vs
77.2 s). On this card, compile is a few percent on the forward at best and
break-even or worse on a run.

## Performance mode: within noise

`powerprofilesctl` `performance` against `balanced` (runs 3 and 2): −4.6% wall,
−4% view-render, −6% pose-render, embed stages flat. The iGPU sits on the
platform profile and the 4060 does not, which fits the pattern. But it is one
run each, and run 3 read the same files from a warm page cache, which probably
accounts for most of the mesh-load drop. Don't quote it as a gain.

Rerunning the same models moved one score in the third decimal
(`KC3900Zeus_Upside_Part_03`, 0.1235 vs 0.1247, same top 3). That is the known
Filament draw-history jitter, not a profile effect.

## Arbiter load

With the arbiter off, `--pose-vlm off` reports what it would have sent: 24 of
40 STLLibrary models (60%) sat below `MARGIN_THRESHOLD` (the ev2 rebuild
escalated 45%). On the test collection the figure was 13 of 40. At n=40 this
is noisy. The arbiter runs on a background pool, so its cost to wall clock was
not measured here.
