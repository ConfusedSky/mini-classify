# Session Learnings — STL miniature classification (2026-08-10)

Journey: set up [Find3D](https://github.com/ziqi-ma/Find3D) → realized it was the
wrong tool for whole-model classification → built a local render + SigLIP
zero-shot classifier (`classify_stls.py`) for a ~1000-model printable
miniature collection.

## Picking the right tool

- **Find3D segments parts *within* an object from text queries** ("head of a
  mickey mouse"). It cannot name an object or its parts — it only scores
  points against queries you supply. Wrong tool for "what is this model?"
- **CLIP/SigLIP-style models match, they don't generate.** A category list is
  mandatory. No list → nothing to score. Alternatives that avoid hand-writing
  one: cluster the image embeddings and name clusters afterward, or use a
  generative VLM (Claude API) that can describe objects freely.
- Options ranked for classifying STL collections: (1) render + VLM API =
  highest accuracy, per-call cost; (2) render + SigLIP local = free, chosen
  here; (3) native 3D zero-shot (Uni3D/OpenShape) = weaker on non-Objaverse
  shapes; (4) supervised training = only with labels + fixed taxonomy.

## What actually caused "everything is terrain"

Not the lack of color — gray renders classify fine (the witch scored top-1
"witch" fully uncolored). The real causes, found by looking at renders:

1. **The sample really was mostly scenery** (barrels, bases, painting
   handles). The label was correct; the taxonomy lacked a "prop" category.
2. **Support structures**: presupported models are encased in a strut cage
   that legitimately renders as scaffolding → filter out
   `presupported/supported/base/hollow` paths and `._*` AppleDouble files.
3. **Near-uniform low scores across all categories = "nothing fits".**
   SpaceTavernKeeper: terrain 0.095 / evil 0.092 / archer 0.090. Adding
   `human townsperson or worker` → decisive 0.132. When top-k scores are
   within noise of each other, fix the taxonomy, not the renderer.
   Scores are only comparable within a single run/setup.

## Rendering STLs for CLIP-family models

- **Orientation matters and conventions are mixed** even within one
  collection (Loot Studios = Y-up, others Z-up). Auto-detect the up axis via
  the flat print base: surface-sample points + normals, score all 6 axis
  directions by "how much down-facing flat surface sits in the bottom 2%
  slab". Verified on witch/gravedigger/bunny/building. Fails gracefully on
  symmetric objects (barrel: ±Y identical).
- **The flat base is missing on a third of this collection**, so the
  print-base heuristic alone is not enough. Measured over a 70-mesh sample:
  31% score below the `ABS_SCORE_FLOOR` (no base at all — leaping figures,
  flying creatures), 24% are an ambiguous ratio, 41% cannot be decided
  without escalating. Median best score is 0.058 against ~0.39 for a genuine
  base. The design assumed the candidate-list geometry would carry tier 1;
  for characters it does not.
- **Geometry and SigLIP fail on opposite populations, so average them.**
  Hand-labelled 23 of a 40-mesh random sample (17 more had no defined
  upright — a moustache, a gate pin, a flat gear disc, a dragon in flight):

  | | terrain / scatter | characters |
  |---|---|---|
  | flat base present | almost always | often none |
  | geometry | good | fails |
  | SigLIP upright probes | fails | 11/11 |

  Alone: geometry 17-18/23, SigLIP 19/23. Averaged: 21-22/23, the oracle
  ceiling — every disagreement had exactly one method right. The one
  unrecoverable model (`Bedienkonsole`, a console with a large flat back
  panel) is wrong under both.
  **That 22/23 did not replicate — see the holdout below. Treat it as the
  number a tuned-on sample produces, not the method's accuracy.**
- **Min-max before averaging is doing real work, not just unit conversion.**
  The two scores are a surface-area fraction and a difference of cosine
  similarities. Because geometry's weakest candidate is almost always exactly
  0, min-max maps its runner-up to `runner/best` — precisely the `ratio`
  confidence already reported (measured: mean |margin − (1−ratio)| = 0.015).
  So geometry votes with a ~1.0 margin when it has base evidence and ~0.0 when
  guessing, and SigLIP decides those models. Ratio-weighting for free.
  Schemes that discard this lose: z-score 21/23, Borda rank 19/23, and
  "scale geometry against the absolute 0.02 floor" 20/23 — the last because
  every candidate over the floor saturates at 1.0 and the margin vanishes.
  A hard switch (geometry if a base was found, else SigLIP) only reaches
  19/23; the soft blend beats it because partial base evidence should be
  weighed, not thresholded.
- **Run the ensemble on every model, not just low-confidence ones.** Gating it
  on `needs_arbiter` drops to 21/23: `32mm_Gate_L` has a 0.43 ratio and a
  0.033 best score — confident by both tests — and is still wrong.
- **Probe wording swings this more than anything else**: 83% down to 4%
  across phrasings. Never phrase them anatomically ("head at the top") — half
  this collection is terrain and that scores 0/12. And always include negative
  probes: with upright probes alone, raw similarity is near-flat across the
  six tiles and the argmax is noise (4%, worse than chance).
- **`sample_points_uniformly` is unseeded.** With 4000 points the winner can
  rest on ~30, and picks moved between runs on identical input
  (`Propane_Tank` −Z→+Z, `32mm_PitFiend` −X→+X, confidence 0.23→0.65). That
  makes the pose cache irreproducible. `o3d.utility.random.seed()` fixes it —
  verified identical scores across runs. Seed before sampling, not once at
  startup, so the result doesn't depend on call order.
- **Fixed world-space lights leave orbit views black** (back view =
  silhouette). Use a camera-following headlight: per view,
  `sun_dir = normalize(center - eye) + [0, 0, -0.6]`, renormalized — every
  azimuth lit, shadows still fall consistently with "up".
- **The sun is the only light Open3D actually gives you here.**
  `add_directional_light` / `add_point_light` / `add_spot_light` all return
  `True` and then contribute nothing — measured <0.1/255 mean change on a lit
  sphere, and a full "3-point rig" rendered pixel-identical to sun-only. Don't
  design a lighting fix around them; verify a light did something before
  building on it.
- **A single light with no ambient means detail dies in shadow, not just
  darkens.** With `enable_indirect_light(False)` anything facing away from the
  key falls to *pure* black: ~11% of object pixels under 25/255, and on
  `32_Unsupported_Arkham_BodyMask` a hat brim turned the whole masked face into
  a featureless hole. Nothing downstream can recover it — SigLIP and the pose
  VLM both see a silhouette.
- **Indirect light is the fill, kept far below the key.** It is world-fixed
  with no rotation API (in a Z-up scene it lights from the side), which is why
  it was originally disabled — but that argument only rules it out as a *key*.
  At `set_indirect_light_intensity(10000)` against the 90k sun, crushed blacks
  go to exactly 0.000 on every view tested while the bias it adds stays a
  brightness swing (13→29/255 across azimuths), not a shading direction.
  A multi-pass camera-relative sun composite avoids the bias entirely but only
  reached 0.010 crushed at 3× the render cost, and flattened form. Escalate
  intensity only until the blacks clear: 20k+ visibly washes out cloth detail.
- STL has no color; paint a neutral gray. Multi-view (4 azimuths) averaged
  embeddings smooth over a single bad view.

## Performance

- **Cache image embeddings, not classifications.** Categories live entirely
  on the text side of the dot product, so per-file embeddings (keyed by
  path + mtime + render params + model) make category iteration nearly free.
- Cached fp32 vs fp16 live model → dtype mismatch at matmul; cast on load.
- **Profile before assuming the GPU work is the bottleneck.** On warm runs
  the time went: 129 s directory walk (USB drive over FUSE — slow on *every*
  run, page cache doesn't help), 9 s model load, <1 s actual scoring.
  Fixes: `os.walk` with excluded dirs pruned via `dirnames[:]` before
  descending (never enters `Supported/` trees), then cache the file list
  itself (JSON keyed by root+skip-tags, explicit `--rescan` to refresh,
  vanished files dropped automatically, age printed each run).
- **Flags that force work must compose with caches.** `--save-renders`
  originally meant "always re-render" and silently turned every run into a
  cold run (`0 from embedding cache` — user-visible symptom: "the cache
  isn't helping"). Fixed semantics: use cache when that file's render PNGs
  already exist, re-render only files whose renders are missing.
- Failed files (corrupt STL → RENDER_ERROR) are not cached, so they retry
  every run. One known: `A Light In the Shadow/.../32mm_FloatingRock2.stl`
  ("no triangles", ASSIMP can't parse it).
- `export HF_HUB_OFFLINE=1` skips the HF Hub version check per run.
- Warm-run anatomy after all fixes: ~10 s total ≈ model load + walk cache
  read + milliseconds of matmul (was ~3 min).

## Tool family (final architecture)

Three scripts sharing the caches, one concern each:

- `classify_stls.py` — the committed batch pipeline: walk → render → embed →
  cache → CSV. The only script that *writes* caches.
- `test_categories.py` — interactive REPL: loads cached matrix + SigLIP once,
  then Enter = reclassify with categories.txt (distribution table + diff vs
  previous iteration), typed text = instant ad-hoc query over the collection.
  Keeps the model warm so category iteration is milliseconds, not 10 s.
- `cluster_models.py` — k-means over the cached matrix, no text model at all
  (~3 s). Prints most-central members per cluster, writes clusters.csv, and
  builds a contact-sheet PNG per cluster from saved renders for one-glance
  naming.

Intended loop: cold classify pass → cluster to see what the collection
actually contains → write categories.txt from that structure → tune in the
REPL → final classify run for the canonical CSV.

Clustering validation (Loot Studios, k=8): clean semantic groups emerged
unsupervised — robots, sci-fi humanoids, brutes, monsters — and "terrain"
split into fortifications / scatter terrain / weapon accessories, i.e. the
collection itself suggests taxonomy refinements.

## View pooling (per-view cache format)

- Scoring against the *averaged* view embedding equals averaging per-view
  scores (dot products are linear) — so mean pooling gives a single-view
  feature ~25% weight. Fine for whole-object category, bad for categories
  that hinge on one angle (undead details, held items).
- Cache stores per-view vectors (4×1152 per file); pooling happens at scoring
  time via `--pool mean|max|softmax`, so switching modes never re-renders.
- Measured on 73 models, mean→max changed 13 assignments. Wins: models whose
  identity is angle-dependent (TatteredTroopers → skeleton/undead,
  FemaleOrcWarrior → orc). Cost: ornament-driven false positives (Cannon →
  undead via its skull decorations). Max also runs higher raw scores than
  mean — raw thresholds are pool-mode-specific.
- Cache format changes are cheap if renders are saved: re-embed from the
  PNGs (~24 s for 73 files), never re-render. Version the cache key ("|pv")
  so old entries orphan cleanly.

## Canonical pose: confidence, VLM arbiter, front view (2026-08-11)

- **`sample_points_uniformly` interpolates vertex normals by default**, which
  blends normals across creases (a cone's base rim inherits mantle tilt) and
  destroys the flat-base up-detection signal on vertex-sharing meshes. Pass
  `use_triangle_normal=True`: measures actual face orientation and drops the
  hidden compute-vertex-normals-first precondition. Real STLs mostly dodge
  this (the reader duplicates vertices per facet) — procedural test meshes
  don't, which is how the tests caught it.
- **Up-detection confidence = runner-up/best flat-base score ratio.** Cone
  (one flat face) is decisive; torus measured 0.98, cylinder ≈1.0 (±Z caps
  identical). Ratio > 0.6 or best score < 0.02 escalates to a VLM arbiter;
  everything else never pays for one. A box is *maximally* ambiguous under
  this scorer (six flat faces) — don't use one as a "decisive" test fixture.
- **Warm caches can survive a pose-pipeline redesign.** The heuristic's
  answer is a deterministic function of the file, so heuristic-resolved
  poses keep the legacy cache-key token ("auto"); only VLM *overrides* —
  where the render genuinely differs — re-key as `vlm:x,y,z`. Rebuilding the
  pose logic invalidated zero existing embeddings.
- **Front view is metadata, not a render decision.** Score cached per-view
  embeddings against front/back text prompts (mean front-sim − mean
  back-sim, argmax) and record the winning index; REPL links and contact
  sheets reorder views at display time. No re-render, caches untouched, and
  the choice retunes for free when prompts change (after deleting
  pose-cache.json — its identity doesn't cover prompts).
- Don't hardcode ollama model tags: the plan assumed `gemma3`, the actual
  pull landed as `gemma4:26b` — hence `--pose-vlm-model`.

Verification results (2026-08-11, all end-to-end on real renders):

- **Poses/fronts confirmed by eye**: bunny detected Y-up (conf 0.16) and
  rendered upright; its `front_view=3` is the face-toward-camera view (view 1
  is the rear). Torus (conf 0.97) escalated to the arbiter as designed.
- **Disable thinking on ollama VLM calls.** gemma4:26b with thinking on
  spent every token in the `thinking` field and returned *empty* content
  under `format` constraints — 3859 tokens / 281 s vs 7 tokens / 25 s with
  `"think": false`. The empty answer degrades identically to a timeout, so
  the symptom (heuristic fallback) hides the cause. `_ask_ollama` now sends
  `think: false` and retries without the field if the server 400s on it;
  timeout bumped 120→300 s for CPU-offloaded models.
- **VLM + SigLIP contend for VRAM (8 GB GPU, 17 GB model): load order
  decides who wins.** Pipeline order (SigLIP first, ollama call later) works
  — ollama CPU-offloads what doesn't fit. Warming ollama *first* leaves
  SigLIP nothing to load into → hard CUDA OOM.
- **A fresh VLM override must invalidate saved renders.** `--save-renders`
  reuses on-disk PNGs on an embedding-cache miss, but those were rendered
  under the old pose — silent wrong-pose embeddings. Guarded: a newly
  resolved `source=vlm` pose forces the re-render.
- Claude CLI backend works in `-p` mode as written (no file-read permission
  issue; 22 s). On the torus it picked "lying flat" where gemma picked
  "standing on edge" — both defensible for a symmetric shape; arbiter answers
  on degenerate geometry are a coin flip between valid conventions.
- Live-transport test doubles as a degrade-path check: with ollama up but
  the default `gemma3` tag unpulled, `ask_vlm_up` returns None (heuristic
  kept), never raises.

## Elevation rings + the run manifest (2026-08-11)

Higher `--render-size` (512 → 2048) visibly improved classification and the
speed cost was acceptable, which made a second view axis worth paying for too.

- **"Turntable" was already the azimuth.** The original loop orbited
  `az = 2πi/n` at a hardcoded 20° pitch, so the axis actually pinned was
  *elevation*. Worth naming precisely: "add another rotation" is ambiguous
  between the two, and the fix is in the parameter nobody had exposed.
- `--elevations 20,-10,55` renders a full `--views` ring per elevation
  (product, not sum). Ordering is **elevation-major** so views
  `0..n_views-1` remain the first ring — `view0.png` keeps meaning the same
  camera as every previous run, which is what lets saved renders and
  `front_view` indices stay meaningful across the change.
- **A constant `up` silently kills azimuth near the poles.** Passing world
  `[0,0,1]` to `setup_camera` looks safe at any elevation short of ±90, but
  Filament's `lookAt` calls `up` degenerate once `|up · view| > 0.999` — that
  is **|elev| > 87.44°, not 90°** — and substitutes a *fixed* fallback up.
  The fallback doesn't rotate with azimuth, so a whole ring collapses: a
  `--elevations 20,89,-89,-20` run rendered its two polar rings as 8 copies
  of one camera (mean |view₈ − view₁₂| was 3/255, all of it shading; the ±20°
  rings differ by 13–17). The ±89° clamp was reasoning about the ±90°
  singularity and landed *inside* the broken band.
  Fix: carry `up` around the orbit instead of holding it fixed —
  `(-cos az·sin elev, -sin az·sin elev, cos elev)`. It is exactly orthogonal
  to the view direction, so `lookAt` never falls back, and below 87° it is
  the same frame `[0,0,1]` already produced (verified: max pixel delta 2/255,
  the warm cache stays valid). The poles are now ordinary cameras, so the
  clamp relaxes to ±90.
  Generally: a look-at `up` that is *near* the view direction is already
  broken, and it fails silently — identical-looking frames, no error. Derive
  `up` from the same parameters as `eye`, don't hardcode it.
- The pose contact sheet stays pinned at one 20° tile. Up-detection input
  shouldn't change shape as a side effect of a classify-side render flag.
- **Append-only cache keys, again.** Same trick as the pose token (above): a
  single default 20° ring appends *nothing* to the key string, so every key
  written before elevations existed stays byte-identical. Measured: 24/24
  real files still hit the warm 2048px cache (the 25th is the known
  FloatingRock2 RENDER_ERROR, never cached). Two uses in two sessions —
  treat "extend the key without disturbing the default path" as the default
  approach for cache-format changes here, not a one-off.
- `front_view` is an argmax over *all* views, so with multiple rings the hero
  shot can land on a non-primary elevation. Acceptable (more angles to find a
  face in) but it means heroes can shift on already-processed files.

**Config belongs next to the cache, not in the repo.** The three scripts each
re-declared `--views/--render-size/--model/--up-axis` under a comment reading
"must match the classify_stls.py run that built the cache" — a comment that
names a drift hazard is a design smell, not documentation. Options considered:
a committed `.env`, or a manifest written by the run itself.

- **A committed config can drift from the cache it describes** (edit the file,
  forget to re-run classify, downstream silently reads the wrong tiles). A
  manifest written *by* the run cannot: whatever built the cache is what
  describes it. Chose `<cache-dir>/run-params.json`, gitignored along with the
  cache — it's derived state holding an absolute path to an external drive,
  meaningless on another machine.
- Mechanism: `parser.set_defaults(**manifest)` before `parse_args()`, after a
  `parse_known_args()` pass to learn `--cache-dir`. Explicit flags still win
  because set_defaults only moves the fallback. Print which keys came from the
  file — silent hidden state is worse than the retyping it replaces.
- **Record the input directory too.** It's the same class of parameter (the
  walk cache is keyed on it) and it's the one argument you can't tab-complete:
  `/run/media/masa/Files\ and\ S/STL/Loot\ Studios`. Both downstream tools now
  run with zero arguments. Guard: a single-file classify run leaves the
  recorded collection root alone rather than clobbering it with a file path.
- Declaring the shared args once (`add_cache_args`) matters more than the
  defaulting: a new cache-identity flag can no longer be added to the writer
  and forgotten in the readers.
- Bonus catch: `cluster_models.py` never had a `--renders-dir` default, so
  contact sheets were opt-in. It inherits the classifier's from the manifest.

## Where a 7-hour run actually went (2026-08-12)

First full-collection run with the pose ensemble: 601 models (602nd is the
known FloatingRock2 parse failure), 7 h 21 m, **~44 s/model**. Profiled the
same settings (2048 px, 8 azimuths × 2 elevations = 16 views, plus 6 up-tiles)
on an *uncontended* GPU:

| stage | avg | share |
|---|---|---|
| mesh load | 2.48 s | 47% |
| up-candidate tiles (6 renders) | 1.20 s | 23% |
| view renders (16) | 0.87 s | 16% |
| tile + view embedding | 0.76 s | 14% |
| geometry up-axis scan | 0.01 s | 0% |
| **pipeline total** | **5.32 s** | |

**5.32 s against 44 s observed — 85% of the run was not the pipeline.** Two
things accounted for the gap, and neither was where I first looked.

- **PNG encoding was ~half the run.** Saving 16 renders at 2048 px costs
  **21–23 s per model** — pixel-bound, so it is constant whether the mesh has
  3k or 1.8M triangles, and it dwarfs every other stage. It never appeared in
  any timing I did until I measured it directly, because it isn't rendering
  and it isn't inference.
- **VRAM contention.** SigLIP (~2.8 GB) and ollama (~4 GB) do not both fit in
  8 GB, so ollama gets evicted between arbiter calls. Measured on one call:
  `load_duration` **10.1 s** against `eval_duration` **0.49 s** — the reload
  costs 20× the inference. One window of the run ran at **187 s/model for
  three hours**; that is the signature. Run the VLM pass separately from the
  render pass, or pin the model resident.

### `compress_level=1` is the whole fix

Measured on real 2048 px renders, 16 images:

| encoding | time | size each |
|---|---|---|
| PNG `compress_level=6` (PIL default) | 23.31 s | 3.2 MB |
| PNG `compress_level=1` | **3.83 s** | 3.9 MB |
| PNG `compress_level=0` | 1.99 s | 12.6 MB |
| JPEG q92 | 0.13 s | 205 KB |
| WEBP q90 | 1.02 s | 100 KB |

**6.1× faster for 22% more disk, and losslessly identical.** ~19.5 s/model,
about 3 hours off a 600-model run, for one keyword argument.

### Saved renders are an *input*, not a debug artifact

The trap that makes the lossy rows above the wrong answer:
`classify_stls.py:505` re-embeds straight from the saved PNGs whenever the
render files exist but the embedding cache misses. So the encoder sits on the
classifier's input path. Measured per-view cosine against the in-memory render:

| encoding | mean cos | worst view |
|---|---|---|
| PNG (any compress_level) | 1.00006 | 0.99941 |
| JPEG q92 | 0.99224 | 0.97238 |
| WEBP q90 | 0.99289 | 0.97480 |
| grayscale PNG | 0.99409 | 0.98658 |
| **render at 512 px instead of 2048** | 0.97587 | 0.95081 |

A worst-case 0.972 reads as "nearly identical" and is not: competing
categories separate by ~0.02–0.03 cosine, so a 0.028 perturbation is the same
order as the signal. **Only lossless compression is safe here.** If lossy
renders are ever wanted, the fix is to stop re-embedding from disk — not to
pick a quality level.

Two things that fell out of that measurement:

- **The renders are not neutral gray.** `max|R−G| = 4/255` on every view: the
  material is gray but the indirect-light fill is a coloured environment map,
  so an `L`-mode conversion is lossy, not free.
- **Render size is a real quality knob, the biggest in the table.** Rendering
  at 2048 and letting the processor downsample to 384 is supersampling, and it
  differs measurably from rendering at 512 natively. `render_size` being in
  the cache key is correct, not incidental.

### Up-tile rendering is geometry-bound, not pixel-bound

`render_up_candidate_tiles` on a 1.8M-triangle mesh: **3.74 s at 2048 px vs
3.33 s at 384 px**. Shrinking the tiles buys nothing. The cost is that it
builds six rotated *copies* of the mesh and `render_views` does
`clear_geometry()` + `add_geometry()` per tile, re-uploading millions of
triangles six times. The fix is to upload once and move the camera. (I
recommended shrinking them first, from a measurement that had mesh loading
folded in — isolate the stage before optimising it.)

### Other measurements worth keeping

- Mesh load averages ~2.5 s at ~15 MB/s effective off the USB drive,
  single-threaded, and scales with file size (121 MB → 9.4 s). Prefetching in
  a thread pool would hide it; Open3D's reader is C++ and releases the GIL.
- CPU never exceeded one core of 16, GPU compute utilisation sat at 1%. Both
  processors were idle while a single-threaded PNG encoder was the bottleneck.
- Production pose sources over 509 models: **heuristic 340, vlm 136,
  ensemble 33**. VLM overrides fell from 31% (pre-ensemble, 157/508) to 26.7%,
  so the ensemble is absorbing arbiter load as intended.

### Ground truth lives in `up_axis_labels.json`

44 hand-labelled up axes — the only ground truth any pose number in this file
was measured against, and expensive to reproduce (labelled by eye from the
6-tile up-candidate contact sheets). Keyed by **path relative to
`collection_root`**, never by sample index: the walk grew 509 → 602 files
mid-session, so a `random.sample(files, 40)` with the same seed no longer
draws the same models. Two sets, and the distinction is the whole point:

- `orig` (23) — the probes and the min-max scheme were tuned against these, so
  any score on this set is optimistic. Never quote it as accuracy.
- `holdout` (21) — drawn from the 562 files `orig` never touched, method
  frozen before scoring.

19 of the holdout's 40 and 17 of the original's 40 were **excluded, not
mislabelled**: loose hands, wings, swords, pipes, pins, a moustache, a flat
gear disc, a dragon in flight. Their upright genuinely isn't defined, and
scoring against a guess would have manufactured signal. Excluding ~45% of a
random sample is itself the finding — a large part of this collection is
parts and props, not posable models.

Before this file existed the labels were three drifting copies of a `GOLD`
dict in separate scratch scripts plus a JSON keyed by sample index. Persist
the labels, not the harness.

### The holdout: the ensemble's win was mostly selection effect

Fresh 40-model sample drawn from the 562 files the first sample never touched,
probes and normalisation **frozen**, 21 hand-labelled (19 excluded — hands,
wings, swords, pipes, a throwing axe). Scored blind:

| | geometry | ensemble | VLM | pipeline |
|---|---|---|---|---|
| original (tuned here) n=23 | 74% | **91%** | 78% | 91% |
| **holdout (frozen) n=21** | **86%** | **81%** | 76% | 81% |
| pooled n=44 | 80% | 86% | 77% | 86% |

**On unseen data the ensemble scored *below* geometry alone.** The first
sample simply contained more of the models geometry fails on (the wolves, the
gas cylinders, the gate); the holdout is dominated by upright +Z figures that
the print-base heuristic already nails, so there was little left to win and
one model to lose (`Mortimer_BodyNoMask`, which the ensemble flipped to −Z).

The fairest read is the head-to-head, which is not sample-composition
dependent: across all 44 models the two methods **differ on only 6**, and
there the ensemble is right 4, geometry 1, both wrong 1. So it does help — on
14% of models, not the 26% the first sample implied, and 4–1 on six trials is
p≈0.375 by sign test. **"Probably helps, unproven"** is the honest summary,
and it is worth keeping only because it is cheap (~1.6 s/model).

Generalised: a 17-point gap measured on the sample that shaped the method
became −5 points on fresh data. Always spend the holdout before believing a
number, and never quote the tuned figure as the accuracy.

### The contact sheet was starving the VLM — `thumb=256` is too small

`make_contact_sheet(tiles, thumb=256)` produces a 768×512 sheet where each
candidate is 256 px and the miniature inside it maybe 150 px. Rebuilding the
same 44 sheets at `thumb=512` (1536×1024, numerals scaled to match) changed
the answer to every question below:

| pooled n=44 | @256 | @512 |
|---|---|---|
| gemma4:26b | 34/44 | 37/44 |
| sonnet | 27/44 | **37/44** |
| haiku | 20/44 | 26/44 |

As the arbiter tier, gemma goes from net 0 to **net +1**, and sonnet from
**net −4 to net +3** — a full pipeline of **41/44 (93%)** against 38/44 for
the ensemble alone. The tier that looked worthless was resolution-starved.

The tell was in the answer distribution, before any of this was confirmed:
haiku picked `+X` twelve times when the truth is `+X` zero times. A model that
misjudges orientation makes varied mistakes; one that picks the same tile
position repeatedly is guessing. **Check the distribution of a classifier's
answers against the distribution of the labels — a positional prior is
visible there and invisible in the accuracy number.** (haiku still shows it at
512: `+X` ten times. It is genuinely weaker here, not just starved.)

Also note the production pipeline has been running the arbiter at 256 the
whole time, so every `source: vlm` pose in the cache was decided on a sheet
too small to read.

### At `thumb=256`, the VLM tier was exactly net zero

Same 44 models. The arbiter fires on 24 of them, and the VLM overrides the
ensemble on those:

```
rescued 2   Bedienkonsole, Mortimer_BodyNoMask
broke   2   Concrete Chunk (6), arc-1a-doors-none4
            -> net 0
```

It is also the **worst standalone method (77%)**, below both geometry and the
ensemble, and it is by far the most expensive tier — 354 calls on a
full-collection run. Its errors are stable, not noisy: 3 samples per model on
the first set gave **21/23 unanimous (91%)**, and majority voting changed
nothing, so it returns the *same wrong answer* every time on terrain like
`tile9`, `Bunker_MiniV2_Roof_` and `Floor`. Voting cannot fix it.
Its one genuine virtue: it is the only method that ever got `Bedienkonsole`.

What keeps it from being harmful is the `needs_arbiter` gate — applied to all
44 models it would have been −3. Worth trying anyway: gate on the *ensemble's*
own margin rather than geometry's confidence, since the two models it broke
were ones the ensemble already had right.

**All of the above is the 256 px measurement and stands only there.** At 512
the same tier is net positive. The lesson worth keeping is not "the VLM is
weak" — it is that a tier was nearly deleted on the strength of a number that
turned out to be measuring the input pipeline rather than the model.

### `source` records what *moved* the answer, not what ran

`source: "heuristic"` does **not** mean the ensemble was skipped — it runs on
every model, and the label stays `heuristic` when it agrees with geometry, so
the pose is byte-identical to the legacy answer and the embedding-cache key can
stay the legacy token. Verified by re-resolving 15 `heuristic`-marked models:
**0 of 15 moved.** The name reads like "the ensemble didn't run" and invites
exactly the wrong conclusion; `confirmed` would say it better.

### Open gap: the pose cache is not keyed on render size

The up-candidate tiles are rendered through the main renderer, so the
ensemble's answer depends on `--render-size` — but `pose.file_identity` is only
`path + mtime + size`. Same mesh, different `--render-size`, potentially a
different pose, and the cache serves whichever was computed first. Live
example: `Damaged Roofing (4).stl` resolves `+Y` with 2048 px tiles and `-Y`
with 384 px tiles. It also cost an hour of confusion — a spot-check of cached
poses at the wrong tile size showed a phantom disagreement.
The clean fix is to render up-candidate tiles at a **fixed** size regardless of
`--render-size`, so pose resolution is render-size independent (better than
widening the cache key, and the tiles are geometry-upload-bound anyway so
there is no speed cost either way).

## Open-set queries: detecting "not in the collection"

- Cosine scores are only comparable *within* a query — some phrasings run
  hot, some cold — so a raw threshold alone can't tell "present" from
  "absent"; something always ranks first.
- Z-score against the collection's own distribution per query. Plain
  mean/std z fails when a category is *well*-represented (8 robots inflate
  the mean → the query looks weak); robust z (median/MAD) fixes it.
- **Measured: no z cutoff separates modest correct matches from semantic
  near-misses.** Correct "skeleton" → TatteredTroopers at z 2.4–2.7;
  wrong-but-nearest "witch on a broomstick" → AurochRider (a mounted rider)
  at z 3.7. Layer the defenses instead: z < 2.0 = whole query is noise,
  suppress output entirely; raw score < 0.1 (default --min-score) trims weak
  individual matches; displayed z + clickable render link covers the
  judgment calls no threshold can make.
- Near-misses are often *semantically legitimate* ("wizard with a staff" →
  OrcShaman, who carries a staff) — treat threshold tuning as UX, not truth.

## Filter gotchas

- Substring tags bite: `"supported"` matched inside "**un**supported", which
  in miniature packs means NO supports — the files you want. Strip the
  exception word before tag matching.
- When filter semantics change, bump the walk-cache key or stale cached file
  lists silently keep the old behavior.

## REPL affordances that proved useful

- OSC 8 terminal hyperlinks (`file://` URIs, tty-gated) — linking each result
  to its *render* (what SigLIP actually scored) beats linking the STL for
  judging classifications at a glance.
- Paths shown relative to the collection root (pack context), `:find` for
  absolute paths, `:pool` / `:min` to retune live without restart.

## Repo hygiene

- Git repo in `mini-classify/`; gitignore all derived outputs (embed-cache/,
  *.csv, *renders*/, test meshes). uv venvs self-ignore (`.venv/.gitignore`).
- The embedding cache is "derived" but represents the expensive cold pass
  (~1 h for 1000 models) — worth backing up separately once built.

## Environment / tooling

- **uv replaces conda fine for legacy ML repos** — `uv python install 3.8` +
  faithful pins (torch 2.0.0+cu118) all as prebuilt wheels; system
  CUDA 13.3/GCC 16 never involved. Add `numpy<2` for torch-2.0-era stacks.
- **Read the imports before building from source.** Find3D's README demands
  Pointcept pointops (compile) + FlashAttention ("up to 3 hours") — the
  inference path imports neither necessarily: pointops is never imported, and
  flash-attn ships prebuilt wheels (`cu118torch2.0cxx11abiFALSE-cp38`) on
  GitHub releases. Also: PTv3 has a non-flash fallback (`enable_flash=False`).
- uv + multi-index pinned requirements needs `--index-strategy
  unsafe-best-match` (first-index-wins otherwise).
- transformers 5.x: `get_text_features`/`get_image_features` return a
  `BaseModelOutputWithPooling`, not a tensor — unwrap `.pooler_output`.
- Background-shell output capture proved unreliable for long uv installs in
  this harness; foreground with a generous timeout was dependable.
