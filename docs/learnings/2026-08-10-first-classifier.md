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
- **One `OffscreenRenderer` per process — a second one core-dumps.** Creating
  it does not fail politely; Filament's resource manager throws
  `Trying to destroy nonexistent resource ([VertexBuffer...])` from a
  destructor and the interpreter aborts. Reproduced deliberately, and it is
  also what killed an A/B that tried to compare two lighting configurations in
  one process — the fix there was one config per process invocation.
  This is the hard limit on parallelising the render stage: rendering cannot be
  threaded *or* multi-instanced inside a run, only moved to separate processes
  (which then contend for the same GPU). It is why the async work overlaps mesh
  loading and the arbiter with rendering rather than rendering with itself.
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
  isn't helping"). Fixed semantics: use cache when that file's renders
  already exist, re-render only files whose renders are missing. That is
  still the rule, but the renders no longer feed embeddings — see
  "Saved renders are debug output".
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

