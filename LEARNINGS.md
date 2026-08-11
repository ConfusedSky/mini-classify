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
- **Open3D's indirect/environment light has a fixed Y-up orientation** and no
  rotation API — in a Z-up scene it lights from the side. Disable it
  (`scene.enable_indirect_light(False)`) and use explicit lights.
- **Fixed world-space lights leave orbit views black** (back view =
  silhouette). Use a camera-following headlight: per view,
  `sun_dir = normalize(center - eye) + [0, 0, -0.6]`, renormalized — every
  azimuth lit, shadows still fall consistently with "up".
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
