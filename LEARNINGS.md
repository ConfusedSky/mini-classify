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
