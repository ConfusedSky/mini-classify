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
  path + mtime + render params + model) make category iteration nearly free:
  ~1000 models ≈ 45–60 min cold, then ~30 s per categories.txt experiment.
- Cached fp32 vs fp16 live model → dtype mismatch at matmul; cast on load.

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
