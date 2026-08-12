# Canonical Pose Pipeline — Design

**Date:** 2026-08-10
**Status:** Approved — amended 2026-08-11 and 2026-08-12, see [Amendments](#amendments)
**Goal:** Ensure every model renders upright and we know which view faces the
camera — improving classification robustness, render-sheet readability, and
giving each model a canonical "hero" view. Fixes known failures (symmetric
objects like the barrel) by detecting ambiguity and escalating to a VLM.

## Context

`classify_stls.py` renders N azimuth views of each STL and classifies with
SigLIP zero-shot. Today `detect_up_axis` scores 6 axis-aligned up candidates
by "how much down-facing flat surface sits in the bottom 2% slab" (the print
base). It has no confidence measure — ambiguous meshes silently get a
coin-flip — and there is no notion of "front"; azimuth 0 is arbitrary.

Models in this collection sit in one of the discrete axis-aligned
orientations, so the candidate-list approach stays; no arbitrary-angle plane
fitting.

## Architecture: three tiers, cheapest first

1. **Geometry** decides the up axis for every model and reports confidence.
2. **SigLIP** (already loaded, embeddings already cached) picks the front
   view for every model.
3. **A VLM arbitrates only low-confidence uprights** — local Gemma via
   Ollama, or Claude via the Agent SDK.

### Tier 1: Upright detection with confidence

`detect_up_axis(mesh)` keeps its current scoring but returns
`(up, confidence)`:

- `ratio = runner_up_score / best_score`. A clean flat base is decisive
  (low ratio); a symmetric mesh (barrel: ±Y identical) gives ratio ≈ 1.0.
- **Low-confidence** when `ratio > 0.6` or `best_score < 0.02` (no flat base
  found). The ratio threshold is settable via `--up-conf` (default 0.6); the
  absolute floor is a code constant. Defaults get tuned on test-stls during
  implementation.
- High-confidence: proceed as today. Low-confidence: escalate to tier 3.

### Tier 2: Front detection via SigLIP (all models)

Score each cached per-view embedding against fixed text probes:

- Front probes, e.g. "a miniature figurine facing the camera, face visible".
- Back probes, e.g. "a miniature figurine seen from behind".
- `front_score(view) = mean(front sims) − mean(back sims)`; argmax = front.

**Front is metadata, not a re-render.** We record which existing view is the
front; the mesh is never rotated for it. Per-view embeddings, classification
pooling, and the embedding cache are untouched. The front index drives
display only: hero thumbnail, view ordering in render/cluster sheets, REPL
links. With `--views 4` front resolution is 90°; use 8 views for finer.

Symmetric props (barrels) have flat front scores across views; keep view 0.
No escalation — front is meaningless for such objects and nothing downstream
depends on it being "right".

### Tier 3: VLM arbiter (low-confidence uprights only)

Render one labeled contact sheet: 6 tiles, one per candidate up, same
azimuth/elevation. Ask: "Which numbered tile shows the model standing
upright as it would sit on a table?" Require a JSON-only answer
`{"tile": n}`.

- `--pose-vlm {auto, ollama, claude, off}`:
  - `ollama`: local API at `localhost:11434`, gemma3 vision model.
  - `claude`: Claude Agent SDK (user's Claude Code subscription).
  - `auto` (default): try ollama; if unreachable, warn and behave as `off`.
  - `off`: keep the geometric best guess.
- Invalid/unparseable answer → one retry → fall back to geometric best
  guess and log. The pipeline never hard-fails because of the VLM.
- After a VLM correction the model re-renders with the new up, and tier 2
  picks its front. The VLM is never asked about front.

## Pose cache

`pose-cache.json` in the embed-cache directory, keyed by file identity
(resolved path + mtime + size), storing:

```json
{"up": [0, 0, 1], "front_view": 2, "confidence": 0.15, "source": "heuristic"}
```

`source` is `heuristic` or `vlm`. VLM answers are one-time; reruns read the
cache. The embedding-cache key changes from the `--up-axis` argument to the
**resolved up vector**, so only files whose up actually changed re-render and
re-embed; the rest of the warm cache survives.

## CLI and outputs

- New flags: `--pose-vlm` (above), `--up-conf` (threshold override).
- `results.csv` gains columns: `up`, `pose_conf`, `pose_source`,
  `front_view`.
- Render sheets, cluster sheets, and REPL file links show the front view
  first.

## Error handling

- Ollama unreachable / VLM errors: warn once, fall back to geometry,
  continue the run.
- Meshes with no triangles or degenerate extents: unchanged from today
  (skipped with the existing error path).

## Testing

- Unit tests, synthetic meshes: box with flat bottom (decisive, correct up,
  high confidence); cylinder (ambiguous, escalation triggers); rotated box
  (detects Y-up).
- VLM path: test skipped (`pytest.mark.skipif`) when ollama isn't running.
- Eyeball pass: regenerate sheets for test-stls and the known set
  (witch, gravedigger, bunny, building, barrel) and confirm poses visually.

## Out of scope

- Arbitrary (non-axis-aligned) up detection via plane fitting — collection
  doesn't need it; add later if tilted exports appear.
- Re-rendering models so azimuth 0 physically faces the camera — front is
  metadata by design.
- Using the VLM for front detection or for classification.

## Amendments

The body above is left as approved on 2026-08-10. Amendments record what
changed since, and why.

### 2026-08-11 — SigLIP also decides the up axis (tier 1.5)

**What broke.** Tier 1 assumed geometry decides the up axis for every model and
only ambiguity needs escalating. That assumption does not hold for this
collection: over a 70-mesh sample, 31% score below `ABS_SCORE_FLOOR` (no flat
base at all — leaping figures, flying creatures), 24% are an ambiguous ratio,
and **41% cannot be decided by geometry without escalating**. Median best score
is 0.058 against ~0.39 for a genuine print base. The tier-3 VLM was absorbing
that load, one model at a time.

The Context section's other premise — that models sit in discrete axis-aligned
orientations, so the six-candidate list stays — held up and is unchanged.

**What changed.** SigLIP now votes on the up axis as well as the front view,
scoring the same six candidate tiles tier 3 renders, against upright/toppled
text probes. The geometry and SigLIP score vectors are min-max normalised per
model and averaged; the argmax is the up axis.

- Runs on **every** model, not only low-confidence ones. Gating it on
  `needs_arbiter` costs a model that is confidently wrong: `32mm_Gate_L` has a
  0.43 ratio and a 0.033 best score, passes both confidence tests, and picks
  the wrong face.
- Tier 3 is unchanged: same trigger (geometry's ratio and floor — geometry is
  what knows there was no base to measure), and the VLM still wins when it
  disagrees. Tiles are rendered once and shared by both tiers.
- `--no-up-ensemble` reverts to geometry alone.

**Evidence.** 23 hand-labelled models from a 40-mesh random sample (17 further
models excluded — a moustache, a gate pin, a flat gear disc, a dragon in
flight all have no defined upright). Geometry alone 17-18/23, SigLIP alone
19/23, **averaged 22/23** — the oracle ceiling, since every disagreement had
exactly one method right. The two fail on opposite populations: terrain almost
always has a base and defeats the probes, characters often have no base and
defeat the geometry.

Min-max is load-bearing rather than cosmetic. Geometry's weakest candidate is
almost always exactly 0, so min-max maps its runner-up to `runner/best` — the
`ratio` this spec already defines as confidence (mean |margin − (1−ratio)| =
0.015). Geometry therefore votes hard when it has base evidence and abstains
when guessing, with nothing thresholded. Schemes that discard that structure
score worse: z-score 21/23, Borda 19/23, absolute-scaled geometry 20/23, and a
hard geometry-or-SigLIP switch 19/23.

**Not yet validated.** `object_generic`'s probe wording was chosen after
watching earlier phrasings fail on some of these same models, and the min-max
scheme is the best of six measured against the same 23 labels with only 8
disagreements to arbitrate. Both rates are optimistic. A clean holdout on an
unseen sample is outstanding; `--no-up-ensemble` exists to A/B it.

### 2026-08-11 — Pose cache: `source` gains `ensemble`, sampling is seeded

Supersedes "**`source` is `heuristic` or `vlm`**" in [Pose cache](#pose-cache).
`source` is now `heuristic`, `ensemble`, or `vlm`. It records which tier last
*moved* the answer, so an ensemble or VLM run that merely confirms geometry
stays `heuristic` and leaves the embedding-cache key untouched.

Also note a pre-existing drift, unchanged by this amendment: that section says
the embedding key "changes from the `--up-axis` argument to the **resolved up
vector**". The implementation is append-only instead — `embed_cache_token`
returns the `--up-axis` argument for `heuristic` poses and a `source:vector`
token only for overrides, which is what let earlier warm caches survive.

`detect_up_axis` now seeds Open3D's RNG before sampling. Unseeded, the winner
can rest on ~30 of 4000 sampled points, and picks moved between runs on
identical input (`Propane_Tank` −Z→+Z, `32mm_PitFiend` −X→+X, confidence
0.23→0.65) — which made the pose cache irreproducible and would have made the
ensemble unstable.

### 2026-08-12 — Saved renders are debug output, and per-config

Amends "**After a VLM correction the model re-renders with the new up**" in
[Pose cache](#pose-cache), and the assumption behind it.

That rule existed because `--save-renders` output was a second-tier *input*: on
an embedding-cache miss the classifier re-embedded the saved PNGs rather than
re-rendering, so a stale file meant a silently wrong embedding. It no longer
does. Embeddings come from the `.npy` cache or a fresh in-memory render, and
nothing reads the saved images back.

Two consequences for this design:

- **The re-render after an override is now cosmetic, and still required.** The
  embedding re-keys on its own, because an override moves `embed_cache_token`.
  The re-render survives so the files on disk show the pose that was actually
  used — poses are graded by eye off those files. It now also fires for
  `source: "ensemble"`, which the 2026-08-11 amendment added without revisiting
  this rule.
- **Renders live under the camera config that produced them**
  (`<renders-dir>/2048px-8v-e20,-20/`). A render filename carries only stem and
  view index, but the embedding key covers render size, views and elevations —
  so a rerun at a different size used to leave the previous config's images in
  place, and the contact sheets stopped describing what was classified. A config
  change is now a different directory.

Non-goal unchanged: the up-candidate tiles still render through the main
renderer at `--render-size`, so `pose-cache.json` still crosses render sizes.
That gap is recorded in `OPEN_QUESTIONS.md`, not fixed here.
