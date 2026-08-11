# Canonical Pose Pipeline — Design

**Date:** 2026-08-10
**Status:** Approved
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
