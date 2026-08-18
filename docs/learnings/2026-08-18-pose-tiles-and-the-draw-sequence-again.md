# The eval rig's tiles are not the old tiles (2026-08-18)

Porting the harnesses off `classify_stls.render_up_candidate_grid` and onto
`src.renderer.Renderer.pose_tiles` — the path the render child actually takes —
changes the pixels.

**Measured**, post-processing off, one path per process:

| | |
|---|---|
| difference | max **1–30 of 255** on **0.05–1.1%** of pixels |
| determinism | each path repeats byte-identically across processes (18/18) |
| 24-triangle mesh | zero difference |
| 1.2k- and 69k-triangle meshes | nonzero |

Ruled out one at a time: the mesh bytes (identical), the cameras (identical),
a shared vs fresh `MaterialRecord` (identical pixels), and the residency LRU's
hide-vs-remove (identical pixels with a 1-byte budget, forcing eviction). What
remains is the **add/show/hide sequence Filament sees** before the shot.

## Why this is the third sighting, not a new bug

The same phenomenon, reached a third way:

1. `OPEN_QUESTIONS` recorded renders as irreproducible **across pose-cache
   states** — a cold run draws the six up-candidate tiles before the views, a
   warm run does not.
2. The refactor's parity run
   (`docs/learnings/2026-08-18-old-against-new-parity.md`) generalised it: the
   *same old binary* differs from itself by as much (7.0e-03 per embedding
   component) when a model is rendered **in company rather than alone**, so it
   is not about the pose cache — it is about anything that changes the
   preceding draw sequence.
3. This: two code paths that agree on geometry, camera, material and framing
   still differ, because they add and show geometry in a different order.

Filament's output is a function of the scene's history, not just its current
state. Three independent measurements now say so.

## Why the port is still right

The old tiles were `classify_stls`' single-process arrangement; the new ones
are what `src/render_child.py` produces on every real run. `eval/README.md`
has always claimed the harnesses measure the production path, and until now
that was false for anything touching tiles. The port makes it true.

**It moved no answer.** `eval/parser_gate.py` re-run on all 49 labels gives
`ensemble 44/49, geometry 37/49` in both modes with **0/49 picks moved**, and
its A/A control's own noise floor (mean 0.0239, max 0.1149) is wider than the
difference. `eval/geo_floor.py` reproduced its recorded finding exactly
(p=2 fixes `32mm_Orguss_Head` `+X → +Y` at w=0.14, escalations 9/44 → 6/44).
One cosmetic consequence: `parser_gate`'s verdict *string* flipped from
"exceeds the A/A mean but not its max" to "at or below the A/A noise floor",
because the A/A mean drifted up while the A/B mean did not (0.0237 vs a
recorded 0.0235). Both are the not-attributable branch. That a mean-comparison
verdict can flip while every number holds is itself the argument for keeping
the A/A control.

## What this costs

Tile caches under `eval/out/` predating the port were filled by the old path.
Anything that compares new tiles against stored old ones is comparing across
this difference — small, but not zero. Rebuild rather than mix.
