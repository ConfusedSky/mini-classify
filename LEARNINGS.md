# Session Learnings — STL miniature classification (2026-08-10)

Journey: set up [Find3D](https://github.com/ziqi-ma/Find3D) → realized it was the
wrong tool for whole-model classification → built a local render + SigLIP
zero-shot classifier (`classify_stls.py`) for a ~1000-model printable
miniature collection.

This file is the index. Each session's write-up lives in `docs/learnings/`,
one file per dated section, split from the original single file with content
unchanged — a citation like "LEARNINGS, Overlap and the thermal ceiling"
resolves through the list below. New sessions get new files; the evergreen
notes at the bottom are amended in place. Open work is tracked separately in
`OPEN_QUESTIONS.md`.

## Session write-ups

- [2026-08-10 — the first classifier](docs/learnings/2026-08-10-first-classifier.md)
  — why Find3D was the wrong tool; rendering STLs so CLIP-family models can
  read them; the embedding cache; the three-script tool family; view pooling.
- [2026-08-11 — canonical pose](docs/learnings/2026-08-11-canonical-pose.md)
  — up axis from flat-base geometry with a confidence ratio, the VLM arbiter,
  and the `front_view` column.
- [2026-08-11 — elevation rings and the run manifest](docs/learnings/2026-08-11-elevations-and-run-manifest.md)
  — multi-elevation turntables, and `run-params.json` so the reader tools stop
  needing the classifier's flags retyped.
- [2026-08-12 — overlapping the arbiter](docs/learnings/2026-08-12-overlapping-the-arbiter.md)
  — deferring network arbiter calls, measured at 28% end to end, and the bug
  where the first version ran slower than what it replaced.
- [2026-08-12 — where a 7-hour run actually went](docs/learnings/2026-08-12-where-a-7-hour-run-went.md)
  — the accuracy campaign: arbiter comparisons across models and sheet sizes,
  the margin gate, the backbone sweep, geo_floor, and the hand-labelled ground
  truth. The largest write-up in the set.
- [2026-08-13 — instrumented](docs/learnings/2026-08-13-instrumented.md)
  — where the run goes stage by stage; rendering happens on the iGPU, not the
  4060; the GIL findings; what the numpy STL parser removed.
- [2026-08-13 — overlap and the thermal ceiling](docs/learnings/2026-08-13-overlap-and-thermal-ceiling.md)
  — a renderer child process takes the 4060 from ~57% to ~94% busy for a true
  1.2×; the 80 W card trades clocks for duty cycle; better cooling moves the
  ceiling; the ascending-sweep benchmark trap.
- [2026-08-13 — the round trip, half the tiles, and the full label set](docs/learnings/2026-08-13-roundtrip-tiles-and-the-full-label-set.md)
  — the pose→embed cycle keeps 1.11× of the overlap's 1.21× in the all-cold
  worst case, clearing the modularization; UP_TILE_AZIMUTHS=2 flips nothing on
  49/49; parser_gate gets its A/A control and passes; the reorg's six orphaned
  labels recovered, hard 5/5 at its true n=5.
- [2026-08-14 — hardening the unpacker](docs/learnings/2026-08-14-unpacker-hardening.md)
  — eight review passes, twenty-eight findings: Deflate64/cp437 archives,
  a swap with no zero-copy moment, CRC-verified redundancy with an override,
  flag-independent destinations, collision diversion to a fixed point — and
  four corrections to this write-up's own first draft.
- [2026-08-14 — data structures, and the queue's transport tax](docs/learnings/2026-08-14-ipc-transport.md)
  — the actor refactor's shapes fixed in `docs/actor-refactor/data_structures.md`;
  the render-child queue's cost isolated from render-wait: the pipe is the tax,
  not pickle, shm is 4–5× — and still only ~0.5% of a cold run, so v1 ships on
  `mp.Queue` behind a transport interface.
- [2026-08-14 — precision, and what it actually flips](docs/learnings/2026-08-14-precision-and-compile.md)
  — the fp16 scoring cast flips 1 of 2943, compile flips 1 of 341, both at
  coin-toss margins; jpeg dilution and the compiled-tower-that-wasn't canary;
  max-autotune is GEMM-gated off this card; `--compile` ships with the
  numeric regime keyed into the embedding cache.
- [2026-08-17 — the cache learned its schema, and the design was reviewed into shape](docs/learnings/2026-08-17-cache-schema-and-design-by-review.md)
  — cache-meta.json and the up_str token (a rename, not a re-embed; the 144
  the walk left behind); map-on-load beats version bumps; and what fourteen
  adversarial passes taught: walk protocols, remove carve-outs, make
  invariants mechanical, guard the clean path; convergence is severity
  falling, not counts.
- [2026-08-17 — camera rotation and the world-fixed fill](docs/learnings/2026-08-17-camera-rotation-and-the-world-fixed-fill.md)
  — the refactor's "rotate into the camera, never the mesh" rule measured and
  reversed (I11): the ambient fill is a world-fixed environment map, so a
  rotated rig lights the model differently — 75/255 on half the pixels under
  the production config. `views` rotates a copy instead; the caches stay valid,
  residency keeps the parse and pays the ~275 ms upload, and the roundtrip
  spike's 1.11× was always measured on this design.

- [2026-08-18 — old against new: the refactor's parity run](docs/learnings/2026-08-18-old-against-new-parity.md)
  — the branch measured against `main` on real runs: every cache key, pose,
  and top-3 ordering matches, and cold single-model runs are byte-identical on
  two of three models. The differences that remain are the renderer's own —
  the same old binary differs from itself by as much (7.0e-03) when a model is
  rendered in company rather than alone, which generalises the pose-cache-state
  entry to any change in the preceding draw sequence.
- [2026-08-18 — the refactored pipeline at scale](docs/learnings/2026-08-18-the-refactored-pipeline-at-scale.md)
  — first real run of the actor architecture: 133 models cold in 6:34, then a
  warm re-run in 12.4 s with a byte-identical CSV and no re-embedding. A
  baseline, not a speedup — but the 32× warm path, the exact 133×16 render
  count and the 1:1:1 rows/embeds/poses are `route()` and invariant 1 holding
  at scale. Tag filtering takes 292 files to 133, so per-model estimates off a
  raw file count run ~2× high.
- [2026-08-18 — the eval rig's tiles are not the old tiles](docs/learnings/2026-08-18-pose-tiles-and-the-draw-sequence-again.md)
  — porting the harnesses onto the render child's own `pose_tiles` changes
  pixels by up to 30/255 on ~1% of them, with geometry, cameras, material and
  framing all identical. The residue is the add/show/hide sequence Filament
  sees, making this the third independent measurement that its output depends
  on the scene's history. It moved no answer: `parser_gate` gives 0/49 picks
  moved and the difference sits inside its own A/A floor.
- [2026-08-18 — the untimed join, and why a missing EGL line is not evidence](docs/learnings/2026-08-18-the-untimed-join-and-the-missing-egl-line.md)
  — a 1-in-17 stall traced by elimination to `child.join()` being untimed on
  the one path that reaches quiescence by *killing* the child rather than
  waiting for it to idle. Two pieces of evidence read as damning and are
  worthless: the progress bar hits 100% at admission, and Open3D's EGL line is
  block-buffered, so its *absence* is a fingerprint of SIGKILL rather than of a
  stall before renderer startup.

- [2026-08-18 — a deferred import charges the caller](docs/learnings/2026-08-18-deferred-imports-charge-the-caller.md)
  — two modules whose docstrings asserted they were light were made heavy by
  the same move: `add_cache_args` charged 836 modules and ~0.9 s of torch to
  every tool that built a parser, and `embed_store` pulled a rendering library
  (+2602) to read `.npy` files. Deferring an import relocates its cost onto
  callers; it removes cost only when the signature already implies the
  dependency, which is why `up_axis_scores(mesh)` may defer open3d and a
  string-returning helper may not. The obvious fix was also the wrong one:
  open3d was 2596 of the 2602 and served exactly one function.

- [The query API, live](docs/learnings/2026-08-19-the-query-api-live.md) —
  2801 models, ready in 16 s, `/query` median **49 ms**. Two SigLIP instances
  fit the 4060 (**4740 of 8188 MiB**), so the server and a classify run
  coexist — unlike ollama, which is a reload thrash rather than a capacity
  problem. The API's top-10 is identical to `test_categories.py`'s, which is
  what `src/query.py` was extracted to guarantee.

## Evergreen notes

- [Queries and filters](docs/learnings/queries-and-filters.md) — open-set
  "not in the collection" detection; filter gotchas.
- [Workflow](docs/learnings/workflow.md) — REPL affordances, repo hygiene,
  environment and tooling.
