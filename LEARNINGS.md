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
  — six review passes, twenty reproduced findings: Deflate64/cp437 archives,
  a swap with no zero-copy moment, CRC-verified redundancy with an override,
  flag-independent destinations, and collision diversion to a fixed point.

## Evergreen notes

- [Queries and filters](docs/learnings/queries-and-filters.md) — open-set
  "not in the collection" detection; filter gotchas.
- [Workflow](docs/learnings/workflow.md) — REPL affordances, repo hygiene,
  environment and tooling.
