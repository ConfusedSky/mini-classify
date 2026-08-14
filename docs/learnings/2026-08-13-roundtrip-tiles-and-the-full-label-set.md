## The round trip, half the tiles, and the full label set (2026-08-13)

Three measurements before the modularization: does the pose → embed cycle
survive the process boundary, can `pose-embed`'s tile count be cut, and is the
parser gate attributable now that it has a control. Along the way the six
labelled models orphaned by the Loot Studios reorg came back, so every number
below is over all 49 labels — the first complete-set run since the collection
was reorganised. Harnesses: `eval/overlap_spike.py --modes roundtrip`,
`eval/tile_count.py`, the rebuilt `eval/parser_gate.py`.

### The cycle costs about half the overlap win — in the all-cold worst case

`roundtrip` mode runs the real dependency graph: the child renders a model's
24 tiles and *holds the mesh*; the parent embeds them, resolves the pose with
the production ensemble math, and answers on an unbounded back-edge queue; the
child then rotates and renders the views, working `--inflight 3` models ahead.
Same-session, cool-start against cool-start, 60 cold models:

| mode | wall | 4060 busy |
|---|---|---|
| baseline (sequential) | 130.3 s | 55% |
| overlap, cycle cut | 107.5 s (1.21×) | 91% |
| round trip | 117.3 s (1.11×) | 88% |

This is the floor, not the expectation: a warm run sends ~80%+ of files
straight through as embeds (the 1.21× shape), and the child's stall pattern
(parent idle 13.5 s vs 9.2 s) says a deeper in-flight window buys some back.
The residency question the actor proposal sized an LRU for resolved to a
three-entry dict, and the unbounded back-edge deadlocked nothing. No
architectural blocker: the modularization can start.

### `UP_TILE_AZIMUTHS = 2` flips nothing on 49 of 49

| set | n_az=4 (24 tiles) | n_az=2 (12) | n_az=1 (6) |
|---|---|---|---|
| orig (23) | 21/23 | 21/23 | 19/23 |
| holdout (21) | 18/21 | 18/21 | 17/21 |
| hard (5) | 5/5 | 5/5 | 5/5 |

Zero ensemble pick changes at n_az=2; escalation at the 0.45 gate rises
8 → 9 of 49 — one extra call. Half of the run's largest GPU item for one
arbiter call. n_az=1 is firmly out: it breaks three models *including*
`32mm_Gate_L`, and needs 16 escalations. Margins compress as azimuths drop,
so adopt-then-re-read-the-gate, not adopt-and-assume. (`eval/tile_count.py`
cross-validates: its 4-azimuth ensemble matches `parser_gate`'s arm a exactly,
44/49.)

Two by-catches. `render_up_candidate_grid` is **not** pixel-identical to
rotating the mesh, measured independently of the review that first refuted it:
~1.9 mean|dpx| against a bit-exact renderer, agreeing only on +Z where R is
the identity. And the cached `orbit384x4` eval tiles are stale for 39 of 43
models — a flag on the published `front_first`/`arbiter_gate` numbers, which
sit on that cache.

### The parser gate passes, and margin-level claims are weak *permanently*

`parser_gate.py` now runs an A/A null control (same loader both arms) beside
A/B, alternates arm order per model, calls the production `resolve_up` instead
of reimplementing it, and counts missing meshes loudly. Over 49:

| | A/A (floor) | A/B (parser) |
|---|---|---|
| picks moved | 0/49 | 0/49 |
| escalation crossings | 2 | 0 |
| margin shift mean / max | 0.0187 / 0.1187 | 0.0235 / 0.1706 |

The parser moves no pick and causes no crossing; the do-nothing control causes
two. The `Container_complete` flip seen on a 43-model run did not recur — a
coin flip, exactly as its sub-noise margin said. The standing lesson: at this
renderer, a ~0.02 margin shift is inside the null control's range whatever
caused it, so margin-level claims stay weak by construction — pick-level and
crossing-level evidence is what this harness can attribute.

### The reorg's six orphans, recovered

Five had moved within the collection; `32mm_Gate_L` was still inside
`GateWall_NoSupports.zip` (never unpacked — not a skip-tag bug; the two
`Supported` siblings were correctly tagged). Labels re-pathed, root re-anchored
to `/run/media/masa/STLLibrary`, all 49 resolve. All six score correctly in
both parser arms; `hard` at its true n=5 is ensemble 5/5 against geometry 2/5,
and `32mm_Gate_L` correctly escalates (margin 0.13–0.21) instead of sailing
through — the exact behaviour the ensemble was built for.
