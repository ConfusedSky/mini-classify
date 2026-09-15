# Resolved questions

Questions that were in `OPEN_QUESTIONS.md` and are now settled. One line each:
the question, the answer, and a link to the `docs/learnings/` write-up that
holds the full story where one exists. `LEARNINGS.md` is the measured record;
this is the quick "what did we decide about X" index. A question graduates here
(and leaves `OPEN_QUESTIONS.md`) the moment it is settled — see the lifecycle in
`CLAUDE.md`.

## Pose, ensemble and arbiter

- **`UP_TILE_AZIMUTHS = 2`** — adopted; halving the azimuths flips zero ensemble
  picks on all 49 labels, at one extra escalation.
  [2026-08-13 roundtrip](docs/learnings/2026-08-13-roundtrip-tiles-and-the-full-label-set.md)
- **Gate the VLM on the ensemble's margin** — shipped as
  `pose.needs_arbiter_margin`, `MARGIN_THRESHOLD = 0.45`, exposed as
  `--up-margin`; ~20% escalation instead of ~55%.
  [2026-08-12 the 7-hour run](docs/learnings/2026-08-12-where-a-7-hour-run-went.md)
- **Would a 512 px SigLIP help?** — +1/44 and resolution-invariant where
  `patch14-384` flips on render size; `siglip2-so400m-patch16-512` adopted.
  [2026-08-12 the 7-hour run](docs/learnings/2026-08-12-where-a-7-hour-run-went.md)
- **Is there headroom inside SigLIP itself?** — effectively no; throughput flat
  across batch 1–128, `torch.compile`'s 1.10× disqualified by drift.
  (`eval/siglip_bench.py`)
- **Do newer Gemini models beat the arbiter default?** — no; gemini-3.6-flash
  ties, 3.1-pro is worse and dearer, all rescue the same four models. Default
  unchanged. (`eval/gemini_vlm.py`)
- **Raise the contact sheet to `thumb=512`** — done; a large gain for weak
  arbiters (sonnet +10), barely moves gemini-3.5-flash.
  [2026-08-12 the 7-hour run](docs/learnings/2026-08-12-where-a-7-hour-run-went.md)
- **Wire a Gemini backend into the arbiter** — done; `--pose-vlm gemini`,
  `auto` resolves gemini-or-nothing, `--pose-vlm ollama` retired (C-R1-4).
- **Rename `source: "heuristic"` to `"confirmed"`** — resolved differently: the
  vocabulary is `geometry`/`siglip`, mapped on load, no cache migration.
  [2026-08-17 cache schema](docs/learnings/2026-08-17-cache-schema-and-design-by-review.md)
- **Which side of the arbiter retry split gets enumerated** — the permanent
  side; only `pose.VLMRejected` pins `"rejected"`, everything else re-asks.
  [2026-08-21 tri-state pass 2](docs/learnings/2026-08-21-tri-state-pass-2-and-embed-cache512.md)
- **Six of the 49 up-axis label paths are stale** — fixed 2026-08-31;
  re-anchored to the reorganised `Enemies_Part*` folders, `resolve()` deleted,
  all 49 resolve through `common.load_labels()`.

## Cache, renders and identity

- **Does the cache key record the pose but not the recipe that drew it?** —
  answered by `identity.RECIPE_VERSION = 1` (elided at 1) in the ev2 rebuild;
  the recipe surface is a fixed five-item list (`rotation_to_z_up`,
  `orbit_camera`, the 1.4 radius factor, the sun direction, the material).
  [2026-09-01 ev2 rebuild](docs/learnings/2026-09-01-the-ev2-rebuild.md), `docs/cache-rebuild.md` §6
- **Should saved renders keep feeding SigLIP?** — no; embeddings come from the
  `.npy` cache or a fresh in-memory render, saved renders are debug output.
  Also fixed a wrong-key embedding bug (old size's pixels under the new key).
- **`compress_level=1` on saved renders** — superseded; `--render-format`
  defaults to jpg (23.3 s → 0.13 s/model, ~27 GB → ~2 GB).

## Performance and structure

- **Does overlapping render and embed saturate the 4060?** — yes on duty cycle
  (57% → 94%) but only 1.17–1.21× wall-clock (thermal cap); the render-child
  split is adopted.
  [2026-08-13 overlap and the thermal ceiling](docs/learnings/2026-08-13-overlap-and-thermal-ceiling.md)
- **Overlap the VLM arbiter with everything else** — done, 28% end to end
  (631 → 456 s on 74 models).
  [2026-08-12 overlapping the arbiter](docs/learnings/2026-08-12-overlapping-the-arbiter.md)
- **Prefetch mesh loads** — obsolete; the numpy STL parser took `mesh-wait`
  from 18.6% of a run to 0.4%.
  [2026-08-13 instrumented](docs/learnings/2026-08-13-instrumented.md)
- **Split the VLM pass from the render pass** — closed by removal; the arbiter
  is a thread pool of network calls, so no second model lands on the 4060.
- **Dedup pass over the wave-1 `src/` modules** — done; keying helpers moved to
  `src/identity.py`, the AST sweep comes back empty, no cache invalidated.

## Decisions on scope

- **Is the spinning volume retired?** — no; kept general so the library can be
  distributed later. Masa's stated position (2026-08-19), not a hardware
  decision, and not a mandate to generalise everything now.
