## Elevation rings, finally measured (2026-09-01)

The second elevation ring predates every search-quality instrument in the
repo: `--elevations 20,-20` shipped on 2026-08-11 on "visibly improved
classification" — an eyeball judgment — and no number was ever attached to
what the −20° ring buys. The renderer pilot's judged sample makes the
question nearly free to answer now: the cached embeddings are per-view, so a
view subset is a slice of the `(n, 16, 1152)` matrix, not a render.

`eval/renderer_pilot/elev_check.py` (the `ev2_check.py` pattern): rank the
pilot's 12 queries over the 301-model sample from the live embed-cache512
rows (post-ev2 rebuild, STLLibrary) under four view subsets, score nDCG@20
against the pooled GLM-5.3-Flash judgments, and delta-judge only the pairs a
subset ranking newly surfaces — 22 pairs, $0.01, into
`out/judgments_glm_elev.json`. Ordering is elevation-major
(`pose.view_angles`), so views 0–7 are the +20° ring and 8–15 the −20°.

| arm | mean nDCG@20 | vs 16v | queries better/worse | top-10 overlap |
|---|---|---|---|---|
| 16 views (production) | 0.886 | — | — | — |
| 8v, +20° ring only | 0.881 | −0.005 | 4/8 | 0.88 |
| 8v, −20° ring only | 0.878 | −0.008 | 3/9 | 0.88 |
| 8v, both rings, alternate azimuths | 0.875 | −0.011 | 2/10 | 0.93 |

- **Dropping the −20° ring costs −0.005 mean nDCG@20** — half the size of
  the tight-framing gain (+0.012, same instrument: the 16v baseline here is
  ev2_check's 0.887 re-read as 0.886, the ideal sets shifting slightly as
  the 22 new judgments land) and inside the band the pilot called
  "uniformly and insignificantly". The drift is consistently negative
  (4/8 queries better/worse) but per-query swings dwarf it: stone golem
  −0.025, knight in plate armor +0.029.
- **Halve by ring, not by azimuth.** The +20° ring alone is the best
  8-view arm; alternate azimuths of both rings is the worst (−0.011) —
  despite keeping the *rankings* closest to production (top-10 overlap
  0.93 vs 0.88). Staying close to the 16-view ordering and scoring well
  under the judge are different things.
- **What this would buy, and what it wouldn't.** `--elevations 20` is a
  different cache identity — a full rebuild at roughly half the render and
  embed cost, and half the `.npy` footprint. The pose path is untouched
  either way (`UP_TILE_ELEVATION` is fixed and independent of
  `--elevations`). Not acted on: embed-cache512 stays 16 views; this entry
  is the number to weigh against the next rebuild's clock.

Caveats as always: 12 queries, the 301-model pilot sample, one GLM judge —
the same limits as every pilot figure.
