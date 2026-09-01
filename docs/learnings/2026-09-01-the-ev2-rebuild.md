# The ev2 rebuild — 2026-09-01

The full regeneration of `embed-cache512` under the 2026-08-31 stack: tight
per-view framing (`EMBED_CACHE_VERSION 2`), `RECIPE_VERSION 1`, exact
rotation matrices, the paced gemini arbiter with the GLM fallback armed
behind `auto`, and per-judge `arbiter` provenance on every settled pose.
Run by Masa 2026-08-31, `classify_stls.py /run/media/masa/STLLibrary
--cache-dir embed-cache512 --views 8 --elevations 20,-20 --render-size 512
--model google/siglip2-so400m-patch16-512 --pose-vlm auto --save-renders`,
into an empty directory — so nothing below is filtered through a shim.

## The run, from the cache it left

All figures derived from the artifacts (`pose-cache.json`, the walk list,
file mtimes), since the terminal scrolled away; wall-clock is
mtime-bounded.

| | |
|---|---|
| models walked / posed / scored | 3,380 / 3,380 / 3,380 rows |
| wall clock | ~4 h 52 m (walk 15:08 → manifest+CSV 20:00) |
| poses by source | geometry 1,870 · siglip 566 · **vlm 944** |
| arbiter escalations | **1,526 (45%), all answered, zero rejected, zero unanswered** |
| judge, per the new provenance field | `gemini/gemini-3.5-flash` on every one — `auto` never degraded |
| arbiter moved / confirmed | 944 / 582 |
| arbiter bill (est. from 2026-08-12 per-call usage, ~$7.6/1k) | ~$11.5 |
| embeddings | 3,555 `.npy` (3,380 + 175 re-keyed by mid-run pose changes) |
| renders | 54,080 jpg = 16 × 3,380 exactly, under `512px-8v-e20,-20-ev2` |
| cache size | 3.1 GiB |
| ups | +Z 1,210 · +Y 1,168 · ±X 583 · −Y 277 · −Z 142 |

Three things this settles:

- **§8 of docs/cache-rebuild.md is closed by construction.** Every gated
  model reached the arbiter and every entry says who answered. The old
  cache's confirmed-vs-never-asked ambiguity (1,243 indistinguishable
  entries on embed-cache2) cannot exist here: `arbitrated: true` ∪ absent
  is the whole population, and absent means above-gate.
- The escalation rate ran **above** the census: 45% against the ~39–44%
  the `MARGIN_THRESHOLD` comment predicted — consistent with a collection
  that kept growing and with tight framing changing ensemble margins
  (which the framing change's key bump priced in).
- The "Taken 2026-08-31" notes in cache-rebuild.md all held: no legacy
  source spellings, no bare-int `front_view`, every entry at v4 with a
  margin, one render format in one `-ev2` directory. The manifest was
  written by the old exit-time `finally` (the run predates c05c14d by
  hours), which is why a mid-run serve needed explicit flags one last
  time.

## Did tight framing pay? Yes — by the amount the pilot predicted

`eval/renderer_pilot/ev2_check.py`: the pilot's 12 queries over the same
301-model sample, before = the archived pre-ev2 control embeddings
(`arm_open3d.npy`), after = the rebuilt cache's rows, **one judge for both
sides** (GLM-5.3-Flash; the existing 388 judgments covered all but 2 new
pairs — the delta cost ~a cent).

| | nDCG@20 (GLM judge) |
|---|---|
| before (loose 1.4× framing) | 0.876 |
| after (tight per-view fit) | **0.887 (+0.012)** |

8 queries better, 4 worse, mean top-10 overlap 0.86. The pilot's
open3d-tight arm predicted +0.015–0.018 under this judge; the live
rebuild delivered +0.012 — same direction, same order, on embeddings the
pilot never saw. Small and real, exactly as sold: the framing change was
never the lever, it was the free win taken while the `ev2` invalidation
was being paid anyway. The biggest single mover in both directions is
judge-noise territory (knight −0.086, bust +0.059), consistent with the
pilot's finding that renderer deltas live inside one or two rank swaps
per query.

Both run artifacts are in `eval/renderer_pilot/out/` (`ev2_rankings.json`,
`judgments_glm_ev2.json`); the script stays beside the pilot for the next
recipe change.
