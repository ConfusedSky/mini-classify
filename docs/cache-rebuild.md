# Debt payable only at a full cache rebuild

Everything in this repo that exists because **an old cache must keep working**.
None of it is a bug; each item bought something real — usually hours of
rendering or paid arbiter calls that would otherwise have been thrown away.
But each one is also a shim whose only job is to make yesterday's bytes
readable, and a rebuild is the one moment they can be deleted rather than
carried.

**When a rebuild is planned, work this list first**, decide which items to
take, and delete them in the same change that regenerates the cache — not
afterwards, or the shims will look load-bearing again.

Scope note: "the cache" means the whole set — `pose-cache.json`, the per-view
`.npy` embeddings, and the saved renders, for the primary collection. A
rebuild is hours of GPU time; the point of this document is that the hours buy
more than fresh numbers if the debt goes with them.

Most items here are shims to delete. **§8 is the exception and the one with a
deadline of its own** — it is quality the collection does not currently have
rather than code it no longer needs, and it is the only item with a cheaper
route than a full rebuild.

---

## 1. The rotation table keeps floating-point noise on purpose

`src/pose.py`'s `_AXIS_ROTATIONS` stores entries like
`6.123233995736766e-17` and `-0.0` where the mathematically correct values are
`0.0` and `±1.0`. They are `cos(π/2)` and `sin(π)` noise from the Open3D
construction the table replaced, kept **byte for byte** because that is what
drew every cached embedding — cleaning them would re-pose every non-`+Z`
model under unchanged cache keys.

**At a rebuild:** replace the table with exact integers, and simplify
`rotation_to_z_up` to the six exact matrices plus the Rodrigues fallback.
`tests/test_pose.py::test_rotation_to_z_up_matches_open3d_bit_for_bit` is the
test that must change with it — it asserts equality against Open3D's
construction, which is exactly the thing being dropped. Replace it with an
assertion of the exact matrices plus the rotation properties.

Do not do this piecemeal. It is item one on the recipe list (§6) and the whole
reason that list exists.

## 2. Key elisions that exist to keep old keys byte-identical

`identity.cache_key_from_identity` appends a token **only when non-default**,
so that keys written before each flag existed still hash the same:

| token | elided when | note |
|---|---|---|
| `\|e:` | `--elevations` is `[20.0]` | keys predating `--elevations` |
| `\|compiled` | `--compile` off | the two numeric regimes cache separately |
| `\|ev<n>` | `EMBED_CACHE_VERSION == 1` | appended only when bumped |

**At a rebuild:** these can become unconditional, which makes the key say what
it means instead of encoding its own history. `EMBED_CACHE_VERSION`'s elision
in particular is a trap for a future reader — the version is invisible in
every key that currently exists.

Weigh it: unconditional tokens are clearer but permanently break every cache
built before the change, including the eval caches nobody plans to rebuild.
Deciding to keep the elisions is a legitimate outcome; deciding by accident is
not.

## 3. Legacy shapes in the pose cache

* `Pose.from_cache` (`src/pose.py`) absorbs **bare-int `front_view` entries**
  from before front-view was keyed per view config, and a missing `margin`,
  and defaults `v` to 0.
* `pose.RENAMED_SOURCES` maps `heuristic → geometry` and
  `ensemble → siglip` — spellings retired long ago.
* `done.py:175` handles a legacy int `front_view` on the write side.
* `load_pose_cache` drops entries below `POSE_CACHE_VERSION`, whose changelog
  (v2 four-view ensemble, v3 geometry attenuation, v4 512 px contact sheet)
  only matters for entries that predate v4. **A bump also wipes all four
  `arbitrated` states** (§8): the dropped entries take every `"rejected"`
  with them, and the whole gated population is re-billed at the arbiter on
  the next run. Price a v5 with §8's overnight figure in hand
  (docs/tri-state-pass-2.md, 2026-08-21).

**At a rebuild:** every entry is written fresh at the current version, so all
four can go. `from_cache` becomes a plain constructor.

## 4. `cache-meta.json` version 0

`cachedir.CACHE_VERSION` documents `0 = unstamped (every cache from before the
stamp existed)`, and `require_cache_version` exists to refuse those. The
up-token elision it guards against — deterministic poses keyed as the
`--up-axis` string — cannot occur in a fresh cache.

**At a rebuild:** the version-0 branch is dead. Keep the *stamp*; drop the
handling of its absence.

## 5. `migrate_cache_keys.py`'s old key formats

`old_base`, `old_identity`, `old_cache_key`, `old_render_key` and
`old_embed_cache_token` are preserved copies of key schemes no longer written
— absolute paths, full-nanosecond mtimes, the elided up-token, the flat
pre-`embeds/` layout.

**At a rebuild:** the tool's reason for existing goes with the caches it
migrates. Either delete it or reduce it to the root-move case (a library that
changes drives), which is the one migration a fresh cache can still need.

Note this cuts the other way too: **run the migration before rebuilding, not
after.** Anything it could still rescue is cheaper to migrate than to
re-render.

## 6. The render recipe is not in the cache key

The open question in `OPEN_QUESTIONS.md`: the key covers views, elevations,
render size, model and `--compile`, but not the *recipe* — `rotation_to_z_up`,
`orbit_camera`, the 1.4 radius factor, the sun direction, the material. A
change to any of them re-poses cached models silently.

**A rebuild is the cheapest moment this will ever have.** Introducing a recipe
version normally costs a full invalidation; during a rebuild that cost is
already being paid. Introduce it as version 1 = the recipe as rebuilt, with
the changelog form model-browser's `RIG_VERSION` uses (an integer says the
cache is invalid; the log says which models to look at and why).

Doing this *at* the rebuild also removes the awkwardness described in
OPEN_QUESTIONS about introducing v1 elided-from-the-key to avoid invalidating
existing entries — there are no existing entries to protect.

## 7. Mixed render formats on disk

`cachedir.render_index` is deliberately extension-agnostic because a renders
directory may hold PNGs written before `--render-format` existed beside newer
JPEGs. A rebuild writes one format.

**At a rebuild:** the newest-wins tie-break can go. The extension-agnostic
*lookup* is probably still worth keeping — it costs nothing and the next
format change is free — so this is the smallest item here.

---

## 8. Poses that escalated to the arbiter and never got an answer

`pose_is_sufficient` counts any entry with a non-`None` `margin` as a hit. So
a model whose ensemble margin fell below the gate, escalated, and then got no
VLM answer — because the arbiter was off, or because Vertex refused with a 429
— is cached as `source: siglip` (or `geometry`) with its low margin, and **no
later run re-escalates it**. The entry looks complete because it is: the
ensemble really did answer. What is missing is the arbitration the margin
asked for.

Measured on `embed-cache2` after the 2026-08-19 `--rescan` completed, against
the **3396 loaded models** (the pose cache holds 3540 entries; the other 144
are orphans whose files the walk no longer sees):

| source | models |
|---|---|
| `geometry` | 2110 |
| `siglip` | 1072 |
| `vlm` | **214** |

**1243 of 3396 (37%) are non-`vlm` with a margin under the 0.45 gate.**

**Read that number carefully — an earlier revision of this section read it
wrong.** `source` becomes `"vlm"` only when the arbiter *moved* the answer
(`poser.py:258-262`: "a confirmation keeps the label"). So 214 counts
arbitrations that *changed* a pose, not calls that succeeded, and the 1243 is
**refused ∪ confirmed**: models the arbiter declined to answer for, and models
it answered for by agreeing. Those two are indistinguishable on disk.

That cuts both ways. The damage is smaller than "1243 models never got their
arbitration" — some of them did, and were confirmed. But nothing can tell
which, so a rebuild cannot target the refused ones either, and neither can any
report. **The inability to separate those populations is itself the strongest
argument for recording a refusal on the entry**, which is the fix described
below.

This is a consequence of a deliberate rule rather than an oversight. Treating
`margin is None` as a miss is what stops one `--no-up-ensemble` pass pinning
every model to its geometry answer forever; treating a *low* margin as a miss
would instead re-escalate — and re-bill — the same ~39% of the collection on
every run, whether or not anything had changed.

**At a rebuild:** poses resolve from scratch, so every model under the gate
actually reaches the arbiter. Two things to set before starting, both from
2026-08-19:

* Run with the paced arbiter — `--arbiter-workers 4`, `--arbiter-min-interval
  1.0`, now the defaults. The un-paced 8-worker pool is what turned a quota
  refusal into a storm, because a 429 returns in milliseconds and frees a
  worker to fail again.
* Budget the calls honestly. `pose.py`'s own live-margin census says **~1227
  paid calls, not ~560** for a cold collection, at roughly $0.30 per
  full-collection run — and the 1243 measured above, on a collection that has
  since grown, is that census landing almost exactly.

**A rebuild is not the only route**, and this is the one item here that has a
cheaper one: deleting just the pose entries where `source != "vlm" and margin
< gate` makes exactly those models re-escalate on the next ordinary run, with
no re-rendering and no re-embedding. Worth preferring if the arbiter's answers
are the only thing wanted — though note it re-escalates the confirmed models
too, because nothing distinguishes them. **Superseded for this population as
of 2026-08-21**: absence now reads as `false`, so the next arbiter-on run
re-escalates exactly those entries with nothing deleted by hand (below).

**This half is now fixed too, and it no longer needs a rebuild.** As of
2026-08-19 a pose records `arbitrated` — the arbiter *ran and answered* —
which is a different fact from `source == "vlm"`, the arbiter *moved the
answer*. Since 2026-08-21 the flag is **four-state**
(`docs/tri-state-pass-2.md`):

| `source` | `arbitrated` | meaning |
|---|---|---|
| `vlm` | `true` | the arbiter moved the pose |
| ensemble | `true` | it ran and confirmed the ensemble |
| ensemble | `"rejected"` | the API judged the request; never ask again |
| ensemble | `false` | the escalation the margin asked for did not happen |
| any | **absent** | no claim — **reads as `false`** |

`false` covers a transient failure (`pose.VLMUnavailable`), a cancellation,
abandonment at abort, *and* the gate firing in a run with no arbiter — that
last one is the C3 marking, so an `off` or degraded run now says so on the
entry instead of leaving it silent. `"rejected"` is the permanent side and it
is the **enumerated** one: a non-auth 4xx (`pose.VLMRejected`) or a 200 whose
body states a block verdict (`pose.REJECTED_FINISH_REASONS`). Everything else
— an unknown exception type included — records `false` and is asked again,
because three passes each found a transient failure sitting on what used to
be an open-ended permanent fallthrough. Still no `POSE_CACHE_VERSION` bump:
`to_cache` writes the string through unchanged and older readers ignore an
`arbitrated` they do not understand exactly as they ignored `false`.

**Done since 2026-08-19: `pose_is_sufficient` reads it** — and since
2026-08-21 it reads *absence* too, against this run:
`pose_is_sufficient(entry, arbiter_available, margin_threshold)`. An entry
with no claim (or a `false`) is a miss only when this run can actually
escalate it **and** its margin is under this run's gate. Three things reviews
had to add before the flips were safe to ship, all worth knowing at rebuild
time:

* the miss applies to a **later run only** — the driver's re-route of a
  just-folded answer passes `settled=True` to `route`, because re-checking
  sufficiency in the same run turned every rate-limited call into an
  unbounded re-render/re-bill loop (review, 2026-08-20);
* the transient side of the split is `pose.VLMUnavailable`, not just
  `RateLimited`: network drops, HTTP 5xx, gcloud/ADC failures, auth
  (401/403/404) and CLI timeouts all record `false`, and `settle` records
  `false` for the in-flight calls it abandons at Ctrl-C. Only a request the
  API judged on its merits records `"rejected"`;
* `arbiter_available` — a run with no arbiter (`--pose-vlm off`, a degraded
  `auto`, or the tripped `Poser` breaker) must not re-render a marked entry
  it cannot escalate. It would re-resolve the pose with no gate and erase the
  marker, and **production runs `off`** (docs/tri-state-pass-2.md,
  2026-08-21).

**So this section's debt is payable without a rebuild — by one run.** Absence
reading as `false` re-opens the gated legacy population: the **first
arbiter-on run after 2026-08-21** re-renders and re-asks everything under the
gate that carries no marker, and afterwards every gated entry says explicitly
which of the four states it is in. That is the confirmed-vs-never-asked
ambiguity above (the 1243 on `embed-cache2`) closing for good, at the price of
one run rather than a rebuild.

**Budget it at full magnitude.** It is roughly the cold-enable bill the census
already names — **~1227 paid calls** on `embed-cache2` (39–44 % of ~2800
models; `src/pose.py`'s `MARGIN_THRESHOLD` comment, review U4), about $0.30
of API — but the money is the small half. Each of the ~1227 is a full
`PoseRenderTask` (3–28 s of child work), the arbiter tail is bounded by
`WINDOW = 3` at a 24 s mean call — **~2.7 h even with perfect overlap** — and
under `--save-renders` every pose the arbiter moves forces a redraw. **The
first arbiter-on run after this ships is an overnight job on `embed-cache2`**,
not a lunch break; plan it as one, and run it with the paced arbiter above.

**Run that backfill without `--skip-embed`.** Re-resolution replaces the whole
entry, so `front_view` is dropped: on the normal path `Done._score` recomputes
and merges it back, and under `--skip-embed` nothing does. The backfill
re-resolves the entire gated population, so `--skip-embed` would strip
`front_view` — a published API field — from all ~1227 of them (accepted edge,
docs/tri-state-pass-2.md, 2026-08-21).

**The marked set is stable only to ~1e-2.** Filament's draw-history dependence
(CLAUDE.md's hard constraint) moves an ensemble margin by that much between
runs, so a model whose margin sits within ~1e-2 of `MARGIN_THRESHOLD` flips
sides of the gate from one run to the next. A `false` marker whose re-resolved
margin clears the gate is therefore erased — the entry records absent, and
absent-above-the-gate is a hit forever after. That is the one marker-loss path
in the four-state machine, and it is accepted: the ensemble is now confident,
so settling it is arguably correct, and preserving the old marker would need
the Poser to read the store, which J6 forbids. Never quote "the marked
population" as an exact count across runs.

## Not on this list

Things that look like rebuild debt and are not:

* **The duplicate tie-break** (`OPEN_QUESTIONS.md`). Which of two identical
  models wins a tied query is a scoring decision; rebuilding changes nothing
  about it.
* **`read_binary_stl`'s guards.** They parse files, not caches.
* **Filament's draw-history dependence.** Not fixable by rebuilding — a
  rebuild produces one more sequence-dependent set of renders, no more
  canonical than the last.
* **The `--compile` numeric regime split.** A rebuild could standardise on one
  regime, but that is a choice about throughput and precision
  (`docs/learnings/2026-08-14-precision-and-compile.md`), not debt.
