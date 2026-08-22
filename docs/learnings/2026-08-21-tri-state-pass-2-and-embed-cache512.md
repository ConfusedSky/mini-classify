# Tri-state pass 2, and the new primary cache — 2026-08-21

Three threads in one day: the warm/reload race that would not die, the
second (and structural) pass over the arbiter retry contract, and the
collection moving onto a new embedding model. The first two are shipped and
green at 617 tests; the third is rendering as this is written.

## The warm/reload race needed three tries

Yesterday's W9 fix (a generation counter so a finishing warmup cannot revert
a mid-warm `/reload`) guarded the collection bind and nothing else. Masa
reproduced the residue: a reload that *failed* during the warmup window
never bumped the counter, so warm's tail erased the recorded failure —
`/status` back to `present: true, failure: null` while the drive was gone
(W11). The fix — bump on every state-writing reload, failure included —
then over-corrected: guarding warm's collection bind with the same counter
made a failed reload discard the only collection in existence, and the
server answered 503 to every query until someone reloaded again (W11a,
reproduced by the reviewing session). The landed rule: the *error* writes
are guarded by the generation alone; the *bind* also fills an empty slot,
whatever the generation says, because a failed reload binds nothing and
publishing into an empty slot reverts nothing. Three states (collection,
error, loaded_at), three different staleness rules — the lesson is that "a
generation counter" is not one decision but one per field it guards.

## The retry split flips its default

The pass started as one residue item — gcloud failures recording as
permanent — and ended as a rewrite of the contract, because the evidence
said the *shape* was wrong, not the instance. Three passes each found
transient failures pinned permanently: W2 found network drops, HTTP 5xx and
the whole claude backend on the permanent side; this pass found gcloud/ADC,
HTTP 401/403/404 and the undiggable 200 body; and the second review found
`subprocess.TimeoutExpired` leaking through this pass's *own first draft*.
An enumerated-transient / fallthrough-permanent rule leaks on every
reading, because "ways a network call can fail" is open-ended and "the API
judged this request" is not. So the enumeration swapped sides:
`pose.VLMRejected` (non-auth 4xx, or a 200 whose body *states* a block
verdict) is the closed permanent set, and everything unrecognised records
retryable — the cost of a misclassification is now a visible re-ask instead
of a silent forever-pin.

With it, the schema Masa proposed: `arbitrated: true | false | "rejected"`,
and **absence reads as `false`**. That erases the absent-vs-false split the
2026-08-19 work leaned on — deliberately, and at a price accepted with the
cache named: the first arbiter-on run re-asks the gated legacy population
(~1227 calls by the embed-cache2 census, each a full pose re-render — an
overnight run, not a lunch). In exchange, `cache-rebuild.md` §8's
confirmed-vs-never-asked ambiguity is paid down with cache continuity, the
laundering bug dies (`arbitrated: false` is a miss only in a run that can
act on it — an `off` run used to re-render the model and erase the marker),
arbiterless runs stop foreclosing what they resolve (the gate-fired-no-call
rule marks instead), and a breaker (5 consecutive unavailable folds
spanning ≥ 60 s — quota is a per-minute *rate*, a sub-minute storm is not
an outage) keeps a broken environment from re-paying the whole backlog.

Two review catches worth remembering as a class:

* **`bool("rejected") is True`.** `to_cache` coerced the flag, so the
  four-state schema would have collapsed to three *on disk* while every
  in-memory test passed. The round-trip test now asserts the serialized
  value. A schema decision is not made until a test reads it back off the
  bytes.
* **A missing key is not a verdict.** Inferring "the API judged this" from
  a `KeyError` on `candidates` also captured `finishReason: MAX_TOKENS` —
  the thinking-token exhaustion this repo has measured twice — which is
  deterministic per model *config* and would have pinned the whole
  collection as "rejected" while looking like a successful arbitration
  pass. Rejection is now read from the API's stated `finishReason`/
  `blockReason`, enumerated like the rest of the permanent side.

The pre-ship capture ran the same evening
(`eval/capture_vertex_verdicts.py`, two paid calls, raw bodies in
`eval/out/vertex-verdicts/`, shapes pinned in
`tests/test_pose.py::test_the_captured_vertex_envelopes_classify_as_designed`).
Both real envelopes classify as designed, with one surprise: **MAX_TOKENS
from gemini-3.5-flash arrives with `parts: [{"text": ""}]`** — a non-empty
parts list holding one empty text, not the documented no-parts husk — so it
reaches the retryable record through the unparseable lane rather than the
no-parts branch (the husk shape stays pinned transient by the earlier
test; both roads lead to retryable). The SAFETY block arrives as `content`
with **no `parts` key** and a candidate-level `finishReason: "SAFETY"` —
no `promptFeedback.blockReason` at all — and `VLMRejected` fires through
the enumerated branch: the rejection arm's first exercise against a real
body, since the backfill hit it zero times. Uncaptured and stated:
`IMAGE_SAFETY` from an actually-unsafe image, which will not be
manufactured.

## embed-cache512 is the primary cache now

`google/siglip2-so400m-patch16-512` at render_size 512, **16 views**
(8 azimuths × elevations 20/-20), jpg renders — against embed-cache2's
SigLIP-1 recipe. Masa reports search quality under it is noticeably
better; no measured retrieval comparison yet. The pose cache was carried
over from embed-cache2 (3540 entries, 214 `vlm`, every `arbitrated`
absent), so the full classify running today (`--cache-dir embed-cache512
--rescan`) is simultaneously the SigLIP-2 re-embed and the four-state
backfill: with absence reading as `false`, every gated entry escalates —
100% of this run's arbiter calls are escalations by construction, and the
observed ~5 s/it (Masa, mid-run — treat as provisional, the run saturates
the disk and the 4060) puts the whole pass near five hours, comfortably
inside the WINDOW=3 arbiter-tail bound.

None of embed-cache2's operational figures transfer — warmup, query
latency, the ~206 MB matrix copy, the ~1227 census are all
measured-against-embed-cache2 and the matrix is now 16-view SigLIP-2.
Re-measure before quoting.

## The census: the arbiter moves the pose 62% of the time it is asked

The run exited cleanly the same evening (results.csv: **zero render
errors**; pose cache and CSV stamped 21:18). First census over the
four-state markers, embed-cache512, 3540 entries:

| population | count |
|---|---|
| moved — `(vlm, true)` | **651** |
| confirmed — `(ensemble, true)` | **397** (247 siglip + 150 geometry) |
| `"rejected"` | 0 |
| still `false` | 0 |
| legacy `(vlm, absent)` — never re-asked, by design | 214 |
| ungated `(ensemble, absent)` | 2224 |

**1048 arbitrations completed, and the arbiter moved the pose on 651 of
them — a 62.1% move rate.** That is the number this whole design was
missing, and it settles "is it worth it" emphatically: on the population
the margin gate escalates, the ensemble's answer is wrong (by the
arbiter's 43/44-accurate judgment) more often than it is right. Those 651
models — 18% of the collection — were embedded from sideways or
upside-down renders in embed-cache2, and every one re-keyed and
re-rendered under the corrected pose here. Some unmeasurable share of
"searching under embed-cache512 feels a lot more accurate" is this, not
the model upgrade; the two shipped together and cannot be separated after
the fact.

Zero `"rejected"` and zero residual `false` also matter: no safety blocks
on this collection's renders (the `REJECTED_FINISH_REASONS` arm went
unexercised in production — the pre-ship capture is still owed), no
transient failures, breaker never armed past zero. The paced pool
(workers 4, min-interval 1.0) took ~1048 calls through Vertex without a
single 429 recorded — the 2026-08-19 storm's fix holding at full scale.

Reconciliation of the gated leftovers, because an unlabelled residue
becomes next month's mystery: 54 gated entries carry no marker — 39 are
orphans (files the walk no longer sees), 11 are stale identities (the
file was edited; its live identity got arbitrated), and **4 are
unexplained** (present in the walk, gated cached margin, no marker, no
render error). 4 of 3540 is noise-level; worth a look only if the next
run grows it.

## Phase 3 re-measured against embed-cache512

Same methodology as the 2026-08-19 run (12 distinct sequential queries,
localhost, GPU idle at 33 MiB before start), same-day caveat named: the
backfill wrote this cache hours earlier, so the page cache was warm.
`embed-cache512`, 3380 models × 16 views × 1152 (~249 MB matrix),
`missing: 0` — full coverage, a first.

| | embed-cache2 (2026-08-19) | embed-cache512 (2026-08-21) |
|---|---|---|
| models loaded | 2801 | **3380** |
| first `/status` | immediate | 0.32 s |
| time to ready | 16.0 s | **13.6 s** |
| first query | 202 ms | 193 ms |
| `/query` median | 49 ms | **27 ms** (min 26, max 31, n=12) |
| `/similar` | 50 ms | 66 ms |
| server resident | ~2370 MiB | 2361 MiB |

The headline is the median **nearly halving against a 21 % larger
matrix**: that is the unscoped-`/query` fix landing (review, 2026-08-20 —
the full-collection scope no longer fancy-index-copies the matrix per
request, and at this matrix size the copy alone would now cost ~249 MB per
query). `/similar` tells the same story from the other side: it still
slices per request (its rows genuinely exclude the target), which is why
it now costs *more* than an unscoped `/query` where on embed-cache2 the
two were even. If `/similar` latency ever matters, that slice is the
known place to look. VRAM is unchanged within noise, so the two-instances
concurrency answer from 2026-08-19 stands for SigLIP-2 as well.
