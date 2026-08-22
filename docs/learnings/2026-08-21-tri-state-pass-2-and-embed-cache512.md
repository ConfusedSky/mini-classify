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

Still owed: the pre-ship capture of one real blocked body and one
MAX_TOKENS body against this repo's model (one paid call, raw output to
eval/out/, write-up here) — until then `REJECTED_FINISH_REASONS` matches
the documented Vertex shapes, not a captured one.

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

**Census pending.** When the backfill completes, the four-state markers
make the arbiter's value measurable for the first time: `(source=vlm,
true)` = moved, `(ensemble, true)` = confirmed, `"rejected"` and `false`
tails. That split — how often the arbiter changes a pose, against the
43/44 labelled accuracy — is the number "is the arbiter worth it" has
always been missing, and it gets appended here when the run exits.
