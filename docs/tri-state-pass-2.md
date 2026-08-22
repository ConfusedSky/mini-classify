# Tri-state pass 2 — design (2026-08-21)

**Status: confirmed for implementation (Masa, 2026-08-21).** All three
review-2 judgment calls confirmed: the breaker's ≥ 60 s window rides with
N = 5; the bill's full magnitude (an overnight first arbiter-on run) is
accepted; both Accepted edges stand as accepted.

The second pass over the `arbitrated` retry contract. Designed against the
findings in `docs/reviews/2026-08-20-past-week.md` ("Not acted on"), one
opus design review, and three decisions made by Masa in session. This file
is the implementation spec; the review record will carry the outcome.

## Problems this fixes

1. **Three failure classes still record as permanent** (key absent →
   settled forever) after W2: gcloud `RuntimeError`s raised from inside
   `_ask_gemini` (the mid-run ADC-expiry case — startup probing in
   `resolve_pose_vlm` covers cold misconfiguration, and the token cache's
   1800 s TTL guarantees a collection run re-mints mid-run); HTTP
   401/403/404 (environment/entitlement, discovered only mid-run since the
   startup probe never makes a Vertex call); and a 200 whose body cannot
   be dug into (`src/pose.py:753` is unguarded — JSONDecodeError, KeyError,
   IndexError all escape to the generic permanent arm).
2. **The laundering bug** (defect in the shipped tri-state): an
   `arbitrated: false` marker written by an arbiter-on run reads as a miss
   in a later `--pose-vlm off` run, which re-renders the model, re-resolves
   it with no gate, and records the key absent — marker erased, entry
   permanently settled. Production runs off, so this destroys markers
   routinely.
3. **Foreclosure by arbiterless runs**: any run without an arbiter
   (explicit `off`, or `auto` silently degrading on a gcloud failure)
   records every fresh resolution with the key absent — permanently
   settled — even for models whose margin was under the gate. One such run
   forecloses arbitration for everything it resolves.
4. **The permanent-by-default rule leaks**: three passes found transient
   failures on the permanent side (W2's network/5xx/claude set; this
   pass's gcloud/auth/body set; the design review's
   `subprocess.TimeoutExpired` leaking through pass 2's own first draft) —
   because transient is an enumerated allowlist and permanent is the
   open-ended fallthrough.

## The schema (decided)

`arbitrated: true | false | "rejected"`, and **absence reads as `false`**.

* `true` — asked and answered, whether or not the answer moved the pose.
* `"rejected"` — asked, and the API judged the request on its merits
  (non-auth 4xx, or a coherent 200 refusal — the safety-block shape).
  Never re-ask; the ensemble answer stands permanently.
* `false` — the escalation the margin asked for did not happen: transient
  failure, cancellation, abandonment at abort, or the gate fired in a run
  with no arbiter. Ask when possible.
* absent — no claim. Read as `false`; the gate check (below) makes this
  precise, since an absent entry whose margin clears the gate is never
  touched.

Mixed bool/string is deliberate: a clean string enum would force load-time
mapping of every `true`/`false` written since 2026-08-19 for no semantic
gain. Readers compare explicitly (`in (True, "rejected")`), never by
truthiness.

**Serialization (review 2 blocker B1):** `Pose.to_cache` currently coerces
`bool(self.arbitrated)` (src/pose.py:129-130) — `bool("rejected") is
True`, which would collapse the schema to three states on disk while every
in-memory test passes. The coercion must pass strings through, the
annotation at src/pose.py:69 widens to `bool | str | None`, and the
round-trip test asserts the **on-disk** value
(`to_cache()["arbitrated"] == "rejected"` and back through `from_cache`),
not the `Pose`-level one. `json.dumps` serializes the string identically
in `Done.flush` and `save_pose_cache`, so byte-parity holds; `Pose` stays
picklable for the message types (I13).

**Confirmed cost (Masa, 2026-08-21): the one-time legacy re-ask bill.**
Absent-as-false re-opens the gated legacy population: the first arbiter-on
run after this ships re-renders and re-asks roughly the cold-enable budget
the census already names (~1227 calls; 39–44 % of ~2800 models,
embed-cache2 — `src/pose.py` MARGIN_THRESHOLD comment, review U4). This is
the already-documented "enable the arbiter cold" budget made uniform, and
it pays down `docs/cache-rebuild.md`'s ~1243-entry
confirmed-vs-never-asked ambiguity without a rebuild: afterwards every
gated entry carries an explicit marker and the population is reportable.

Stated at full magnitude (review 2, S7): the bill is not just the call
money. Each of the ~1227 is a full `PoseRenderTask` (3–28 s of child
work), the arbiter tail is bounded by WINDOW = 3 at a 24 s mean call —
~2.7 h even with perfect overlap — and under `--save-renders` every pose
the arbiter moves forces a redraw. **The first arbiter-on run after this
ships is an overnight job**, not a lunch break. Plan it as one.

## Changes

### C1 — `_fold` flips its default (src/poser.py)

New exception in `src/pose.py`: `class VLMRejected(RuntimeError)` — "the
API judged the request; a retry cannot succeed." The permanent set is
closed (HTTP status space is finite; a judged verdict can only arrive as a
non-auth 4xx or a coherent 200 refusal), so it is the side that gets
enumerated. `_fold`'s arms become:

* answer → `arbitrated=True` if `idx is not None`, else `False`
  (unparseable-twice stays retryable, unchanged);
* `CancelledError` → `False` (unchanged);
* `except pose.VLMRejected` → `"rejected"`;
* `except pose.VLMUnavailable` → `False` — this arm is **mandatory**, not
  cosmetic: it is where C5's breaker counts (the record is the same as the
  generic arm's, the counter is not);
* `except Exception` → `False` — the flipped default. Unknown failure
  types now retry loudly instead of pinning silently.

### C2 — `_ask_gemini` classification (src/pose.py)

* `gcloud_project`/`gcloud_token` **normalise internally**: wrap their
  `subprocess.run` in `except (subprocess.SubprocessError, OSError) → 
  RuntimeError` (the `_ask_claude` pattern), so a hung or missing gcloud
  raises the type the helpers already promise. Callers'
  contracts are unchanged (`resolve_pose_vlm` catches `Exception`;
  `eval/gemini_sheet_fill.py:69` is unguarded either way).
* `_ask_gemini` wraps both helper calls at the call site:
  `except RuntimeError → VLMUnavailable(f"gcloud: {e}")`, token minted
  once before the header dict. (`gcloud_project()` is unreachable from the
  pipeline — `resolve_pose_vlm` always populates `args.gemini_project` —
  so the live mid-run path is the token half; wrap both anyway.)
* HTTPError mapping: 429/503 → `RateLimited`; **401/403/404/408/409/425 →
  `VLMUnavailable`**; ≥500 → `VLMUnavailable`; any other 4xx →
  `VLMRejected`. (408/409/425 per review 2: a request timeout from an
  intermediary is transient by definition, and mapping it permanent is the
  leak this pass exists to end.) The `except HTTPError` clause must stay
  **above** `except (OSError, HTTPException)` — `HTTPError` subclasses
  `OSError`; `.read()`'s `IncompleteRead` is an `HTTPException` and lands
  transient correctly.
* Body handling, read split from parse — and **`"rejected"` is inferred
  from the API's stated verdict, never from a `KeyError`** (review 2
  blocker B2: a `finishReason: MAX_TOKENS` with no parts — the
  thinking-token exhaustion this repo has measured twice, src/pose.py's
  ollama `"think": False` note and eval/gemini_vlm.py's
  `thoughtsTokenCount` — is deterministic per model *config* and would
  otherwise pin the whole collection as "rejected"):

  ```python
  try:
      body = urlopen(req, timeout=300).read()
  except HTTPError ...            # mapping above, ordering as above
  except (OSError, http.client.HTTPException) → VLMUnavailable
  try:
      d = json.loads(body)
  except ValueError → VLMUnavailable(f"unparseable 200 body: {body[:200]!r}")
  cand = (d.get("candidates") or [{}])[0]
  reason = cand.get("finishReason") or (d.get("promptFeedback") or {}).get("blockReason")
  parts = (cand.get("content") or {}).get("parts")
  if not parts:
      if reason in REJECTED_FINISH_REASONS or "blockReason" in (d.get("promptFeedback") or {}):
          raise VLMRejected(f"blocked ({reason}): {body[:200]!r}")
      raise VLMUnavailable(f"200 with no answer (finishReason={reason!r}): {body[:200]!r}")
  return parse_tile_answer("".join(p.get("text", "") for p in parts), n_tiles)
  ```

  with `REJECTED_FINISH_REASONS = ("SAFETY", "BLOCKLIST",
  "PROHIBITED_CONTENT", "RECITATION", "SPII", "IMAGE_SAFETY")` — the
  permanent side stays the enumerated one all the way down; `MAX_TOKENS`,
  `OTHER` and a missing reason are transient. An unparseable body is
  transport damage; only a body that coherently states a block verdict is
  the API judging the request. **Pre-ship check — done 2026-08-21**
  (`eval/capture_vertex_verdicts.py`, two paid calls, raw bodies in
  eval/out/vertex-verdicts/): both real envelopes classify as designed.
  MAX_TOKENS arrives with `parts: [{"text": ""}]` — retryable via the
  unparseable lane, not the no-parts branch — and SAFETY arrives as
  content-without-parts with a candidate-level `finishReason`, firing
  `VLMRejected` through the enumeration. Shapes pinned in
  `tests/test_pose.py::test_the_captured_vertex_envelopes_classify_as_designed`;
  write-up in the 2026-08-21 learnings entry. `IMAGE_SAFETY` from an
  actually-unsafe image stays uncaptured, deliberately.
* `ask_vlm_up` gains no new arm for `VLMRejected` — the generic arm
  already retries once and re-raises under `raise_failures`; a judged
  rejection on attempt 1 rarely differs on attempt 2, but one immediate
  retry is cheap and keeps the loop untouched.

### C3 — gate-fired-no-call marking (src/poser.py)

In `on_tile_embeds`, mark whenever the gate fires and no call is made,
whatever the reason (no backend configured, `off` run, degraded run,
breaker tripped):

```python
gated = pose.needs_arbiter_margin(margin, self.cfg.margin_threshold)
if gated and self.can_arbitrate():
    record False; park; return None          # park-time record is False now
p = self._make_pose(*resolved, arbitrated=False if gated else None)
```

The park-time record becomes `False` (was `None`): every completion path
overwrites it, so W3's settle-record and the CancelledError re-record
become consequences of the default rather than special cases, and the
`drop()`-on-parked path (currently unreachable, defensively) is covered.

### C4 — availability-conditional, gate-aware miss (src/pose.py + route)

`pose_is_sufficient(entry, arbiter_available, margin_threshold)` — **no
defaults on the new parameters** (the repo's `Resolved.pose_changed`
precedent; a default silently un-pins the W1 regression test). Logic:

* `source == "vlm"` → hit (unchanged);
* `arbitrated in (True, "rejected")` → settled → hit iff margin recorded;
* `margin is None` → miss (geometry-only upgrade, unchanged);
* else (`false` or absent) → miss **iff** `arbiter_available and
  needs_arbiter_margin(entry margin, margin_threshold)` — no re-render in
  arbiterless runs (kills the laundering bug), no re-render for entries
  whose margin clears this run's gate (a threshold change cannot launder
  markers).

`route` gains `arbiter_available` (no default). The driver passes
`cfg.poser.can_arbitrate()` at both call sites; the resolved backend stays
owned by `VlmConfig` — no new `CacheContext` field, no message-shape
change. At the `settled=True` site the parameter is **dead by
construction** (`settled or ...` short-circuits before the sufficiency
check) — passed because no-default demands it, and stated here so nobody
concludes the breaker can flip a settled re-route into a re-render.
`settled=True` remains load-bearing and orthogonal: in an arbiter-on run
`_fold` writes `false` and the re-route would loop without it (W1). The
threshold comes from `ctx.args.up_margin` (verify the field rides in
`CacheContext.args` at implementation; it is the same value
`VlmConfig.margin_threshold` is built from).

**Complete caller-breakage list** (review 2, S1/S2 — the no-default
choice makes every one a loud `TypeError`/`AttributeError`, which is the
point):

* `src/driver.py:290, :433` (production) and `src/cache_checker.py:78`
  (the reader);
* `tests/test_cache_checker.py` — nine `route()` calls (:215, :246, :256,
  :257, :273, :274, :279, :288, :292), and `make_args` (:53-61) has no
  `up_margin` field — every auto-path case needs it;
* **the W1 pin is factually mis-stated by an earlier draft of this spec**:
  its fixture (`ENTRIES["siglip"]`, margin 0.61) is *above* the 0.45 gate,
  so under C4 the cold call stops returning `PoseRenderTask` even with
  `arbiter_available=True`. Add a below-gate entry (e.g. `"siglip-gated"`,
  margin 0.2) and point the pin at it — otherwise the test passes for the
  wrong reason and W1's loop is unpinned;
* `tests/test_done.py:350` (forced path — parameter dead there, call still
  breaks); `tests/test_driver.py:279` (`Rig._route` signature) and
  :178-220 (`FakePoser` needs `can_arbitrate()`);
* `tests/test_poser.py:356, :456` — one-arg `pose_is_sufficient` calls,
  and their fixtures carry margin 1.98 under ESCALATE's threshold 5.0, so
  they must pass `margin_threshold=5.0` or the assertions invert.

### C5 — run-level circuit breaker (src/poser.py)

`Poser.can_arbitrate()` — true when a backend is configured and the
breaker has not tripped. `_fold` counts **consecutive** `VLMUnavailable`
folds, RateLimited included. Trip rule: **N = 5 consecutive AND the
failures span ≥ 60 s** (one timestamp). The time window is review 2's
correction to the "quota exhaustion is typically run-long" premise: Vertex
quota is a *rate*, refilled per minute (`--arbiter-min-interval` exists
because the 2026-08-19 incident was a storm, not an outage), effective
concurrency is 3 (WINDOW, not --arbiter-workers), and VLM_BACKOFF makes
five consecutive rate-limited folds reachable in well under a minute — a
sub-minute storm must not zero out an overnight run's arbitration.

Reset/count rules per `_fold` exit (review 2, S4):

* an **answer** — `True` or unparseable-`False` — resets the counter (the
  API spoke; the environment is healthy);
* **`"rejected"`** resets (the API spoke — five safety-blocked models in a
  row must not disable arbitration);
* **`VLMUnavailable`** counts;
* **`CancelledError`** neither counts nor resets — it exits before
  classification, and the abort path's fold_done/settle must not trip or
  clear the breaker during shutdown;
* the generic arm (unknown exception) counts — it is unavailability as far
  as the breaker knows.

The counter and `can_arbitrate()` live on the Poser and are only ever
touched from the parent thread (`poll`/`fold_done`/`settle` and the
driver's routing all run there) — no lock, and the count is fold-ordered
by construction. On trip: one loud line, no further submissions; the
gate-fired-no-call rule marks `false` from then on, and `can_arbitrate()`
turning false stops route re-rendering the marked backlog for the rest of
the run (bounding the wasted re-renders to the ~N models already in
flight). Already-parked futures still fold normally. The breaker never
untrips within a run. **The run's closing output reports the count of
gate-fired-no-call `false` records** — the run's product is
reportability, and a breaker-tripped full-collection run must not be
indistinguishable from a healthy one in the log tail.

### C6 — the auto-failure prompt (classify_stls.py resolve_pose_vlm)

Decided (a)/(b): prompt on auto-probe failure, default **No**.

* tty (require `sys.stdin.isatty() and sys.stderr.isatty()`; prompt on
  **stderr** so a redirected-stdout run does not appear to hang; note a
  backgrounded job with tty stdin still SIGTTIN-stops — same hazard as
  `cache_root`'s prompt, accepted): print the failure and what continuing
  means — gated poses keep the ensemble answer, are marked
  `arbitrated: false`, and are revisited by the first arbiter-on run —
  then `continue without the arbiter? [y/N]`.
* non-tty: `SystemExit` naming the explicit choices: `--pose-vlm off`, or
  fix gcloud (`gcloud auth application-default login`). No new flag —
  existing options express both intents (decided).
* Explicit `--pose-vlm gemini` keeps its startup `SystemExit`; explicit
  `off` stays silent.
* Mechanics (review 2, N10): `input()` writes its prompt to **stdout**, so
  "prompt on stderr" means `print(..., file=sys.stderr)` then a bare
  `input()`. This deliberately deviates from the `cache_root` precedent
  (stdin-only isatty, prompt on stdout) — deviation, not precedent, and
  say so in the code comment.
* Testing: nothing may import `classify_stls.py` (CLAUDE.md), tests
  included — drive the prompt paths in a **subprocess** (non-tty branches
  asserted from exit code and stderr text; the interactive y/N branch via
  a pty harness if cheap, else asserted through the non-tty message and
  left for a manual check, stated in the test docstring).

## Accepted edges (named, not fixed)

* **Gate drift can erase a marker (review 2, S6)** — the one marker-loss
  path in the four-state machine: a `false` entry whose margin sits within
  the ~1e-2 render-nondeterminism band of the threshold re-resolves on an
  arbiter-on run to a margin *above* the gate, records absent, and is
  settled thereafter. Accepted: the ensemble is now confident, so
  foreclosure is arguably correct — and preserving the old marker would
  require the Poser to read the store, which J6 forbids. One sentence in
  the docs carries it.
* **Re-resolution wipes `front_view`, and `--skip-embed` never restores
  it (review 2, S8)** — pre-existing (`record_pose` replaces the entry;
  `_make_pose` never carries `front_view`), but C4 multiplies the exposure
  to the whole gated population, and `front_view` is a published API
  field. On the normal path `Done._score` recomputes and merges it back;
  under `--skip-embed` nothing does. Accepted with guidance rather than a
  merge-in-`record_pose` (which would need up-comparison subtlety):
  **run the arbiter backfill without `--skip-embed`** — one line here,
  in cache-rebuild.md's amended entry, and in the run's closing output if
  cheap.
* **A `POSE_CACHE_VERSION` bump wipes all four states** (review 2, N7) —
  `load_pose_cache` drops mismatched `v`, so a v5 re-bills the gated
  population and loses every `"rejected"`. A line goes in
  cache-rebuild.md beside the existing version-bump entry.
* **The claude backend can never write `"rejected"`** (review 2, N8) —
  `_ask_claude` raises only `VLMUnavailable` or falls to the flipped
  default, so a deterministic claude parse failure re-asks once per
  arbiter-on run forever. The filed attempt-counter question covers both
  backends; say so in its text.
* **Forced `--up-axis` runs are inert** — route skips the pose store and
  `Done.flush` writes no pose cache outside `auto`; no marker is read or
  lost (verified, review 2, N6).
* **Verified inert, no change needed** (review 2, N9):
  `migrate_cache_keys.py` copies entries verbatim and rewrites through
  `save_pose_cache`, so `"rejected"` survives re-keying;
  `Collection.pose_of`, `/status`, the REPL and the render child never
  read `arbitrated`; `EmbedRenderTask` stays picklable with a str field.

## Tests

* `test_gemini_maps_each_transport_failure_to_the_retry_split` grows:
  401/403/404 → `VLMUnavailable`; 400 → `VLMRejected`; gcloud raising
  (monkeypatch the helper) → `VLMUnavailable`; helpers normalising
  `TimeoutExpired`/`OSError` → `RuntimeError`; the body cases via a fake
  response object whose `.read()` returns crafted bytes, per B2's
  verdict-enumerated split: not-JSON → `VLMUnavailable`; a stated
  safety/blocklist verdict → `VLMRejected`;
  `{}`/empty-candidates/no-parts *without* a stated verdict →
  `VLMUnavailable`; parts-without-text → `None` (unchanged lane). (An
  earlier draft of this bullet predated B2 and mapped bare no-parts to
  `VLMRejected`; B2 is authoritative, and implementation resolved it that
  way — 2026-08-21.)
* `_fold` end-to-end (`test_poser.py`): `VLMRejected` → `"rejected"`;
  bare `RuntimeError`/`TypeError` → `False` (the flipped default — the
  new load-bearing assertion); existing True/False cases updated.
* Round-trip, four states **on disk** (B1):
  `to_cache()["arbitrated"] == "rejected"` and back through `from_cache`
  — the Pose-level assertion alone passes with the bool() coercion bug in
  place.
* Body verdicts (B2): safety/blocklist shapes → `VLMRejected`;
  `finishReason: MAX_TOKENS` with no parts, missing finishReason, empty
  `{}` → `VLMUnavailable`; parts-without-text → `None` (unchanged lane).
* C3: off-run and degraded-run fresh resolutions record `False` when
  gated, `None`→absent when not; park-time record is `False`.
* C4: absent + gated + arbiter-on → miss; absent + gated + arbiter-off →
  hit; `false` + margin above this run's threshold → hit; the W1 pin
  moved onto a **below-gate** fixture (S1: the current `ENTRIES["siglip"]`
  margin 0.61 clears the 0.45 gate, so `arbiter_available=True` alone
  leaves the pin asserting nothing) and passing `arbiter_available=True`
  on both calls.
* C5: five consecutive unavailable folds spanning ≥ 60 s trip the breaker
  (injected clock); four-then-an-answer resets; five inside the window do
  NOT trip; `"rejected"` and unparseable-`None` folds reset;
  `CancelledError` neither counts nor resets (the abort path must not
  trip or clear it); after the trip, gated models resolve marked, nothing
  new parks, `can_arbitrate()` drives route, and the closing output
  carries the marked count.
* C6: monkeypatched `isatty`/`input`: y continues degraded, n/default
  exits, non-tty exits with the message.
* Fixtures through the production writers, per CLAUDE.md.

## Docs to move with it

`to_cache`/`pose_is_sufficient` docstrings (the four-state contract);
`docs/cache-rebuild.md` — the three-population table becomes four states,
the un-arbitrated-poses debt entry is amended (paid down by the first
arbiter-on run after this ships, no rebuild needed — run it without
`--skip-embed`), and the version-bump entry gains the wipes-all-four-states
line; `docs/actor-refactor/interfaces.md` §route (signature);
`data_structures.md` — `Pose.arbitrated` **and** the line stating
"the function is single-arg: `pose_is_sufficient(entry)`", which C4
falsifies (review 2, N2); the `src/cache_checker.py` comment explaining
why `pose_is_sufficient` "no longer takes the availability flag" — it now
takes a different one, and the comment would read as a contradiction.
`OPEN_QUESTIONS.md`: review 2 (N1) verified neither question exists there
yet, so both are **new entries**, not amendments — one recording the
default-direction decision (flipped, this pass, with the
three-passes-three-leaks evidence), one filing the attempt-counter
question (bounding re-asks of a deterministic per-model failure the
breaker cannot catch — it trips only on *consecutive* failures — covering
both backends). One sentence somewhere durable on gate noise: margins
within ~1e-2 of the threshold flip sides between runs (Filament
draw-history nondeterminism), so "the marked set" is stable only up to
that band — and see Accepted edges for its marker-loss consequence.

## Out of scope (unchanged)

`eval/gemini_vlm.py`'s twin parsing gap (spike harness); negative token
caching (a failed mint re-spawns gcloud per call — the breaker now bounds
the wall-clock case that made this tempting); the attempt counter (filed).
