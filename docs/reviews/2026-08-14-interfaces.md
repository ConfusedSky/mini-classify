# Review of `docs/actor-refactor/interfaces.md`

Review note, 2026-08-14. Covers the design note as written in `de5c7fe`, read
against `classify_stls.py`, `pose.py`, `eval/overlap_spike.py`, and the two
companion notes it binds itself to.

Method: the note fixes **calling conventions**, so it was read as a protocol —
each signature traced to the code path it replaces, and the driver pseudocode
walked against the four run modes the flags actually produce (cold, warm,
redraw, `--skip-embed`). Shapes were not relitigated; that is
[the data-structures review](2026-08-14-data-structures.md)'s six passes, and
where a finding here touches a shape it says which of those passes it lands on.

Findings carry IDs (`I1`…`I16`). Cite them in commits and in the note's own
revisions.

## Verdict

**The module map and the seams are right; the protocol does not terminate.**
The decomposition is the one the spikes earned, every seam is a
message-shaped function boundary as claimed, and the per-module conventions are
faithful to what the code does today.

What fails is the part a calling-convention note exists to get right. Three of
the four run modes hang or lose work:

* the end-of-input sentinel is sent **before** the arbiter tail drains, so the
  ~20% of files that escalate cannot be finished (`I1`);
* both boundary queues are bounded, which closes the render cycle the overlap
  spike deliberately left open — a hard deadlock, reachable on the redraw path
  (`I2`);
* `--skip-embed` files retire through nothing (`I3`).

Each is a two-line fix; none is a design change. But `I1` and `I3` both break
`while admission.admitted > admission.retired`, which is the note's own
Invariant 1 — the loop spins forever rather than failing, so they will be found
at 3 a.m. on a full run, not at implementation time.

Below that, one import rule contradicts itself in its own table (`I8`), one
convention is asserted as settled when it is the Spike-3 gate that was never
measured (`I11`), and the parent has no error boundary at all (`I4`).

Two questions gate the fixes (§7).

## 1. The protocol does not terminate

### I1. The sentinel is sent before the arbiter tail — HIGH

```python
tasks.send(None)                          # end of input
while admission.admitted > admission.retired:
    drain(block=True)
```

`drain` ends with `for task in poser.poll(): tasks.send(task)`. So any file
parked on an arbiter call when the walker runs dry resolves *after* the
sentinel, and its `EmbedRenderTask` is pushed onto a queue whose reader has
already read `None` and left. No `EmbedViews` comes back, `done.on` is never
called, `retired` never catches `admitted`, and the loop never exits.

This is not an edge case. It is the deferral path — ~20% of files — and it is
the entire reason today's code has a second pass *after* the main loop
(`classify_stls.py:1222-1236`, `print(f"resolving {len(deferred)} deferred
arbiter calls")`). `data_structures.md` §Supervisor accounting states the
premise correctly and this pseudocode contradicts it in the next document: *"a
file can still be parked on an arbiter call long after the input is
exhausted."*

Fix: quiescence first, sentinel second.

```python
while admission.admitted > admission.retired:
    drain(block=True)
tasks.send(None)
child.join(timeout); done.flush()
```

The child blocks in `recv` through the tail, which costs nothing. Note this
also reorders `done.flush()` past the join — worth deciding deliberately
(the abort path flushes before the join on purpose; the drain path need not).

### I2. Bounding both queues closes the cycle — HIGH

> Two transports, both `mp.Queue` at depth = the admission window

The measured version does not do this:

```python
out_q  = ctx.Queue(maxsize=args.queue_depth)
pose_q = ctx.Queue()             # back-edge: never bounded (actor rule)
```
(`eval/overlap_spike.py:132-133`)

`pose_q` is the parent→child direction — the note's `tasks` — and the spike
left it unbounded *citing the proposal's own deadlock rule*. Bound both and the
`Renderer → Poser → Renderer` cycle closes: the child blocks in `results.send`
with the queue full, the parent blocks in `tasks.send` with its queue full, and
the parent is therefore not draining `results`. Nothing breaks it; Ctrl-C is
the only exit.

The note answers this in advance, and the answer does not hold:

> `tasks.send` blocking is the *secondary* guard — the Admission window keeps
> in-flight work below the depth, so a blocked send means a bug in the window.

That is exactly the case pass 2 of the data-structures review already carved
out as `R7`: on the `needs_embed=False` path a file retires at `Done` via
`CachedHit` **while its render work is still queued**, so admission stops
bounding the child's backlog and "the bounded task queue, not the admission
counter, is what bounds the child" — by *blocking the sender*. A warm run over
a collection whose renders were cleared is a run of nothing but that path. The
window is not buggy there; it is correctly reporting zero in flight while the
queue is full.

Two things to reconcile, and they are the same thing:

* **Who owns admission?** The spike's child owns it (`while len(resident) >=
  inflight: finish_one()`, `eval/overlap_spike.py:107-108`) and the parent is a
  pure responder that never blocks — `pose_q.put` into an unbounded queue is
  the last thing it does per message. The note inverts this: the parent owns
  admission and the child is the responder. The inversion is defensible (the
  parent is where `route` and `Done` live) but it removes the structural
  property that made the measured version deadlock-free, and the note does not
  mention that it is changing it.
* **The `in_flight` never-evict rule inherits the same crack.** `ResidentMesh.
  in_flight` is exempt from eviction and the hard bound is "admission window ×
  heaviest mesh" — which is a bound only while admission tracks what the child
  holds.

Cheapest fix consistent with the measurement: leave `tasks` unbounded, keep
`results` bounded, and let admission be the only pressure — which is what
`data_structures.md` §Queues already says the rule is.

### I3. `--skip-embed` files retire through nothing — HIGH

Invariant 1 says every admitted index retires via `Embedded`, `CachedHit`, or
`Failure`. Under `--skip-embed` a file produces none of the three:

* pose resolution still runs — that is the flag's whole purpose
  (`classify_stls.py:925-927`: *"skip embedding and scoring the classification
  views; pose resolution, including the SigLIP up-ensemble, still runs"*);
* `need_embeds` is forced False (`classify_stls.py:1155`) and scoring is
  skipped (`:1197`), so there is no row and no `.npy`;
* so `route` must return something, the file goes `PoseRenderTask` →
  `PoseTiles` → `EmbedRenderTask(needs_embed=False)`, and per Invariant 4 the
  child sends **zero** results for that task.

`admitted` incremented, nothing retires it. Same hang as `I1`, on the mode used
to build a pose cache without paying for embeddings.

A `CachedHit` cannot stand in — `Done` loads the `.npy` and scores it, which is
the work the flag exists to skip. Either the union gains a retirement-only
message (`Retired(file, index)`, `Done` counts it and writes no row), or
`--skip-embed` short-circuits at the Poser and the note says which. The second
gating question in §7.

Related and already correct: `rows` legitimately has holes here — pass 1 of the
data-structures review settled that (`len(rows) == admitted` must never be
asserted). This finding is the counter, not the rows.

### I4. The parent has no error boundary — MEDIUM

The `Failure` convention is scoped to the child:

> **Every exception between `recv` and `send` becomes `Failure(file, index,
> str(e))`** — one bad mesh must not end the run

Nothing says the same for the parent, and the parent runs four things that
raise: `embedder.embed_tiles` / `embed_views` (CUDA OOM is the realistic one),
`poser.on_tiles` / `on_tile_embeds`, `poser.poll()`'s future folding, and
`done.on`'s `.npy` load. Today both are guarded — `rows.append({"top1":
f"RENDER_ERROR: {e}"})` appears at `classify_stls.py:1127` for the pose half
and `:1169` for the render/embed half — so this is a regression, not an
omission of something new.

Worse, it fails in two directions. Unguarded, one bad file ends the run and
takes the un-flushed pose cache with it. Guarded naively (a `try` around the
drain body), the file is admitted and never retired — `I1`'s hang again.

The `drain` loop is the natural boundary: wrap each `match` arm, convert to
`Failure`, hand it straight to `done.on`. One paragraph, and it makes
Invariant 1 true rather than aspirational.

## 2. The Transport protocol

### I5. `None` means two things, and the child cannot tell them apart — MEDIUM

```python
def recv(self, timeout: float):   # a message, or None on timeout
```
against
> `None` on the tasks queue is end-of-input

The child reads `tasks`. A `None` from `recv` is either "nothing arrived within
`timeout`" or "stop" — and the protocol offers no untimed blocking receive, so
the child has no way to avoid the ambiguity. It would exit on its first idle
window.

The sentinel choice is defended on the wrong axis: the note argues `None` is
safe *here* because the parent owns admission, which answers "is a poison pill
premature" and not "is the value distinguishable". Two fixes, either fine: a
frozen `EndOfInput()` message in `messages.py` (consistent with "no `kind`
field, routing is `match`"), or `timeout: float | None = None` plus a
module-level sentinel object. Prefer the message — it survives the shm variant,
where `recv` returns `(block_id, shapes)` and `None` is even less distinctive.

Note the spike never exercised this: its sentinel travels child→parent
(`eval/overlap_spike.py:125`, read at `:143-145`), and the parent→child queue
has no sentinel at all because the child's own loop ends it.

### I6. `close()` is unspecified, and the abort order can still hang — MEDIUM

`Transport.close()` appears in the protocol and is never mentioned again — not
in the drain path, not in the abort order, which is the one place it would
matter.

The concrete risk is the same class of bug the arbiter shutdown already handles.
`mp.Queue` has a background feeder thread, and it is joined at interpreter exit;
a `tasks` queue holding unflushed pickles when the driver aborts will hold the
process open even though the child is a daemon. The abort order fixes four
steps in a specific sequence precisely so Ctrl-C is fast, then omits the step
that keeps it fast. `cancel_join_thread()` on `tasks` belongs in that list, and
`close()` needs a stated meaning (flush-and-close, or drop).

### I7. `cancel_futures` does not skip the 24 s tail — MEDIUM

```python
def shutdown(self) -> None        # cancel_futures, never joins the 24s tail
```

`cancel_futures=True` cancels futures that have not started. Calls **already
running** are not cancellable, and `concurrent.futures`' atexit hook joins the
pool's non-daemon threads regardless of `wait=False`. With `--arbiter-workers`
N, the worst case at Ctrl-C is one full in-flight call per worker, ~24 s each,
in parallel.

Today's comment is careful about exactly this distinction and the note drops it
(`classify_stls.py:1238-1243`): *"Drop **queued** arbiter calls rather than
letting the interpreter join them at exit."* Restore the qualifier, and state
the residual wait — it is the difference between "Ctrl-C is instant" and
"Ctrl-C takes up to 24 s", which is worth knowing before someone builds a
second-Ctrl-C path on the assumption.

## 3. Module map and import rules

### I8. The import table lets `messages` import torch — MEDIUM

Two rows of the same table:

| module | may import | must NOT import |
|---|---|---|
| child side (`loader`, `renderer`, `render_child`) | … `messages`, pose | **torch** |
| `messages` | pose, numpy, **torch (annotations)** | everything else |

The child imports `messages` to unpickle its tasks. If `messages.py` has
`import torch` at module scope, the child has torch — the exact cost the first
row exists to prevent, and the rule is load-bearing enough that the note gives
it a "because" column.

Only two of the seven message types need the name: `TileEmbeds.embeds` and
`Embedded.embeds`, both of which live entirely in the parent and are never
pickled (`data_structures.md` says so explicitly). So the fix is free —
`from __future__ import annotations` plus `if TYPE_CHECKING: import torch`, or
split the two parent-only messages into their own module. Say which; "torch
(annotations)" is a hint that a reader will implement as an import.

## 4. Ownership the map does not assign

### I9. The canonical pose dict has no home — MEDIUM

Four modules touch it and none owns it:

* `poser.on_tile_embeds` "writes the resolved `Pose` into the canonical pose
  dict";
* `done.on` writes `front_view` through it with `replace`;
* `done.flush` serialises it;
* `route` reads it through `CacheContext` — and must see *this run's*
  resolutions, or a file that resolved at index 40 and is redrawn later
  re-resolves.

Invariant 3 lists it as two-writer and then says "nothing writes another
module's state", which cannot both be true. It is the single piece of shared
mutable state in the design and deserves the same treatment the note gives the
scene and the rows: name the holder (`Done` is the natural one — it already
flushes it and owns durability), give the Poser a one-line write API rather
than a dict reference, and note that `CacheContext` holds the same object, not
a copy.

This matters more than it reads: `save_pose_cache` is still a bare `write_text`
(`pose.py:150`), the one atomicity defect the proposal has left open, and it is
`Done.flush` that is supposed to fix it. Ambiguous ownership is how that gets
missed a second time.

### I10. `Done` increments a counter the driver owns — LOW

"the driver … owns Admission", and `Done.on` "increments `Admission.retired`".
Same shape as `I9`, much smaller. Either hand `Done` an `Admission` and call it
the deliberate exception, or have `drain` retire on the way out. One sentence
either way; it is only worth raising because Invariant 3 is stated absolutely.

## 5. A convention asserted past its measurement

### I11. The camera-rotation rule is the unmeasured Spike-3 gate — MEDIUM

> `EmbedRenderTask` → `renderer.views(lm, pose.up)` — up-rotation in the camera
> (`R.T`), never `mesh.rotate`, or residency is worthless.

The rule is almost certainly right, and it is stated as settled fact. It is not:
`actors_proposal.md` §Spike 3 lists it first among what is unmeasured —
*"Whether the rotation moves to the camera cleanly. Everything else here is
worthless if it does not"* — and the spike that produced the residency numbers
this design rests on does the opposite:

```python
mesh = resident.pop(i)
mesh.rotate(rotation_to_z_up(np.array(up, dtype=float)), center=(0, 0, 0))
```
(`eval/overlap_spike.py:101-103`)

So "three resident meshes at 88% busy" measured **holding decoded meshes in a
dict across the round trip**, not re-showing hidden scene geometry — the
write-up says so plainly (*"the child then rotates and renders the views"*,
`docs/learnings/2026-08-13-roundtrip-tiles-and-the-full-label-set.md:16`).
`render_up_candidate_grid`'s `R.T` is proven for the tile grid at one
elevation; the classification views are 8 azimuths × 2 elevations through
`view_angles`, and no measurement covers them.

Nothing to change in the design — keep the rule, it is what makes `ResidentMesh`
worth having. Change the framing to a precondition with a citation, so the
implementor knows a pixel-identity check against `mesh.rotate` is part of
building `renderer.views` and not a paranoid extra. Two candidates for the
verification already exist: `eval/render_determinism.py` and the
pixel-identical check `render_up_candidate_grid` was originally validated with.

This is the interfaces-note half of a claim `data_structures.md` §Renderer-child
mesh residency also makes; correct both in one pass or the next reader re-finds
it.

## 6. Smaller

### I12. `ctx.get_context("spawn")` — LOW

`mp.get_context("spawn")` (`eval/overlap_spike.py:131`). `ctx` is the result,
not the receiver.

### I13. Two types are introduced with no shape — LOW

The preamble says *"Types named here are data_structures.md's"*. `RenderConfig`
and `CacheContext` are neither there nor here — `RenderConfig` gets a
parenthetical field list, `CacheContext` gets a sentence. `RenderConfig` in
particular crosses a `spawn` boundary and so has a picklability constraint worth
one line. Either add both to `data_structures.md` or drop the preamble's claim.

### I14. `route` returns one value and the driver re-derives the second — LOW

> the driver derives the accompanying `CachedHit` from the same `ctx` lookup
> rather than `route` returning two values

That puts the decision in two places, which is what §Cache Checker's
"trivially unit-testable" claim rests on avoiding — a test of `route` no longer
covers what the driver does with the redraw path. Returning
`tuple[EmbedRenderTask, CachedHit]`, or a `Redrawn(task, hit)` case in the
union, keeps the decision table whole and costs nothing.

### I15. `abandon()` returns work the caller discards — LOW

`def abandon(self) -> list[EmbedRenderTask]` on the abort path, where the note
then says there is "nothing to do but drop". Make it `-> None`; a return value
that must be ignored is an invitation.

### I16. Pseudocode nits — LOW

* The wiring diagram feeds `done.on(msg)` from an `EmbedRenderTask` arrow
  (line 57); `done.on` takes `CachedHit | Embedded | Failure`.
* `for m in results:` needs `Transport` to be iterable, which the protocol does
  not offer. Spell it as the `recv_nowait` loop the prose describes.
* `drain` routes new tasks two ways for the same type — `dispatch(...)` in the
  `PoseTiles` arm, `tasks.send(task)` in the `poll` loop. Pick one.

## 7. Questions that gate fixes

**Q1. Does the parent or the child own admission?** The note says parent; the
measurement says child. Decides `I2` (whether `tasks` may be bounded at all)
and the `in_flight` eviction exemption's bound. Parent-owned is defensible —
but then `tasks` is unbounded and the note should say the spike's shape changed
and why.

**Q2. What retires a file that produces no row?** Decides `I3`, and it is the
same question `--skip-embed` and a hypothetical future "renders only" mode both
ask. A `Retired` message is the smaller answer; short-circuiting at the Poser is
the cheaper one and puts a second retirement path in a module that currently has
none.

## 8. What checked out — do not re-verify

* **`spawn`, not `fork`** — correct and load-bearing with CUDA initialised in
  the parent, and it matches the spike (`eval/overlap_spike.py:131`).
* **`PoseTiles.geo_scores` crossing with the tiles** — required, and the reason
  given is right: `resolve_up` is `combine_up(geo_scores, sig)` and the parent
  holds no mesh. Correctly credited to this note in `data_structures.md`.
* **The child's `Failure(file, index, str(e))`** matches what today's error path
  carries (`classify_stls.py:1127`, `:1169`) and the file is genuinely needed to
  retire.
* **The `Future` as the Arbiter transport** — matches `ThreadPoolExecutor` at
  `classify_stls.py:1079` and `submit` at `:1135`, and Q1 of the previous
  review settled it.
* **Abort ordering: arbiter, then abandon, then flush, pose cache before rows** —
  matches today's nested `finally` chain and its stated reason (the pose cache
  is the artifact whose loss costs money, `classify_stls.py:1244-1252`).
* **The Poser owning the single `.float().cpu().numpy()`** and building the
  contact sheet with `Image.fromarray` — consistent with `data_structures.md`
  and with `make_contact_sheet` taking PIL images (`pose.py:326`).
* **`embed_tiles`/`embed_views` blocking as the v1 pacing** — torch releases the
  GIL, so the child renders on; this is the Spike-4 result applied correctly.
* **The import rules for `poser`, `embedder` and `pose.py`**, and the note that
  the parent transitively imports open3d through `pose.py` without ever creating
  a renderer.

## Suggested order

1. **`I1`, `I3`** — the two hangs. Pure pseudocode fixes; `I3` needs Q2 first.
2. **`I2`** — needs Q1. Until it is answered, do not write `maxsize` on the
   `tasks` queue.
3. **`I4`, `I5`, `I6`, `I7`** — the protocol's error and shutdown edges. All
   four are paragraphs, and all four are things the current code already gets
   right that the note would regress.
4. **`I8`, `I9`, `I10`** — ownership. `I9` before anyone implements
   `Done.flush`, since the open `save_pose_cache` atomicity fix rides on it.
5. **`I11`** — reframe, and schedule the pixel-identity check as part of
   building `renderer.views`. Touches `data_structures.md` too.
6. **`I12`–`I16`** — any time.

## Aside: two convention gaps found while checking citations

Not findings against the note, but they are what made verifying it slower than
it should have been, and CLAUDE.md says to fix these in place rather than log
them:

* **`CLAUDE.md`'s `docs/actor-refactor/` bullet does not list
  `interfaces.md`** — it names the other three. Fixed alongside this note.
* **`eval/README.md` has no row for `overlap_spike.py`**, nor for
  `renderer_gil.py` or `parser_gate.py`. `overlap_spike.py` is the most-cited
  spike in the entire refactor set — the 1.17–1.21×, the 94% busy, the 1.11×
  roundtrip, the `--inflight 3` residency answer, and (per `I2` and `I11`) two
  structural decisions this note inherits without knowing it. The convention
  exists precisely so the next reader finds the harness from the number.

---

# Pass 2 — 2026-08-14, against `d18254b`

Two commits: `f644e6a` recorded pass 1 (with the CLAUDE.md doc-map fix),
`d18254b` is the response — `interfaces.md` substantially rewritten,
`data_structures.md` gaining the driver-side shapes, and the `eval/README.md`
aside taken with it.

Same method as pass 1: the protocol walked through the four run modes rather
than read. New findings carry `J` IDs. This pass also proposes one change that
closes four of them at once (§P2.3), because they turn out to be one defect
seen from four sides.

## P2.0 Verdict

**All sixteen findings taken, the two gating questions answered well — and the
answer to Q1 removed the pipeline's only backpressure on the redraw path.**

The rewrite is materially better. `I1`'s ordering is right, `EndOfInput` is the
correct shape and is defended on the right axis now, the import rule states its
own consequence, and `I11` is reframed into a verification step in *both* notes
rather than softened. Q1 is answered with the argument I would have made.

What is left is one finding half-taken and three consequences of the Q1 answer
that were not followed through:

* `--skip-embed --save-renders` still hangs, and is now *unrepresentable* in
  `route`'s return type (`J1`) — `I3` covered one of its two paths.
* Retirement is not idempotent, and two paths now retire the same index twice
  (`J2`). One of them is created by the redraw carve-out; the other by the new
  error boundary.
* Unbounding `tasks` was correct, and it removed the throttle that the bounded
  queue was accidentally providing. A redraw-heavy warm run now reaches
  quiescence in seconds with the child holding the entire collection, and
  `child.join(timeout)` silently discards the remainder (`J4`).
* `data_structures.md` §Supervisor accounting still describes the bounded task
  queue as what bounds the child's backlog (`J5`).

`J1`, `J2` and `J4` are the same defect: **the `needs_embed=False` carve-out
decouples "the pipeline is finished with this file" from "`Done` has its row".**
§P2.3 proposes making the child's contract uniform, which closes all three and
`J5` with it.

## P2.1 Disposition of pass 1

| finding | status |
|---|---|
| I1 | taken — quiescence, then `EndOfInput`, then join. Flush-after-join taken as the deliberate call it was flagged as. See `J4` |
| I2 / Q1 | **answered and taken** — `tasks` unbounded, `results` bounded, parent never blocks on a send. Consequences not propagated: `J4`, `J5` |
| I3 / Q2 | **half taken** — `Retired` is the right shape and covers the no-renders paths only. See `J1` |
| I4 | taken inside `drain`; two of the four raising sites the note itself names are still outside the guard. See `J3` |
| I5 | taken — `EndOfInput` frozen message, and correctly noted as what survives the shm variant |
| I6 | taken — `close()` = `cancel_join_thread` + close, and it joins the abort order in the right position |
| I7 | taken — queued-vs-running restored, residual stated as ~24 s per worker in parallel |
| I8 | taken — `TYPE_CHECKING` + `from __future__ import annotations`, with the reason in the table |
| I9 | taken — `Done` owns the store, `record_pose` is the write API. The signature is uncallable from the Poser (`J6`) and ownership stays conventional (`J7`) |
| I10 | taken — `Admission` split per field, named as the exception in Invariant 3. `in_flight()` is used but not in the shape (`J5`) |
| I11 | taken in both notes, as a precondition with a named verification step. The `eval/README` row for `overlap_spike.py` carries it too — better than asked |
| I12–I16 | taken; `Redraw`, `RenderConfig` and `CacheContext` all have shapes, `abandon() -> None`, diagram and drain corrected |

## P2.2 New findings

### J1. `--skip-embed --save-renders` still hangs, and is now unrepresentable — HIGH

`Retired` covers the two paths where nothing is wanted. The flag combination
where *renders* are wanted and embeddings are not has both halves still open:

* **Cold.** `on_tile_embeds` returns `EmbedRenderTask(needs_embed=False)` — the
  note's own carve-out — the child sends nothing per Invariant 4, and nothing
  retires the index. `I3`'s hang, unchanged.
* **Warm.** `route` must send a render task *and* retire. The only return that
  does both is `Redraw`, whose second field is typed `hit: CachedHit` — and a
  `CachedHit` makes `Done` load the `.npy` and score it, which is exactly the
  work `--skip-embed` exists to skip, on a file whose `.npy` may not exist.
  There is no legal return value.

The combination is real in today's code: `need_renders` is computed
independently of `skip_embed` (`classify_stls.py:1155-1158`), the render runs,
and scoring is skipped at `:1197`. "Build the pose cache and the debug renders
without paying for embeddings" is the obvious reason to pass both.

Narrow fix: widen to `Redraw(task, hit: CachedHit | Retired)` and state the rule
the type is enforcing — **every `needs_embed=False` task is dispatched together
with its retirement token.** §P2.3 is the wider one.

### J2. Retirement is not idempotent, and two paths double-retire — HIGH

Invariant 1 says "exactly once". Nothing enforces it, and the rewrite added a
second way to violate it:

* **Redraw + child error.** Invariant 4 exempts `needs_embed=False` tasks from
  sending a result; the child's error convention is blanket — *"every exception
  between `recv` and `send` becomes `Failure(file, index, str(e))`"*. So a
  redraw task that fails in `loader.get` or `renderer.views` sends a `Failure`
  for a file the `Redraw`'s `CachedHit` retired seconds earlier. (Not the save
  half — `save_renders` swallows `OSError` by design, `classify_stls.py:374-384`
  — but the load and render halves raise.)
* **The new error boundary.** `except Exception: done.on(Failure(m.file,
  m.index, str(e)))` wraps arms that call `done.on` themselves. If `done.on`
  raises *after* incrementing `retired` — scoring, `front_view` resolution and
  the `.npy` write all sit after that point on a path the note describes as
  "retires exactly once" — the handler retires the same index again.

The consequence is worse than a bad count. `retired` overtaking `admitted` makes
`in_flight()` negative, so `while admission.in_flight() > 0` exits with files
still in flight, `EndOfInput` races them, and the run flushes a complete-looking
CSV. That is `I1`'s hang inverted: instead of never finishing, it finishes early
and silently.

Fix, independent of everything else: `Done` keeps `retired: set[int]` and
ignores a repeat. Invariant 1 becomes mechanical rather than a convention four
modules have to honour, and it is the same set `rows` is already keyed by.

### J3. The error boundary stops short of two of the four sites it names — MEDIUM

The note lists what raises: *"the embedder, the poser, future folding, the
`.npy` load"*. The `try` covers the first two. The other two are outside it:

* **`poser.poll()`** — the future-folding call — sits below the `while` loop in
  `drain`, unguarded. One `apply_arbiter` fold that raises ends the run.
* **`dispatch(route(f, index, ctx))`** in the walker loop is where
  `done.on(CachedHit)` does the `.npy` load, and `route` itself stats files that
  the walk cache may list but that have since vanished. Neither is inside any
  `try`. This is the most-executed path on a warm run, and it reaches the same
  `done.on` the drain arm carefully guards.

One correction to pass 1 while fixing this, because the note repeats my framing:
*"today both halves of the pipeline convert those to error rows"* is too strong.
There is no per-file guard around `process(f)` (`classify_stls.py:1219-1220`);
the two `RENDER_ERROR` sites cover mesh-load/render/embed only, so a corrupt
`.npy` at cache-load ends the run today as well. Guarding the walker loop is an
improvement to make, not a regression to avoid — the argument for it is
Invariant 1, not parity.

### J4. Nothing bounds the child's backlog now, and the join discards it — HIGH

Unbounding `tasks` was right, and the bounded queue was doing a second job
nobody costed: throttling the driver to the child's rate.

Walk a warm run with `--save-renders` newly enabled, or with `renders/` cleared
— every file returns `Redraw`:

1. `route` is a dict lookup; `done.on(hit)` retires immediately.
2. `in_flight()` never rises, so the `WINDOW` gate never engages.
3. The walker loop completes in seconds, having pushed ~1000 render tasks into
   an unbounded queue.
4. `while admission.in_flight() > 0` is already false. `EndOfInput` is sent.
5. `child.join(timeout)` — and the child has tens of minutes of rendering left.

Whatever the timeout is, the daemon child dies at parent exit with most of the
collection unrendered, and `done.flush()` writes a CSV that looks complete.
`data_structures.md` promises the opposite: *"the child's outstanding render
work is drained by the child join at shutdown."*

The drain-path join and the abort-path join now mean different things — abort
genuinely wants a timeout, drain must not have one. That alone is a fix. But it
converts a silent truncation into an unbounded wait with no progress reporting,
which is why §P2.3 is the better answer.

### J5. §Supervisor accounting was not updated with the Q1 answer — MEDIUM

`data_structures.md:415-421` still reads as it did before `tasks` was unbounded:

* *"the admission limit is the single knob. The child's task-queue depth and the
  residency exemption below both derive from it"* — the task-queue depth no
  longer exists. It is one window, **two** consumers now.
* *"on that path the bounded task queue, not the admission counter, is what
  bounds the child's backlog"* — the R7 nuance is now simply false, and its
  falseness is `J4`.

Two smaller ones in the same block: `retired  # incremented by Done: success OR
Failure` predates `CachedHit` and `Retired`; and `in_flight()`, which the driver
epilogue calls twice, is not on the `Admission` shape.

### J6. `record_pose(ident: str, pose)` cannot be called from the Poser — MEDIUM

The Poser is constructed with `(up_T, down_T, arbiter, record_pose, vlm_cfg)`
and its input is `PoseTiles(file, index, geo_scores, tiles)`. Computing `ident`
means `pose.file_identity(f, root)` (`pose.py:107`) — and the Poser has no
`root`, correctly: Invariant 2 keeps modules off `Path`-derived keys, and the
caches are named as the one exception.

`record_pose(file, index, pose)` and let `Done` derive the identity — it holds
`CacheContext.args`, which is where `root` lives. Same one-line API, callable.

### J7. `Done` "owns" a store it is handed — LOW

`Done.__init__(admission, text_embeds, cache_ctx)`, and `CacheContext.poses` is
*the* object. So the store is loaded by someone else (`load_pose_cache` in
`classify_stls.py` or the driver), stored in a structure `route` reads, and
"owned" by a module that receives it third-hand. That is fine as wiring and thin
as ownership — and `I9` exists because the still-open `save_pose_cache`
atomicity fix (`pose.py:150`) needs an unambiguous home. One sentence naming who
constructs it and hands it over closes it.

### J8. `while (m := ... )` is a truthiness test — LOW

`while (m := results.recv(SHORT) if block else results.recv_nowait()):` should
be `is not None`. Frozen dataclasses are truthy, so it is correct today and
breaks silently the first time a message grows `__len__` or `__bool__` — and
this note is a spec that will be implemented literally.

## P2.3 The change that closes J1, J2, J4 and J5

All three of the substantive findings trace to one carve-out:

> **The child sends exactly one result per task**, except `needs_embed=False`
> (zero)

That exception is what decouples "`Done` has the row" from "the pipeline is
finished with this file", and every consequence follows: a file retires while
its work is outstanding (`J4`), so admission stops bounding the child (`J5`), so
a later `Failure` for that same work double-retires (`J2`), and a path that
needs a render *and* a retirement has no way to express it (`J1`).

**Make the contract uniform: the child always sends exactly one result per
task.** `needs_embed=False` returns `Rendered(file, index)`; the file retires on
that, not on the `CachedHit`.

| finding | why it closes |
|---|---|
| `J4` | admission holds the slot until the child acks, so the walker loop throttles again and quiescence genuinely means the child is idle |
| `J2` (redraw half) | one retirement path per file — a child `Failure` replaces the ack rather than duplicating a retirement |
| `J1` | `--skip-embed --save-renders` retires on the ack like everything else; no `Redraw`-shaped special case, cold or warm |
| `J5` | "one window, three consumers" becomes true again, and R7 dissolves rather than needing a rewrite |
| Invariant 4 | loses its exception; the child's blanket `Failure` rule stops contradicting it |

The cost, stated honestly: `Done` must separate *writing the row* from
*retiring the index* on the redraw path — the `CachedHit` writes, the ack
retires. That is one flag on one message, against four findings and a carve-out
that has now generated defects in two consecutive passes.

Keep `J2`'s idempotent `retired: set[int]` regardless. It is what makes
Invariant 1 mechanical, and the error boundary needs it whatever the child's
contract is.

## P2.4 What checked out — do not re-verify

* **The `I1` ordering is correct**, including the subtle half: `poser.poll()`
  can produce tasks during the quiescence loop, and the sentinel now follows
  every one of them.
* **`tasks` unbounded / `results` bounded is the deadlock-free pair.** The
  parent's only blocking call is `results.recv(SHORT)`, which times out; the
  child blocks on `results.send` or `tasks.recv`. No cycle. `results` occupancy
  stays ≤ WINDOW *given* the `J2` fix — the failing-redraw path is the only
  thing that can exceed it.
* **`close()` on `tasks` only is right.** The feeder thread lives on the sending
  side, the parent sends only on `tasks`, and the child's own feeder dies with
  the daemon process.
* **`EndOfInput` surviving the shm variant** — correct, and a better reason than
  the one pass 1 gave.
* **The `I7` rewrite is accurate to `concurrent.futures`' semantics**, including
  that the second Ctrl-C is what bounds the residue.
* **The `I11` reframing**, in `interfaces.md`, `data_structures.md` and the new
  `eval/README.md` row alike — the row is the one that will actually be read
  before someone re-quotes the residency numbers.
* **The three new `eval/README.md` rows are accurate**: `overlap_spike.py`'s
  three modes are `baseline`/`overlap`/`roundtrip` (`:249`, `:276-277`),
  `parser_gate.py`'s A/A-control history matches its docstring, and
  `renderer_gil.py`'s GIL split matches Spike 4.

## P2.5 Suggested order

1. **`J2`'s idempotent retirement** — smallest, closes the silent-early-exit
   failure, and is needed under either answer to §P2.3.
2. **§P2.3** — decide it before `J1` and `J4`, since taking it closes both. If
   it is rejected, take `J1` as `Redraw(task, CachedHit | Retired)` and `J4` as
   an untimed drain-path join, and say in the note why the carve-out is worth
   the two special cases.
3. **`J5`** — follows whichever way §P2.3 goes; do not update
   §Supervisor accounting twice.
4. **`J3`, `J6`** — both are one line, and `J6` blocks anyone writing the Poser.
5. **`J7`, `J8`** — any time.

---

# Pass 3 — 2026-08-17, against `13b3e72`

Two commits: `f6ec298` recorded pass 2, `13b3e72` is the response — §P2.3 taken
in full, plus all eight `J` findings.

New findings carry `K` IDs. This pass follows the uniform contract into the one
place §P2.3 did not reach: the child's residency table.

## P3.0 Verdict

**§P2.3 taken in full, and it does what it was argued to do.** `Rendered` as the
ack, `CachedHit.retires`, admission bounding the child again, the untimed
drain-path join, `retired_ids`, three error boundaries instead of one — the four
findings that were one defect are closed as one change, and Invariant 4 lost its
exception rather than acquiring a second one. `results` occupancy is now provably
≤ WINDOW, and the residency hard bound (`admission × heaviest mesh`) is true
again rather than aspirational.

One consequence went unnoticed, and it is the mirror image of the last two
passes: they were about files that never retire, this is about **meshes that are
never released**. Nothing clears the child's `ResidentMesh.in_flight` flag on the
three paths where a `PoseTiles` is *not* followed by an `EmbedRenderTask` — and
`in_flight` entries are exempt from eviction, so they are pinned for the process
lifetime. Under `--skip-embed` that is every file in the collection (`K1`).

Below that: the child's clean exit now runs straight into this repo's
hardest-won constraint and the untimed join waits for it (`K2`); `dispatch`'s
routing table does not cover a value `poll` can now return (`K3`); and the
argument for unbounding `tasks` still rests on the R7 claim §P2.3 dissolved,
two bullets above the sentence that dissolves it (`K4`).

## P3.1 Disposition of pass 2

| finding | status |
|---|---|
| §P2.3 | **taken in full** — `Rendered` ack, `CachedHit.retires: bool`, Invariant 4 uniform, and the `results`-depth/residency bounds true again as a result |
| J1 | closed on both halves — cold retires on the ack, warm is a plain `needs_embed=False` task with no `Redraw` special case |
| J2 | taken — `retired_ids: set[int]`, and Invariant 1 now says "mechanical". Composes correctly with the new boundaries (a guarded `done.on` that raises post-retirement is now a no-op) |
| J3 | taken, and better than asked — the walker loop gets its own boundary *and* `poll` converts per-future fold failures rather than the driver guessing attribution. See `K3` |
| J4 | taken — untimed drain-path join, timeout kept for abort only. See `K2` |
| J5 | taken — "one window, two consumers", R7 dissolved, `in_flight()` on the shape, `retired`'s comment enumerates all five retiring messages. See `K4` for the one stale copy left in `interfaces.md` |
| J6 | taken — `record_pose(file, index, pose)` and `CacheContext.root`. See `K7` |
| J7 | taken — loaded by the CLI entry, handed to `Done` at construction, owned thereafter |
| J8 | taken |

## P3.2 New findings

### K1. Nothing releases a resident mesh when no embed render follows — HIGH

The child marks a mesh `in_flight` when it sends `PoseTiles` — *"awaiting a pose
answer — exempt from eviction"* — and the answer is what clears it. Three paths
never send one:

* **`Retired` from `on_tile_embeds`** — `--skip-embed` with no renders wanted.
  Pose resolution was the whole job, the file retires at `Done`, and no
  `EmbedRenderTask` is ever sent. **Under `--skip-embed` this is every file**, so
  the child accumulates the entire collection, all of it exempt from eviction,
  until it is killed. `budget_bytes` cannot reclaim any of it — the note already
  says the exemption makes it a soft bound.
* **`Failure` from `poser.poll()`** — new in this pass (`J3`). A fold that raises
  retires the file and pins its mesh.
* **A drain-arm exception on the pose path** — `embedder.embed_tiles` raising is
  the realistic one (CUDA OOM). The `except` retires via `Failure`; no task
  follows; the mesh is pinned. This is the most likely of the three on a normal
  run, and it pins a mesh on precisely the failure that suggests memory is
  already tight.

The design inherited an assumption its shape no longer guarantees. In the
roundtrip spike the child owned the loop and every held mesh was popped by
`finish_one()` (`eval/overlap_spike.py:100-101`) — there was no skip path, no
retirement that wasn't a render, and no way for a mesh to be abandoned. The
parent-owned admission this note chose (correctly, `I2`/Q1) makes abandonment
reachable, and `in_flight` has no other clear.

Fix, and it is small: a **`Release(file, index)` control message on `tasks`**,
sent by whatever retires a file that has an outstanding `PoseTiles`. The child
clears `in_flight` and drops the mesh to normal LRU eligibility.

Note the interaction with Invariant 4 and treat it explicitly rather than as an
exception: `Release` is a **control message, not a task** — the same category
`EndOfInput` already occupies, which is why Invariant 4 can stay "one result per
task, no exceptions" without qualification. Say that in the invariant, or the
next pass will read `Release` as the carve-out coming back.

`Done` is the natural sender (it is where all five retirements land and it
already knows the index), which means `Done` needs the `tasks` transport — or
the driver sends it from the one place retirement is observable. Either is fine;
pick one, because "whoever retires it" is how this got missed.

### K2. The child's clean exit runs into the teardown abort, and the join is now untimed — MEDIUM

`EndOfInput` terminates `run_child`, which returns, which lets the interpreter
tear down `src/renderer.py`'s `OffscreenRenderer`. That is the one thing this
repo has a hard constraint about: *"`OffscreenRenderer` teardown aborts; creation
does not … the abort is Filament throwing from a destructor. Keep renderers alive
for the process lifetime; never destroy one"* (CLAUDE.md, measured in
`docs/reviews/2026-08-13.md` §3.1).

So the child's **clean** exit is its dangerous one, and `J4` just made the parent
wait for it with no timeout. `join()` still returns — SIGABRT is process death —
but every successful run ends with a Filament abort on stderr and
`child.exitcode == -6`, which a reader will file as a bug in the refactor rather
than the known constraint.

Two lines close it: `run_child` ends with `os._exit(0)` after `EndOfInput`
(skipping interpreter teardown, which is exactly the "never destroy one" rule
expressed as code), and the note states that the child's renderers are never
destroyed — the constraint is currently absent from a note that assigns
renderer *creation* to a module and describes the child's whole lifecycle.

Worth stating alongside it: `child.exitcode` is not checked anywhere in the
driver. Under `os._exit(0)` it becomes meaningful, and a nonzero one is the only
signal that the child died with tasks outstanding.

### K3. `dispatch` does not route `Failure`, which `poll` now returns — MEDIUM

`poll() -> list[EmbedRenderTask | Retired | Failure]` (new, `J3`), and the driver
does `for out in poser.poll(): dispatch(out)`. `dispatch`'s stated table is
*"`CachedHit`/`Retired` → `done.on`; tasks → `tasks.send`; `Redraw(task, hit)` →
both."* No `Failure` arm.

The intent is obvious and the fix is one word, but `dispatch` is the routing
table the note holds up as the single place message types meet module calls, and
it is now incomplete for a value produced two lines above it.

### K4. The unbounded-`tasks` argument still rests on the dissolved R7 — MEDIUM

Two adjacent bullets in §The boundary protocol now disagree. The first still
argues:

> An earlier draft bounded both and called a blocked `tasks.send` "a bug in the
> window" — refuted by the `needs_embed=False` path (data-structures review R7),
> where files retire while their render work is still queued

The next one says the opposite, correctly:

> With the uniform child contract (§P2.3), the ack restores **admission** as what
> bounds the child's backlog — every task holds its slot until its result comes
> back

`data_structures.md` §Supervisor accounting was rewritten for this (`J5`); this
copy was not. And the correction has a consequence the note should own rather
than leave for someone to rediscover: **under the uniform contract, bounding
`tasks` at the window would no longer block.** Admission caps in-flight files at
WINDOW, each holds at most one outstanding task, and `EndOfInput` is sent when
the queue is empty. So unbounding is no longer load-bearing for deadlock — it is
defence in depth plus the threaded successor's back-edge rule, which is precisely
what the second bullet says ("only for deadlock-freedom").

Keep the decision. Re-point the argument, and delete the R7 sentence rather than
leaving the dissolved claim as the stated reason — this is the third pass in
which R7 has generated work.

### K5. Row-overwrite semantics on the redraw-failure path — LOW

On the redraw path `CachedHit(retires=False)` writes the row, then a failed
render sends `Failure` *instead of* the ack — and `Failure` "doubles as the error
row". So `rows[index]` is written twice and the note does not say which wins.

Checked against today, because the answer should be parity and is: a render
failure on the redraw path appends a `RENDER_ERROR` row and returns before
scoring (`classify_stls.py:1160-1170`), so the file reports `RENDER_ERROR` rather
than its cached score. "A later `Failure` overwrites the row" reproduces that.
One clause in `Done.on`.

### K6. Send-after-save is not stated — LOW

"Quiescence means the child is idle" (the basis for the untimed join) holds only
if the child sends its result *after* `save_renders` returns, not before. The
child section lists saving and sending without fixing the order, and overlapping
them is the obvious optimisation someone will reach for. One word: the ack is
sent after the save completes.

### K7. `root` now has three homes — LOW

`CacheContext.root` (new, `J6`), `CacheContext.args` (which carries the input
path), and `RenderConfig.collection_root`. The last is genuinely separate — it
crosses the spawn boundary — but two copies in one dataclass invite drift. Drop
`args`-derived root usage explicitly, or note that `root` is the only sanctioned
reader.

## P3.3 What checked out — do not re-verify

* **`results` occupancy is now provably ≤ WINDOW.** Every task returns exactly
  one result, admission caps in-flight files at WINDOW, and each holds at most
  one outstanding task. The bound in the note is now a theorem rather than a
  hope, and the failing-redraw case that broke it in pass 2 is gone.
* **The residency hard bound is true again** — `in_flight` exemption × admission
  window (~450 MB at 3 × 150 MB) holds now that admission bounds the child.
  Subject to `K1`, which breaks it a different way.
* **`retired_ids` composes with all three error boundaries.** A guarded `done.on`
  that raises after retiring is now a no-op on retry; a `Rendered` ack arriving
  after a `Failure` retired the file is ignored. Both were reachable in pass 2.
* **`Retired` vs `Rendered` are genuinely different messages**, not a rename:
  `Retired` is parent-originated (the Poser, or `route`), `Rendered` is the
  child's ack. Collapsing them would have re-coupled the two.
* **The `--skip-embed` decision table is complete now**: warm+nothing →
  `Retired`; warm+renders → `EmbedRenderTask(needs_embed=False)`; cold+nothing →
  Poser `Retired`; cold+renders → task, ack retires. All four reachable, all four
  retire once.
* **`in_flight()` on `Admission`**, `retired`'s five-message comment, and
  Invariant 1's "mechanical" wording are all accurate to the new shape.

## P3.4 Suggested order

1. **`K1`** — the only one that costs a run. `Release` as a control message, and
   say in Invariant 4 that control messages are not tasks.
2. **`K2`** — two lines, and it decides what a nonzero `child.exitcode` means
   before anyone builds on it.
3. **`K3`, `K4`** — corrections; `K4` should end R7's third appearance.
4. **`K5`, `K6`, `K7`** — any time.

---

# Pass 4 — 2026-08-17, against `e072d2e`

Two commits: `c1e3664` recorded pass 3, `e072d2e` is the response — all seven
`K` findings, `Release` as a control message, and the child's exit path rewritten.

New findings carry `L` IDs. This pass follows the one thing pass 3 asked for that
opened a new question: the child-death check that `K2` implied.

## P4.0 Verdict

**All seven taken, and `K1`'s fix is right down to the ordering argument.**
`Release` as a *control* message keeps Invariant 4 unqualified rather than
re-introducing a carve-out; the FIFO reasoning holds under every path I could
walk, including the redraw case where a task and a retirement race; and `Done`
holding the `tasks` transport is the correct answer to "who sends it", for the
reason the note gives — "whoever retires it" is spread over three paths, which is
how the leak hid. `K4` is taken with more honesty than asked for: the note now
says a bounded `tasks` would never block *and* why it stays unbounded anyway.

The finding is `K2`'s other half. Checking `child.exitcode` after the join is
right in principle, but it is placed where it cannot do the job the note assigns
it — *"the only signal the child died with tasks outstanding"* is unreachable
there, because reaching that line means nothing was outstanding. The failure it
names is real and still uncovered: **a child that dies mid-run hangs the
quiescence loop forever** (`L1`). And where the check does fire, it discards a
completed run's artifacts (`L2`).

Two small ones round it out: the wiring diagram has no `Release` edge now that
`Done` sends to the child (`L3`), and `os._exit(0)` skips stdio flushing (`L4`).

## P4.1 Disposition of pass 3

| finding | status |
|---|---|
| K1 | taken, and correctly — `Release` is a control message, `Done` is the sender, unconditional with a child-side no-op, FIFO ordering argued rather than assumed. `in_flight`'s two clears are documented in `data_structures.md` with all three leaking paths named |
| K2 | half taken — `os._exit(0)` and the never-destroy rule are right (see `L4` for one gap). The `exitcode` check is in the wrong place: `L1`, `L2` |
| K3 | taken — `dispatch` routes `Failure`, credited to `poll` |
| K4 | taken, and better than asked — the R7 sentence is deleted, the argument re-pointed at defence-in-depth, and "a queue that *cannot* block the parent survives future contract mistakes" is the right reason to keep it |
| K5 | taken — later `Failure` overwrites, parity stated |
| K6 | taken — ack strictly after `save_renders`, with the untimed join named as what rests on it |
| K7 | taken — `CacheContext.root` sanctioned, `RenderConfig`'s copy justified by the spawn boundary |

## P4.2 New findings

### L1. A child that dies mid-run hangs the quiescence loop — HIGH

The check says what it is for:

```python
    if child.exitcode != 0:                   # ... the only
        raise ChildDied(child.exitcode)       # signal the child died with tasks
                                              # outstanding (K2)
```

It cannot be. Control reaches that line only after
`while admission.in_flight() > 0: drain(block=True)` has exited, and that loop
exits only when every admitted index has retired — i.e. when **no task is
outstanding**. The stated purpose and the placement are mutually exclusive.

The failure it names is real, and it is the one hang class the previous three
passes did not close. If the child dies with tasks outstanding — the Filament
abort if `os._exit(0)` is not reached, an OOM kill on a 4M-triangle mesh, the
daemon reaped — those tasks never produce results, `in_flight()` never reaches
zero, and `drain(block=True)` spins on a `results` queue that will never receive
anything again. Forever, with no output: `done.flush()` is below the loop.

This is `I1`'s hang with a different cause, and it is the last one reachable:

| cause | closed by |
|---|---|
| arbiter tail after the sentinel | `I1` |
| a file that produces no row | `I3`, `J1` |
| double retirement (early exit) | `J2` |
| **the child dies mid-run** | **nothing** |

Fix: liveness belongs *inside* the loop, not after it. `drain(block=True)`
already wakes every `SHORT` on the recv timeout, so the check is free — if
`child.exitcode is not None` while `in_flight() > 0`, the child is gone and its
outstanding work cannot complete. Convert the outstanding indices to `Failure`
(the driver knows them: admitted-minus-retired, and `Done` has `retired_ids`),
which retires them through the one path Invariant 1 allows, and let the run flush
what it has. A collection where one mesh kills the child should lose one file's
row, not the whole run's pose cache.

That also gives the exitcode check a real job: after quiescence it becomes a
diagnostic about *how* the child ended, which is `L2`.

### L2. The raise discards a completed run's artifacts — MEDIUM

`raise ChildDied(...)` sits between the join and `done.flush()`. At that point
the run is *finished*: quiescence held, every file retired, every row collected,
every pose resolved — and the pose cache is the artifact this design repeatedly
singles out as the only one whose loss costs money (~$0.30 of Gemini calls per
re-resolve, per the proposal's shutdown section).

Worse, the only thing that can make the exitcode nonzero at that point is a child
that died *after* all its work was delivered — most plausibly the very Filament
teardown abort `K2` set out to prevent, if `os._exit(0)` is ever not reached. So
the raise throws away a complete, correct run over a cosmetic death in a process
that had nothing left to do.

The note's own rule already answers this: *"`flush` is idempotent and runs on the
main thread in a `finally`"*. If that outer `finally` exists, the pseudocode is
merely misleading; if the epilogue is read literally — and it will be, it is the
spec — the flush is skipped. Order it flush-then-report, and make it a warning
rather than an exception unless `L1`'s in-loop check has already recorded lost
work.

Minor, same place: `ChildDied` is introduced with no definition, in a note whose
preamble says types named here are `data_structures.md`'s.

### L3. The wiring diagram has no `Release` edge — LOW

`Done` now sends on `tasks`, which makes it the second parent-side writer to the
child and changes the shape the diagram exists to convey — the diagram still
shows `tasks` fed only from `route`/`dispatch`, and `done.on` as a pure sink at
the bottom of the parent box. The module map line is also now incomplete:
`src/done.py  scoring, rows, pose store, retirement, flush` — no mention that it
talks to the child.

Both are one edit. Worth doing because the diagram is what a reader checks
against before believing prose, and "`Done` is the only writer" now has a second
meaning it did not have three passes ago.

### L4. `os._exit(0)` skips stdio flushing — LOW

`os._exit` is the right call and it bypasses more than interpreter teardown: it
does not flush Python's buffered stdout/stderr. The child does print — the
`save_renders` failure path is a `print` on `OSError` (`classify_stls.py:383`),
deliberately non-fatal — and on a pipe those buffers are block-buffered, not
line-buffered. So the diagnostics for the last few files can vanish exactly when
someone is debugging why renders are missing.

`sys.stdout.flush(); sys.stderr.flush()` before `os._exit(0)`. One line, and it
is the same class of "the fast exit skips the delivery guarantee" that `I6`
covered for the queue feeder.

## P4.3 What checked out — do not re-verify

* **`Release`'s FIFO argument holds.** I walked every path where a retirement and
  a task for the same index could race. The only one that gets close is
  `Redraw` — but its hit carries `retires=False`, so no `Release` is emitted
  before the task, and the `Rendered` ack that follows arrives after the child
  has already processed the task. Retirement is terminal and idempotent, so no
  task for an index can follow its `Release`.
* **A redraw mesh is never marked `in_flight`**, so the unconditional `Release`
  on that path is a genuine no-op rather than an accidental early unpin — the
  flag means "awaiting a pose answer", and a redraw file's pose is already known.
* **`os._exit(0)` does not lose results, and the reason is `I1`.** `os._exit`
  skips the `mp.Queue` feeder's delivery guarantee, so an unflushed result would
  be lost — but `EndOfInput` is only sent after quiescence, by which time the
  parent has already received every result. The two fixes are load-bearing for
  each other; if anyone ever moves the sentinel back before the drain, it breaks
  `os._exit` as well as `I1`. Worth one sentence in the note so the coupling is
  explicit rather than incidental.
* **`Release` on a fully warm run** is ~1000 messages to a child with nothing
  else to do — free on an unbounded queue with a child-side no-op, and cheaper
  than tracking which files have an unanswered `PoseTiles`.
* **`Retired` and `Release` are correctly distinct**: one retires a file at
  `Done`, the other unpins a mesh in the child. On the `--skip-embed` cold path
  both are emitted for the same file, in that order, and that is right.
* **Invariant 4 stayed unqualified**, which was the point of making `Release` a
  control message rather than a task with no result.

## P4.4 Suggested order

1. **`L1`** — the last hang, and the in-loop check is where `K2`'s intent
   actually lands.
2. **`L2`** — same edit region; do them together, and the ordering is the whole
   fix.
3. **`L3`, `L4`** — any time.

---

# Pass 5 — 2026-08-17, against `bdcf4f2`

Two commits: `7b67a30` recorded pass 4, `bdcf4f2` is the response.

New findings carry `M` IDs.

## P5.0 Verdict

**`L2`, `L3` and `L4` are clean, and `L1`'s mechanism is exactly right — but it
was installed in one of the two blocking loops, and not the one that matters.**

Failing the outstanding indices through `Failure` is the correct shape: it uses
the one path Invariant 1 allows, needs no new message type, and lets the run keep
its pose cache. The `drain`-then-check ordering is right too, so a child that
dies after delivering its last result is not falsely blamed.

The check sits in the quiescence loop. The other blocking loop — the admission
gate, `while admission.in_flight() >= WINDOW: drain(block=True)` — has no check,
and that is where the child spends essentially the entire run. A child that dies
at model 200 of 1758 hangs there, exactly as before (`M1`). The note's claim that
the hang table is "closed on all four causes" is not yet true, and my pass-4
wording deserves part of the blame: I wrote "inside the loop" where I meant
inside `drain`, which is the one place both callers share.

Two supporting findings: `outstanding()` rests on a map that does not exist
(`M2`), and the closed-table claim papers over a fifth cause — a child that
**wedges** rather than dies (`M3`).

## P5.1 Disposition of pass 4

| finding | status |
|---|---|
| L1 | mechanism taken and correct; placement covers one of two loops — `M1`. Supporting gaps: `M2`, `M4` |
| L2 | taken — flush-then-warn, `ChildDied` gone, and the reasoning about *why* the exitcode is only a diagnostic there is stated rather than implied |
| L3 | taken — `Release` edge in the diagram, `src/done.py` named as the second parent-side writer on `tasks`. See `M5` for a cosmetic placement nit |
| L4 | taken, and the `I1`/`os._exit` coupling is now explicit in the note — *"moving the sentinel earlier breaks both fixes at once"* is the sentence that keeps someone from undoing it |

## P5.2 New findings

### M1. The liveness check is in the wrong loop — HIGH

Two loops in `run()` block on `drain(block=True)`. Only the second got the check:

```python
    for index, f in enumerate(walker):
        while admission.in_flight() >= WINDOW:
            drain(block=True)                 # <-- no liveness check
        ...
    while admission.in_flight() > 0:
        drain(block=True)
        if child.exitcode is not None:        # <-- here
```

The admission gate is where the child does all of its work. A cold run holds
WINDOW files in flight essentially continuously for hours; the quiescence loop is
the last few seconds plus the arbiter tail. So the check guards the window in
which the child is least likely to die, and leaves the one where it actually
does: child dies at model 200, `in_flight()` stays pinned at WINDOW, `drain`
returns nothing every `SHORT`, and the driver spins forever with 199 finished
files unflushed. That is the failure `L1` was raised for, unchanged.

Fix: **put the check inside `drain`**, not in its callers. That is the single
place the `SHORT`-timeout wake happens, both loops route through it, and it cannot
be forgotten by a third caller later:

```python
def drain(block):
    while (m := ...) is not None:
        ...
    for out in poser.poll():
        dispatch(out)
    if child.exitcode is not None:            # after draining: a result already
        fail_outstanding(child.exitcode)      # in the pipe is never mis-blamed
```

`drain(block=False)` in the walker body then also carries it, which is free and
harmless. One caveat to state where the check lands: `fail_outstanding` retires
files, so it must not run while the caller is mid-iteration over something keyed
by the same indices — at the end of `drain` it is not.

### M2. `outstanding()` rests on a map that does not exist — MEDIUM

> `outstanding()` is the driver's admitted `index → file` map minus `Done`'s
> `retired_ids` — both already exist; no new bookkeeping.

Half of that is true. `retired_ids` exists (`J2`). The admitted map does not:
`Admission` is `admitted: int`, `retired: int`, and `in_flight()`
(`data_structures.md` §Supervisor accounting), and the driver's loop is
`for index, f in enumerate(walker)` with nothing retained. There is no
parent-side structure mapping an admitted index back to its `Path`.

It should exist — `Failure(file, index, …)` needs the file, and this is the only
consumer that has to recover it after the fact. But it is new, it needs a home
(the driver, beside `admission`), and it is worth one sentence that it is not
pruned: unpruned it holds one `Path` per admitted file, ~1758 at the end of a full
run, which is nothing — but "no new bookkeeping" will send an implementor looking
for a map that isn't there.

Pruning it properly would need `Done` to tell the driver when an index retires,
which is the coupling `I10` deliberately avoided. Leave it unpruned and say so.

### M3. A child that wedges is not covered, and the table now says it is — MEDIUM

`child.exitcode is not None` detects **death**, not **hang**. A child stuck in a
Filament call — an amdgpu reset, a GPU wedge, a driver deadlock — has
`exitcode is None` forever, sends nothing, and both loops spin exactly as they did
before `L1`. That is a fifth cause, and it is the one the closed-table sentence
now discourages a reader from looking for:

> The child-death table is now closed on all four causes

The table is accurate about *death*. Rename it that, and then make an explicit
call on the fifth, because both answers are defensible and only silence isn't:

* **Accept it**, with the reason: the child's unit of work is one model, bounded
  in practice at 34 ms–2 s, so a stall is a real fault rather than slow progress —
  and Ctrl-C plus the abort path already recovers the pose cache, which is the
  artifact that costs money.
* **Or bound it**: `drain` already knows when it last received anything. A
  no-progress deadline (generous — 60 s is ~30× the slowest observed model) turns
  a wedge into the same `fail_outstanding` path `M1` installs, at the cost of one
  timestamp.

I lean toward bounding it, because `M1`'s fix puts the machinery one line away and
this repo has a documented history of Filament aborting rather than returning.

### M4. Failing parked files discards a paid arbiter call — LOW

`outstanding()` is admitted-minus-retired, which includes files parked on an
arbiter `Future`. A dead child has nothing to do with them: their ensemble pose is
already in the store via `record_pose`, and their VLM answer — ~$0.30 and ~24 s
in flight — is about to arrive.

For most runs failing them is still correct, because the answer only matters if an
`EmbedRenderTask` can follow it, and a dead child cannot serve one. The exception
is `--skip-embed`, where a resolved file needs nothing further from the child and
would retire as `Retired` on its own. So the honest policy is "fail the files that
still need the child", and under `--skip-embed` that is none of the parked ones.

Low value, one clause — but the arbiter answer is the one thing in this pipeline
that costs money to reproduce, which is why it is worth not discarding by default.

### M5. Two undefined names, and a diagram nit — LOW

* `warn(...)` is introduced undefined, the same way `ChildDied` was last pass. A
  `print` to stderr is presumably meant; say so, since the note's preamble claims
  every named type is `data_structures.md`'s.
* The new `Release` line in the wiring diagram sits between `TileEmbeds` and
  `on_tile_embeds` and runs rightward off the parent's column, so it reads as
  crossing the boundary at the ensemble rather than at the `tasks` queue drawn at
  the top. The edge is correct; the routing is misleading in the one artifact
  people check before believing the prose.

## P5.3 What checked out — do not re-verify

* **`drain`-then-check is the correct order.** A child that dies after sending its
  last result has that result consumed by `drain` before the check runs, so it is
  never blamed for work that actually completed. Keeping that order matters when
  the check moves into `drain` (`M1`) — it must be after the recv loop *and* after
  `poser.poll()`.
* **The failure path composes with everything downstream.** Each `Failure`
  retires idempotently (`J2`), emits a `Release` to a dead child (harmless on an
  unbounded queue), and any later `poll()` output for the same index is a no-op.
  `EndOfInput` to a dead child, `child.join()` on an already-dead child, and the
  post-flush warning all behave.
* **`L2`'s reasoning is sound**, and worth keeping verbatim: after quiescence the
  exitcode genuinely is only a diagnostic, because the in-loop check has already
  converted anything lost into rows.
* **The untimed `child.join()` is still safe.** At quiescence the child is idle in
  `tasks.recv` with an empty `results` queue, so it reads `EndOfInput`, flushes,
  and `os._exit(0)`s. The `M3` wedge case reaches the join only if it resolves
  first, in which case there is nothing to wait for.

## P5.4 Suggested order

1. **`M1`** — move the check into `drain`. Everything else this pass is text.
2. **`M3`** — decide it while `M1`'s machinery is open; the bounded version is one
   timestamp on top of `M1`.
3. **`M2`** — name the map's home and that it is unpruned.
4. **`M4`, `M5`** — any time.

---

# Pass 6 — 2026-08-17, against `35cb91e`

Two commits: `24f8680` recorded pass 5, `35cb91e` is the response.

New findings carry `N` IDs.

## P6.0 Verdict

**`M1` landed exactly right, and `M2`/`M4` are clean. `M3` — the one finding
where I offered a lean instead of a fix — came back as a stall detector that
cannot tell a wedged child from a healthy arbiter tail, which is the one thing
this pipeline does by design.**

The liveness check is now the last statement in `drain`, after the recv loop and
after `poser.poll()`, and both blocking loops route through it. That is the fix,
it is in the only place a third caller cannot forget, and the epilogue is
correctly reduced to a diagnostic `print`. `M2` names `admitted_files` as new
bookkeeping and argues the unpruned choice from `I10`; `M4` splits "still needs
the child" out of admitted-minus-retired and protects the paid arbiter answer.

The wedge bound is where it goes wrong, in three compounding ways:

* the deadline treats **arbiter latency as child silence** (`N1`). In the
  quiescence tail the child is idle *by construction* — that is `I1` — while
  parked files sit outstanding. The arbiter is 24 s mean, **45 s p95**
  (LEARNINGS, `2026-08-12-where-a-7-hour-run-went.md:853`) against a 60 s
  deadline. The detector's own trigger condition is a normal end-of-run.
* `STALL_S` kept the number pass 5 suggested but replaced the premise that
  justified it (`N2`). I wrote "60 s is ~30× the slowest observed model" from a
  wrong figure; the response correctly replaced it with the real one — and 60 s
  against 28 s is **2.1×**, which is not "generous".
* nothing stops admission after `fail_outstanding` (`N3`), so a false positive
  does not cost the in-flight window — it converts **every remaining file** in
  the walk to a `Failure` row and finishes with a complete-looking CSV.

Those three multiply: a plausible mis-fire on a healthy run silently destroys the
run. `N4` is the mechanical half — `last_progress` is a local, bound only inside
the recv loop, so it is unbound in exactly the case the branch exists for.

`M1`, `M2`, `M4`, `M5` need nothing further. This pass is the `M3` machinery.

## P6.1 Disposition of pass 5

| finding | status |
|---|---|
| M1 | **taken and correct** — check is the last statement in `drain`, after recv *and* `poll`; the caveat about not running mid-iteration is stated. The epilogue's inline check is gone. See `N4` for the timestamp's scope, not the placement |
| M2 | taken — `admitted_files: dict[int, Path]`, named as new, homed beside `admission`, unpruned with the `I10` reason. Nothing further; `N5` is one clause the *filter* still needs |
| M3 | bounded rather than accepted, which was my lean — but the bound does not hold: `N1`, `N2`, `N3`, `N4` |
| M4 | taken, and the reasoning is better than my finding was: "the files that still need the child" is now the definition rather than a `--skip-embed` special case in prose. `N1` argues the same split belongs in the *stall* predicate, in every mode |
| M5 | `warn` → `print(..., file=sys.stderr)` taken. The diagram edge moved to the top as asked and is now the correct edge — drawn one column short (`N7`) |

## P6.2 New findings

### N1. The stall deadline counts arbiter time as child silence — HIGH

`last_progress` is bumped only by messages arriving on `results`, i.e. only by
the child. The trigger is `outstanding() and now() - last_progress > STALL_S`.
Both halves are satisfied by a *healthy* run's arbiter tail:

* the walker is dry, so the only files left are parked on arbiter `Future`s —
  this is `I1`, the reason the quiescence loop exists at all;
* those files are admitted and not retired, and outside `--skip-embed` they still
  need the child (an `EmbedRenderTask` follows the answer), so `M4`'s filter
  keeps them: `outstanding()` is **non-empty by design**;
* the child has an empty queue and sends nothing, so `last_progress` is frozen at
  whenever the last render finished.

The clock then runs against the arbiter, which the repo measures at **24 s mean,
45 s p95** for 3.5-flash. A 60 s deadline sits 1.33× above p95 on a per-call
basis, and a run makes ~350 arbiter calls (~20% of 1758). The tail only needs one
answer to arrive more than 60 s after the last child result — and `fail_outstanding`
then `kill()`s a perfectly healthy child and fails precisely the files whose
~$0.30 answers were in flight. `M4` exists to avoid discarding that answer; this
path discards it *and* kills the child.

The same shape reaches mid-run whenever every file in the window is parked at
once, which with a small `WINDOW` and a ~20% escalation rate happens a few times
per full run.

Note the irony: `--skip-embed` is the **safe** mode here, because `M4` excludes
parked files from `outstanding()` and the trigger's first half goes false.

Fix: the split `M4` already found is the right one, it is just applied to the
wrong predicate. Keep the broad `outstanding()` for *who to fail on death*, and
add the narrow one for *what counts as evidence of child liveness*:

```python
child_owed()   # admitted − retired − parked, in EVERY mode
outstanding()  # admitted − retired − (parked if skip_embed)   [M4, unchanged]

if child.exitcode is not None or \
        (child_owed() and now() - last_progress > STALL_S):
    fail_outstanding()
```

A wedged child always has `child_owed()` non-empty — a wedge means it is holding
a task — so nothing is lost. A healthy arbiter tail has `child_owed()` empty, so
the clock cannot run. That is the whole fix, and it makes `STALL_S` a statement
about rendering rather than about the network.

### N2. `STALL_S` kept a constant whose premise was withdrawn — MEDIUM

The note justifies 60 s as "generous … against a p99 model's ~28 s of child
work". Both halves need correcting, and the second is mine:

* **The response is right and pass 5 was wrong.** I wrote that the child's unit
  of work is "bounded in practice at 34 ms–2 s". 34 ms is a `show_geometry(True)`
  **re-show of a resident mesh** (`renderer_alternatives.md:46`,
  `actors_proposal.md:111`), not a model. The real figure is the one the response
  found: **3–28 s of local work for a whole model** (`actors_proposal.md:196`),
  with the p99 mesh load alone at 15.4 s. Good catch, and it should be cited in
  the note rather than asserted — as written, "a p99 model's ~28 s" attributes a
  p99 to a source that gives a range.
* **But then 60 s is 2.1× the top of the documented range, not 30×.** The
  constant was chosen under my wrong premise and kept under the corrected one.
  "Generous" is no longer a fair description of a 2.1× margin on a box with a
  documented thermal ceiling and a documented 1.17–1.21× overlap variance — and
  the range is a sample, while the collection contains an 800k-triangle STL.

The asymmetry decides the number. A wedge is **permanent**: detecting it in 300 s
instead of 60 s costs four minutes of a multi-hour run, once. A false positive
costs the run (`N3`). So `STALL_S` should be set where no healthy child can
plausibly reach it — **~300 s**, ~10× the documented top-of-range — and the note
should say that it is deliberately far above the work distribution because the
error is one-sided. With `N1`'s `child_owed()` narrowing, a 300 s deadline still
catches every wedge it was written for.

### N3. Nothing stops admission after `fail_outstanding` — MEDIUM

`fail_outstanding` retires the outstanding files, which drops `in_flight()` below
`WINDOW` and releases the admission gate — so the walker **continues**, admitting
the next file to a dead or killed child. It fills the window again, `drain` sees
`exitcode is not None` again, fails those, and so on to the end of the walk.

It terminates, so this is not `I1`'s hang. What it produces is worse to read: a
run whose child died at model 200 walks all 1758, writes ~1558 `Failure` rows,
prints one line to stderr, and `done.flush()`es a CSV that looks complete. The
`L2` reasoning — "at this point the run is complete and correct" — is stated in
the epilogue and is no longer true on that path.

Two defensible answers; the note should pick one and say it:

* **Stop admitting.** `fail_outstanding` sets a flag the walker checks; the loop
  breaks and the remaining files are simply not in the CSV, which is what a
  crashed run should look like. Note this still finishes the arbiter tail via
  `poll`, so paid answers are not lost.
* **Keep walking, deliberately** — because `route()` still serves warm
  `CachedHit`s without the child, so a warm-cache run survives a child death
  outright and a cold one degrades to rows-per-file. If this is the intent it is
  a genuine feature, but it needs the sentence, and the epilogue's "complete and
  correct" needs a qualifier.

Either way `N1`'s false positive stops being run-fatal, which is why this is on
the critical path and not a nit.

### N4. `last_progress` is a local, unbound exactly when it is read — MEDIUM

```python
def drain(block):
    while (m := ...) is not None:
        last_progress = now()          # assigned only inside the loop
        ...
    if ... now() - last_progress > STALL_S:   # read after it
```

Read literally this is a `NameError` on any `drain` whose recv loop yields
nothing — which is the *only* case in which the stall branch can be true. Read
charitably as driver-level state, it is still wrong in two ways: it has no
initialisation point, and it is not bumped by `poser.poll()` output.

All three need stating:

* **home**: driver state beside `admission` and `admitted_files`, not a local.
* **initialised at spawn**, so the child's Filament/open3d startup is inside the
  first interval rather than being measured against a timestamp that does not
  exist.
* **bumped by `poll()` output too**, not just by `results`. An arbiter answer
  folding is progress; without this the tail's silence compounds `N1` even after
  `child_owed()` narrows the predicate.

### N5. The `M4` filter needs two names `M2` would have insisted on — LOW

`outstanding()` is now "admitted map minus `retired_ids`, filtered to files that
still need the child", and the filter reads two things the note does not name:
the Poser's `parked` dict (`data_structures.md` §Poser continuation state — it
exists, so this is a citation, not new state) and `cfg.skip_embed`, which makes
`outstanding()` mode-dependent. `M2`'s lesson was that "both already exist" sends
an implementor looking for something that isn't there; one clause naming
`poser.parked` and the cfg flag closes the same gap one pass later. `N1`'s
`child_owed()` reads the same dict, so both land in one edit.

### N6. Undefined names, third pass running; and the check re-fires — LOW

* `now()`, `STALL_S` and `kill()` are introduced undefined, the same way `warn`
  was last pass and `ChildDied` the pass before. `STALL_S` is a config constant
  and should sit with `WINDOW` and `SHORT`; `kill()` is `child.kill()`.
* After the child dies, `child.exitcode is not None` is true on **every**
  subsequent `drain`, so `fail_outstanding` runs each time over an empty set.
  Harmless — retirement is idempotent (`J2`) — but the guard reads better and
  self-documents as `if outstanding() and (dead or stalled)`, which also states
  that there is nothing to do when nothing is owed.

### N7. The `M5` diagram edge is drawn one column short — LOW

The `Release` line moved to the top as asked, and it is now unambiguously the
`tasks` edge. It is one character narrower than the box:

```
line 62  │  ...  │ ─────▶ │   borders at cols 0, 41, 50, 80
line 63  │ done ─┼─────▶ │    borders at cols 0, 41, 49, 79   ← the new line
```

One more `─` (`┼──────▶ │`) and one more trailing space put the child box's walls
back at 50 and 80. While there: line 65 (`│ bound- │ … LoadedMesh`) has the same
off-by-one at the right wall and predates this pass — fix both in one edit, since
the diagram is the artifact people check before believing the prose (`L3`).

## P6.3 What checked out — do not re-verify

* **`M1`'s placement is right, and for the stated reason.** The check is after
  the recv loop *and* after `poser.poll()`, so a result already in the pipe is
  consumed before the child can be blamed for it; both blocking loops route
  through `drain`; `drain(block=False)` in the walker body carries it harmlessly;
  and `fail_outstanding` at the end of `drain` is not inside any caller's
  iteration over the same indices. Pass 5's `L1`/`M1` thread is closed.
* **The response corrected the reviewer, and should be believed over pass 5.**
  34 ms–2 s was my error; 3–28 s per whole model is the repo's number. `N2`
  adjusts the constant, not the correction.
* **`--skip-embed` with a dead child still terminates.** `M4` excludes parked
  files from `outstanding()`, so `fail_outstanding` leaves them in flight — and
  they retire on their own when the future folds through `poser.poll()` →
  `dispatch` → `done.on(Retired)`, which needs no child. `in_flight()` reaches
  zero. This looks like a leak and is not; leave it.
* **Kill-before-join is the right order** (`M3`'s own reasoning). The untimed
  `child.join()` cannot meet a live wedge because `fail_outstanding` kills first,
  and a killed child's nonzero exitcode then prints as the diagnostic `M5`/`L2`
  made it — which is the correct signal for that run.
* **`M2`'s unpruned map is settled.** ~1758 `Path`s is nothing, and pruning needs
  the `Done`→driver callback `I10` avoided. Do not revisit; if it ever matters,
  the retirement callback is the cost, not the memory.
* **`admitted_files` is written before `dispatch`, not after** — so a file that
  raises in `route()` and retires through the `except` arm is still in the map
  when `outstanding()` subtracts `retired_ids`. Correct as written.

## P6.4 Suggested order

1. **`N1`** — `child_owed()` for the stall predicate. Without it the detector
   fires on healthy runs, and it is the same split `M4` already made.
2. **`N4`** — the timestamp's home, initialisation, and `poll()` bump; same edit
   region as `N1`.
3. **`N3`** — decide whether the walk stops. It is what makes `N1`'s failure mode
   run-fatal rather than window-sized.
4. **`N2`** — raise `STALL_S` and state the one-sided-error reason; cite
   `actors_proposal.md:196` for the 3–28 s.
5. **`N5`, `N6`, `N7`** — any time.

---

# Pass 7 — 2026-08-17, against `d45d8cb`

Three commits: `d716017` recorded pass 6, `b1a7a79` is the schema/process
write-up, `d45d8cb` is the response.

New findings carry `O` IDs.

## P7.0 Verdict

**Every substantive finding is taken, and one of them is taken better than it
was raised. What is left is scoping — the new state is described as driver state
and written as if it were local, which is the same bug `N4` was about, one level
up.**

`child_owed()` is now a named subtraction sitting beside `outstanding()`, with
the two roles spelled out where they differ ("who to fail" vs "what counts as
evidence of child liveness"). That is `N1` exactly. `STALL_S` is ~300 s with the
one-sided-error argument and the `actors_proposal.md:196` citation. `N7` verifies:
all thirteen box rows now put the child walls at columns 50 and 80.

`N3` came back **better than I asked for**. I offered "stop admitting" and
accepted losing the in-flight arbiter answers as the price; the response took the
option and then noticed that `in_flight() > 0` alone would drop the paid answers
it had just spent `M4` protecting, and added `or poser.parked` so each fold still
`record_pose()`s before `flush`. That is the finding's intent rather than its
letter, and it is the right call.

No HIGHs this pass. The remaining findings are plumbing (`O1`–`O3`), one citation
the clause needs to look bounded (`O4`), one stale signature (`O5`), and one
correction to the process write-up, which now contradicts itself (`O6`).

`O1` is the only one on the critical path: as written, **both** the `N3` and `N4`
fixes are inert.

## P7.1 Disposition of pass 6

| finding | status |
|---|---|
| N1 | **taken, and the shape is right** — `child_owed()` is a named subtraction with its role stated against `outstanding()`'s. See `O2`: it makes half of `N4` obsolete |
| N2 | taken — ~300 s, the 3–28 s citation, and the one-sided-error reasoning stated as the reason rather than the conclusion |
| N3 | taken, **and improved**: `child_failed` stops the walk, and `or poser.parked` keeps the paid answers that the letter of my finding would have discarded. Scoping is `O1`, an off-by-one is `O3`, the clause's bound is `O4` |
| N4 | taken in prose — spawn init, `poll()` bump, "never a local" — but the pseudocode still assigns bare names inside `drain` (`O1`), and the `poll()` bump is now a liability rather than a fix (`O2`) |
| N5 | taken — `poser.parked` cited as existing continuation state, `cfg.skip_embed` named as what makes `outstanding()` mode-dependent |
| N6 | taken — `STALL_S` beside `WINDOW`/`SHORT`, `now()` is `time.monotonic()`, `child.kill()`, and the outer `outstanding()` guard. `child_failed` is the one new name that arrived undefined (`O1`) |
| N7 | verified — columns 50/80 on every box row, `LoadedMesh` included |

## P7.2 New findings

### O1. The new driver state is written as locals — MEDIUM

Two names are now read in `run` and written in `drain`/`fail_outstanding`:

```python
def run(cfg):
    last_progress = now()
    ...
        if child_failed: break

def drain(block):
    last_progress = now()          # <-- rebinds a LOCAL of drain
```

If `drain` is a closure over `run` — which it must be, since it reads `child`,
`poser`, `admission` and `done` — then in Python a bare assignment makes
`last_progress` local to `drain`, and `now() - last_progress` two lines later
raises `UnboundLocalError` on every call whose recv loop and poll both came back
empty. That is `N4`'s bug, unchanged, one scope up: the note now *says* "driver
state … never a local" while the code still shows the binding that makes it one.

`child_failed` has it worse. `fail_outstanding` sets it, `run` reads it, and the
write is invisible to the reader for the same reason — so the walker never
breaks and `N3` silently does not happen. It is also never initialised anywhere.

Pick a mechanism and show it, since this is the third pass in a row where a fix
was correct in prose and unbound in code:

* `nonlocal last_progress, child_failed` at the top of the functions that write
  them, plus `child_failed = False` beside `admission` at spawn; or
* the state these three names describe is really one object —
  `admission`, `admitted_files`, `last_progress`, `child_failed` — so
  `drv.last_progress = ...` and `if drv.child_failed: break` removes the class of
  bug rather than patching this instance.

I lean to the second: `M2`, `N4` and `O1` are all the same finding arriving
three times, which is the signal that the container is missing and not the
declaration.

### O2. The `poll()` bump is now a wedge-detection weakener — LOW

`N4` asked for `last_progress` to be bumped by `poll()` output, and the response
took it. That request was written against the *old* predicate, where the tail's
silence was what caused the false positive. `N1` fixed that at the source:
`child_owed()` is empty in the tail, so the clock cannot run there no matter what
the timestamp says. The bump now prevents nothing — and it costs something.

A child that wedges mid-run while arbiter calls are still outstanding has its
stall clock reset by every fold. Bounded, not unbounded: a wedged child sends no
`PoseTiles`, so no new files park, and the set drains within the arbiter's own
deadline. But the worst case doubles detection — a fold landing at 299 s buys
another full `STALL_S` — for no remaining benefit.

Drop it, or gate it (`if not child_owed(): last_progress = now()`). My finding,
my error; `N1` and `N4` overlapped and only one of them was needed.

### O3. The `break` lets exactly one more file through — LOW

`if child_failed: break` sits at the top of the loop body, but the most likely
place for `fail_outstanding` to fire is the admission gate *below* it:

```python
    for index, f in enumerate(walker):
        if child_failed: break                 # checked here
        while admission.in_flight() >= WINDOW:
            drain(block=True)                  # ...set here
        admission.admitted += 1                # ...so this file is still admitted
```

Retirement drops `in_flight()` to zero, the gate releases, and the current file is
admitted and dispatched to a child already known to be dead — then failed on the
next `drain`. So the CSV `N3` wanted cleanly truncated ends with one `Failure`
row for a file nothing ever tried to render, which is the one row a reader would
take seriously.

`while not child_failed and admission.in_flight() >= WINDOW:` and a re-check
after the gate, or hoist both into one `if child_failed: break` after the gate.

### O4. The `or poser.parked` clause looks unbounded and is not — LOW

On the death path the loop runs with `in_flight()` at zero, waiting on futures
whose child is gone. Nothing in the note bounds that wait, and the reader who has
just read six passes about hangs will look for the bound.

It exists, in the code and not in the note: the arbiter's HTTP call carries
`timeout=300` (`pose.py:456`, and `pose.py:379` for the ollama path). Cite it
where the clause is introduced — *"bounded by the arbiter's own 300 s transport
deadline, not by anything in this loop"* — and note that a killed child plus a
just-submitted call means up to five minutes between the stderr line and the
CSV, so nobody reads the tail as a hang.

While there: that deadline and `STALL_S` are now both 300 s and mean unrelated
things. Either say the collision is a coincidence, or move `STALL_S` off it
(240 s is still ~8.5× the top of range).

### O5. `poll()`'s signature omits `Failure`, and the walrus depends on it — LOW

Line 220 declares `def poll(self) -> list[EmbedRenderTask | Retired]`; the prose
eleven lines down declares `list[EmbedRenderTask | Retired | Failure]`, which is
the true one (`J3`, `K3`). Pre-existing, but two things now lean on it: `dispatch`
walks `polled` and must route the `Failure` arm, and `if polled := poser.poll()`
is only correct because the return is a **list**.

That second one is worth stating where the walrus is, because the prose describes
`poll` with generator language ("each resumed file *yields* its task"). If anyone
ever makes it a generator, `if polled:` is unconditionally true, `last_progress`
is bumped on every `drain`, and the stall clock silently never fires again — a
wedge goes back to hanging the run, with no symptom. Fix the signature, and say
`poll` returns a list *because* the caller tests it.

### O6. The process write-up's convergence claim contradicts its own data — LOW

`docs/learnings/2026-08-17-cache-schema-and-design-by-review.md` is a good note,
and two lines in it are now wrong:

* **"Findings shrinking monotonically is the convergence signal"** — neither
  sequence in the same bullet is monotonic: data structures `15 → 7 → 5 → 2 → 4
  → 1`, interfaces `16 → 8 → 7 → 4 → 5`. Pass 6 then went `5 → 7`. The claim
  matters because it is the stopping rule; on a count test this review would have
  been called converged at `L4`, one pass before the liveness hang and two before
  the stall clock. What actually fell monotonically is **severity and order**:
  `I`/`J` were hangs in the protocol, `K`–`M` were failure-path holes, `N` was one
  mechanism's arithmetic, `O` is scoping and citations. Amend to that.
* **"eleven review passes"** (also in the header and `LEARNINGS.md` index line)
  is twelve with pass 6, thirteen with this one.

Also worth amending rather than deleting: *"Verify a reviewer's numbers before
folding them in (all of them reproduced, every time)"* now has its first
counter-example, and it is the best evidence for the lesson — pass 5's
`34 ms–2 s` did **not** reproduce, the response checked it, found `3–28 s`, and
corrected the reviewer. That is the convention working, not an exception to it.

## P7.3 What checked out — do not re-verify

* **`child_owed()` is the right cut, in the right place.** Empty in the tail
  where the child is idle by construction (`I1`), never empty during a wedge
  because a wedge means holding a task. The outer `outstanding()` guard makes
  the dead-child arm idempotent-cheap. `N1` is closed.
* **`or poser.parked` is correct on both paths, and non-obvious.** Outside
  `--skip-embed` the parked files were already retired by `fail_outstanding`, so
  `in_flight()` is zero while their answers are still in the air; the clause is
  what makes `record_pose` run before `flush`. Inside `--skip-embed` they were
  never failed, so `in_flight()` covers them and the clause is redundant but
  harmless. Folding a task for an already-retired index dispatches to a dead
  child and is a no-op (`J2`). Do not simplify this away.
* **`N7` is verified mechanically**, not by eye: every box row of the wiring
  diagram has its borders at columns 0, 41, 50, 80.
* **The epilogue's qualifier is now accurate.** "Complete and correct" scoped to
  the clean walk, with the death path's stderr line plus truncated CSV named as
  the crash report, is the honest version of `L2`.
* **`STALL_S` at ~300 s clears every documented child work unit** — 3–28 s per
  model, 15.4 s p99 mesh load — by ~10×, and clears the arbiter entirely now that
  the arbiter cannot start the clock at all.

## P7.4 Suggested order

1. **`O1`** — until the two names are scoped, `N3` never fires and `N4` raises.
   Prefer the state object; it retires `M2`/`N4`/`O1` as a class.
2. **`O3`** — same edit region, one line.
3. **`O2`** — delete the bump, or gate it on `not child_owed()`.
4. **`O4`, `O5`** — one citation and one signature.
5. **`O6`** — amend the write-up in place, per the LEARNINGS convention.

---

# Pass 8 — 2026-08-17, against `ea11bc3`

Two commits: `befe283` recorded pass 7, `ea11bc3` is the response.

New findings carry `P` IDs.

## P8.0 Verdict

**The protocol is done. Every finding this pass is about what the note *says*,
not what it does — and the biggest one is that the code's real behaviour on a
dead child is better than the sentence describing it.**

`DriverState` is the right answer to `O1`, and the reasoning attached to it —
three passes found the same bug one field at a time, so the container was the
missing thing — is the correct generalisation. `O3`'s gate/break ordering is
right. `O2` dropped the bump rather than gating it, which also removed the walrus
and mooted half of `O5`; `O4` took the harder half (move `STALL_S` off 300 s so
the number means one thing) rather than annotating a collision. The write-up
amendments in `O6` are honest, including keeping the counter-example as the
evidence rather than deleting the claim.

I checked the things a new container usually breaks and they hold: `src/driver.py`
was already in the module map (line 39), the dataclass is well-formed (no mutable
default, non-default fields first), `now()` is `time.monotonic()` so
`last_progress: float` types, and no stale "a fold is progress" prose survives
anywhere in the note.

What is left:

* the dead-child walk behaves **differently and better** than the note's
  description of it, at both edges (`P1`);
* `DriverState` is half-owned by `src/done.py` and constructed from an
  `Admission` that appears from nowhere (`P2`);
* `data_structures.md` — the note that owns these shapes — knows about none of
  it, and still states a quiescence condition that `N3` changed (`P3`);
* `poser.parked` is read across a module boundary that the Poser's own interface
  listing does not expose (`P4`);
* the amended convergence bullet is off by one pass (`P5`).

**This should be the last pass.** By the write-up's own amended rule — stop when
the findings stop being about the design — pass 8 is the stop. `P1`–`P5` are text
edits against a settled protocol; none of them changes a line of the mechanism.

## P8.1 Disposition of pass 7

| finding | status |
|---|---|
| O1 | **taken, container option** — `DriverState` with all four fields, `drv.*` writes, homed in `src/driver.py`, and the "three passes, one field at a time" reasoning recorded as why. Ownership and construction are `P2` |
| O2 | taken — bump dropped, not gated, and the walrus went with it. The `for out in poser.poll()` form is back to the pre-`N4` shape, which is correct |
| O3 | taken — gate tests `not drv.child_failed`, break sits below it. The remaining fuzziness is `P1`, and it is not this fix's fault |
| O4 | taken, and the harder half: `STALL_S` → ~240 s so 300 s means only the arbiter's transport deadline, with `pose.py:456`/`:379` cited at the clause. ~8.5× checks out against the 3–28 s range |
| O5 | taken — signature includes `Failure`; the generator hazard is moot now that no caller tests the return for truth |
| O6 | taken and amended in place, both files plus the index line. The new bullet is right in substance, off by one pass in its example (`P5`) |

## P8.2 New findings

### P1. The dead-child walk is better than its own description — MEDIUM

The note says:

> un-walked files are simply absent from the CSV, which is what a crashed run
> should look like

That is not what happens at either edge, because `drv.child_failed` is set only
inside `fail_outstanding`, which is itself gated by `if outstanding() and …` —
my `N6` guard. Death is therefore *noticed* only when something is owed:

* **Front edge.** A child that dies while `in_flight()` is low is not noticed.
  The walker keeps admitting, and each admitted file that needs the child sits
  there until the gate fills — so up to `WINDOW` files are dispatched to a known-
  dead child and end as `Failure` rows for files nothing ever tried to render.
  That is the row `O3` just removed, arriving `WINDOW` at a time from the other
  direction.
* **Back edge.** Once `child_failed` is set the walk stops, which discards the
  remaining **warm** files — the ones `route()` would have served as `CachedHit`
  without touching the child at all.

But look at what the guard actually buys, because it is the interesting half: on
a **fully warm run nothing is ever outstanding**, so a dead child is never
noticed, the walk never breaks, and the run completes correctly end to end. A
child that fails to initialise Filament on the iGPU cannot damage a warm re-run.
That is exactly right, and it is currently an accident of where the guard sits.

So: keep the code, fix the sentence. The honest description is three clauses —

1. a run that needs nothing from the child completes normally, however the child
   died;
2. otherwise the walk stops within `WINDOW` files of the first one the dead child
   could not serve;
3. the CSV therefore ends with up to `WINDOW` `Failure` rows and then stops —
   truncated, not complete, and the stderr exit line is the marker.

If you would rather make (1) deliberate instead of emergent, the two-line version
separates noticing from failing:

```python
if child.exitcode is not None:
    drv.child_failed = True                       # noticing is free
if outstanding() and (drv.child_failed or
        (child_owed() and now() - drv.last_progress > STALL_S)):
    fail_outstanding()
```

— but note that on its own this makes the front edge exact **at the cost of (1)**:
a warm run would now break on a dead child it never needed. The current guard
trades an exact edge for warm-run survival, and given that this project's product
is the caches and the CSV is a thin consumer, that is the right trade. Say it is
the trade.

### P2. `DriverState` is half-written by `src/done.py`, and its `Admission` comes from nowhere — LOW

```python
drv = DriverState(admission, {}, last_progress=now())
```

`admission` is never constructed in `run` — the same class of undefined name the
last three passes have been closing (`warn`, `ChildDied`, `now`, `STALL_S`,
`child_failed`). Worse, it cannot be constructed here casually: `Done.__init__`
takes `admission: Admission` (line 282) and is the **only** writer of `retired`
(`data_structures.md` §Supervisor accounting). So `drv.admission` must be the
same object `Done` holds.

A dataclass named `DriverState`, homed in `src/driver.py`, described as "bookkeeping,
not a message", reads as exclusively driver-owned. The first implementor who
constructs `Done` with its own `Admission`, or who deep-copies `drv`, gets an
`in_flight()` that never decreases — `I1`'s hang, reached through the container
that was added to prevent a scoping bug. One clause where the dataclass is
declared:

> `admission` is constructed once and passed to both `Done` and `DriverState` —
> the same instance, not a copy. `admitted` is the driver's, `retired` is
> `Done`'s; the container holds the reference, it does not own the object.

### P3. `data_structures.md` knows about none of this — MEDIUM

Per `CLAUDE.md`'s doc map, `data_structures.md` fixes the shapes and this note
fixes the calling conventions. A four-field dataclass wrapping `Admission` is a
shape, and §Supervisor accounting — the section the data-structures review spent
six passes settling — is now stale in two ways:

* it still reads *"In v1 this is a plain counter the driver loop consults"*, with
  no mention of `DriverState`, `admitted_files`, `last_progress` or
  `child_failed`. Either move the dataclass there and cite it from here, or add
  the pointer line; the current split means the shapes note describes a driver
  that no longer exists.
* it states quiescence as **"Walker exhausted and `admitted == retired`"**, which
  `N3` changed: the loop is now `in_flight() > 0 or poser.parked`, and on the
  death path those differ. That is an invariant living in the invariants note,
  contradicted by the loop in this one. Amend it — quiescence is
  `admitted == retired` **and** no parked file — and note that only the post-
  `fail_outstanding` path can tell them apart.

Cross-note drift is the failure mode this review has caught most often (`I8`, the
`Q1` transport question, `N5`'s citation). It is worth one edit to close.

### P4. `poser.parked` is read across a boundary the Poser does not expose — LOW

The quiescence loop reads `poser.parked` directly, and `outstanding()`'s
`--skip-embed` filter reads it too (`N5`). The `Poser` class sketch (lines
215–222) lists five methods and no attributes; `parked` is declared in
`data_structures.md` as continuation state, i.e. as the Poser's *internals*.

For a note whose entire subject is who calls whom with what signature, two
consumers reaching into another module's dict deserves either a line in the class
sketch (`parked: dict[int, ParkedFile]  # read by the driver: quiescence, N5`) or
a predicate (`def has_parked(self) -> bool`). The predicate is better — it is the
only thing either caller needs, and it keeps `abandon()`'s ownership of the dict
unambiguous.

### P5. The amended convergence bullet is off by one pass — LOW

> a count test would have called the interfaces review converged one pass before
> the liveness hang and two before the stall clock

The sequence is 16 → 8 → 7 → 4 → 5 → 7 → 6. It is still falling at pass 4; the
first violation is pass 5. So a count-based stopping rule fires **after pass 4** —
which means the liveness hang (`L1`, pass 4) *was* caught, and what would have
been missed is the wedge (`M3`, pass 5), the stall clock's arithmetic (`N1`/`N2`,
pass 6) and the scoping bug that made both inert (`O1`, pass 7).

That is a better story for the bullet, not a worse one: the count test does not
fail by stopping before the obvious hang, it fails by stopping right after it,
while three passes of consequences were still unwritten. Third time this bullet
has been wrong; get the arithmetic in it and it can stop being edited.

## P8.3 What checked out — do not re-verify

* **`DriverState` is well-formed and correctly placed.** Non-default fields
  before the default, no mutable default (the `{}` is passed at the call site),
  `last_progress: float` matches `time.monotonic()`, and `src/driver.py` was
  already in the module map with "owns admission" — the container did not need a
  new module or a new import rule.
* **`O2`'s removal is complete.** No "a fold is progress" prose survives; the
  only `last_progress` writes are the spawn init and the recv loop, which is the
  intended pair.
* **The gate/break ordering is right, and `drain(block=False)` composes with
  it.** A `child_failed` set at the bottom of the body skips the next iteration's
  gate via `not drv.child_failed` and breaks immediately; one set inside the gate
  releases it and breaks below. Both paths admit nothing further.
* **`STALL_S` at 240 s still clears everything.** 8.5× the documented 3–28 s
  per-model range, and the arbiter cannot start the clock at all now that
  `child_owed()` gates it — so the 300 s transport deadline and the stall
  deadline never race.
* **`child_owed()` is genuinely "what the child owes".** Every other wait in the
  parent — `on_tiles`, `embed_tiles`, `on_tile_embeds`, `embed_views`, the fold —
  is synchronous inside `drain`, so the arbiter is the only asynchronous
  non-child wait and `parked` is the whole of it. The subtraction is exact.
* **The death-in-the-tail case is designed, not accidental.** A child that dies
  while parked files are outstanding fails them (normal mode), and `or
  poser.parked` still lets each answer fold and `record_pose` before `flush` — the
  pose is kept, the embed is lost, which is the `M4`/`N3` policy applied
  consistently.

## P8.4 Suggested order

1. **`P1`** — the description, and the decision about whether warm-run survival
   is a trade or an accident. Everything else is smaller.
2. **`P3`** — the two `data_structures.md` amendments; the quiescence one is an
   invariant.
3. **`P2`, `P4`** — one clause and one method signature.
4. **`P5`** — arithmetic in the write-up.

Then stop. The protocol terminates, the failure paths close, the shapes have
homes, and the remaining edits are prose.
