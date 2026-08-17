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
