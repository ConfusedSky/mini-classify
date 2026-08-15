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
