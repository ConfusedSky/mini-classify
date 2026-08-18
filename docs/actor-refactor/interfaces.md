# Actor Refactor — Interfaces

Design note, 2026-08-14. Third of the set: [actors_proposal.md](actors_proposal.md)
argues the boundaries, [data_structures.md](data_structures.md) fixes the
shapes, this one fixes the **calling conventions** — who calls whom, with
what signature, what blocks, and who converts errors into `Failure`. Revised
across eight review passes against
[docs/reviews/2026-08-14-interfaces.md](../reviews/2026-08-14-interfaces.md)
(findings I1–I16, pass 2's J1–J8, pass 3's K1–K7, pass 4's L1–L4, pass 5's
M1–M5, pass 6's N1–N7, pass 7's O1–O6, pass 8's P1–P5); the review's gating
questions are answered inline — **Q1: the parent owns admission, and
therefore `tasks` is unbounded; Q2: a `Retired` message is what retires a
file that produces no row; §P2.3: taken — the child always sends exactly one
result per task, `Rendered(file, index)` is the ack that retires a
render-only file, and `Release(file, index)` is the control message that
unpins a resident mesh when retirement happens parent-side (K1).** Types
named
here are data_structures.md's, including the driver-side shapes it now
carries (`RenderConfig`, `CacheContext`, `Redraw`, `Retired`, `EndOfInput`).
Implementation-round findings (2026-08-17: `B-R1-*`, `C-R1-*`, `D-R1-*`,
and the pending `A-R1-*`/`E-R1-*`) resolve through
[docs/reviews/2026-08-17-wave1-implementation.md](../reviews/2026-08-17-wave1-implementation.md).

Everything below is the v1 form: sequential driver, one renderer subprocess.
Where a signature exists only so the threaded successor can slot a queue in
front of it, that is said inline.

## Module map and import constraints

```
src/messages.py      the frozen dataclasses; imports pose (for Pose)
src/transport.py     Transport protocol + MpQueueTransport
src/loader.py        LoadedMesh + get()                    [child only]
src/renderer.py      Renderer: tiles, views, residency     [child only]
src/render_child.py  child process entry: loader+renderer  [child only]
src/cache_checker.py route(): the admission decision
src/poser.py         ensemble + continuations + arbiter calls
src/arbiter.py       windowed ThreadPoolExecutor wrapper
src/embedder.py      SigLIP; the only module that owns torch models
src/done.py          scoring, rows, pose store, retirement, Release, flush
                     — the second parent-side writer on tasks (L3)
src/driver.py        the sequential loop; owns admission
src/pose.py          math + Pose + caches                  [moves from the root]
src/identity.py      cache keying: collection_root + keys  [moves from the root]
classify_stls.py     CLI entry: args, run-params, cache guards -> driver
```

Import rules, and why each is load-bearing:

| module | may import | must NOT import | because |
|---|---|---|---|
| child side (`loader`, `renderer`, `render_child`) | open3d, PIL, numpy, `messages`, pose | **torch** | SigLIP lives in the parent; a torch import in the child costs VRAM and startup for nothing. **This binds `classify_stls.py` too** (wave 2, measured): `spawn` re-executes `__main__` in the child, so a module-scope torch import in the CLI would land in the render child — the CLI imports torch only inside function bodies |
| `poser` | torch (one conversion), numpy, pose, PIL (contact sheet) | open3d renderer calls | the Poser consumes geometry *scores* (computed child-side) and tiles, never the mesh itself |
| `embedder` | torch, transformers | — | the only owner of models |
| `pose` | numpy, open3d, PIL, `identity` | torch, **any other `src/` module** | the standing rule, unchanged by the move: `pose` is the leaf both sides import (the child for `up_axis_scores`, the Poser for `combine_up`, `messages` for `Pose`), so it must depend on nothing in the pipeline. Living in `src/` makes it a sibling of its importers, not a peer that may import back |
| `identity` | stdlib only | **anything in `src/`, and any third-party import** | the deepest leaf: every cache keys on it (invariant 2), `pose` imports it, and a leaf below the leaf must cost nothing to import anywhere — parent, child, or a bare test |
| `messages` | pose, numpy; torch **under `TYPE_CHECKING` only**, with `from __future__ import annotations` | a module-scope `import torch` | the child unpickles its tasks from `messages` — a real torch import there hands the child exactly the dependency the first row forbids (I8). The two tensor-typed messages never cross a queue, so the name is annotation-only |

(pose.py imports open3d for `up_axis_scores`, so the parent transitively
imports open3d too — import only; no renderer is ever created parent-side.)

## The wiring

```
 PARENT PROCESS                                    CHILD PROCESS
┌──────────────────────────┐                      ┌─────────────────────────────┐
│ driver: sequential loop, │                      │ run_child loop              │
│ admission, drain         │  tasks (unbounded)   │                             │
│                          ├────────────────────▶ │  (control: unpins in_flight)│
│ everything else in the   │ *RenderTask | Release│  loader.get(file)           │
│ parent is a function     │ | EndOfInput         │      │ LoadedMesh           │
│ call — the next diagram  │                      │      ▼                      │
│                          │  results (bounded)   │  renderer.pose_tiles /      │
│                          │ ◀────────────────────┤  renderer.views (rot. copy) │
│                          │ PoseTiles(geo,tiles) │  save renders (child owns)  │
│                          │ | EmbedViews         │      │                      │
│                          │ | Rendered | Failure │  resident LRU (bytes)       │
└──────────────────────────┘                      └─────────────────────────────┘
```

The two arrows crossing the box edge are the only queues.

The parent's internals — every arrow a function call or the message it
returns, everything funneling into the two sinks: `tasks.send` (the `tasks`
arrow above) and `done.on` (rows and retirement — not always both; the
`Redraw` arm below writes its row with `retires=False`):

```
 walker ─▶ cache_checker.route(f, i) ─┬─ CachedHit | Retired ──────▶ done.on
                                      ├─ Redraw(task, hit) ────────▶ done.on + tasks.send
                                      ├─ *RenderTask ──────────────▶ tasks.send
                                      └─ raises ─▶ Failure ────────▶ done.on  (J3)

 results, drained every driver iteration — each arm an error boundary (I4):

   Rendered | Failure ─────────────────────────────────────────────▶ done.on
   EmbedViews ─▶ embedder.embed_views ── Embedded ─────────────────▶ done.on
   PoseTiles ─▶ poser.on_tiles
                     │ EmbedTilesRequest
                     ▼
                embedder.embed_tiles
                     │ TileEmbeds
                     ▼
                poser.on_tile_embeds ─┬─ Resolved ─▶ route(f, i) again — the
                                      │             pose store is warm now, so
                                      │             the warm-.npy + redraw arms
                                      │             above apply (parity)
                                      └─ None — parked on
                                         arbiter.submit(call) ── Future ──┐
                                                                          │
   poser.poll(), every iteration — resumes parked files ◀─────────────────┘
        └─▶ Resolved ─▶ route(f, i) again;  Failure ───────────────▶ done.on

 done ─▶ Release, every retirement ────────────────────────────────▶ tasks.send
```

## The boundary protocol

```python
class Transport(Protocol):            # src/transport.py
    def send(self, msg) -> None       # blocks only if bounded and full
    def recv_nowait(self):            # a message, or None if empty
    def recv(self, timeout=None):     # a message; None = nothing arrived
    def close(self) -> None           # drop: cancel_join_thread + close —
                                      # unflushed pickles are abandoned, so an
                                      # aborting parent cannot be held open by
                                      # the queue's feeder thread (I6)
```

* **`tasks` (parent→child) is unbounded; `results` (child→parent) is bounded
  at the admission window** (I2, Q1). The parent-owned admission this note
  chooses (because `route` and `Done` live there) keeps the measured
  property that made the overlap spike deadlock-free: **the parent never
  blocks on a send.** Admission is the *only* forward pressure — every task
  holds its slot until its result or ack comes back (§P2.3), so the child's
  backlog is bounded by admission and `results` occupancy is provably
  ≤ WINDOW. Under that contract, bounding `tasks` at the window would in
  fact never block (K4) — unbounding is defence in depth plus the threaded
  successor's back-edge rule (`Poser → Renderer` is a listed back-edge),
  not load-bearing for deadlock. Kept anyway: a queue that *cannot* block
  the parent survives future contract mistakes the way a provably-unfilled
  bounded one does not.
* `tasks` carries `PoseRenderTask | EmbedRenderTask` plus the **control
  messages** `Release | EndOfInput`; `results` carries
  `PoseTiles | EmbedViews | Rendered | Failure`.
* **End of input is `EndOfInput()`, a frozen message** (I5) — `recv`'s
  `None` already means "nothing arrived within the timeout", and a value
  meaning two things would make the child exit on its first idle window.
  The sentinel is safe *here*, unlike pipeline-wide poison pills, because
  the parent owns admission and sends it only after quiescence (see the
  driver epilogue) — by then nothing can ever enqueue another task.
* **Child lifecycle:** spawned once (`mp.get_context("spawn")`, daemon —
  spawn is load-bearing with CUDA initialised in the parent), handed its
  config whole (`RenderConfig`, shape in data_structures.md; it crosses the
  spawn boundary, so it must stay picklable). The child never reads argv or
  run-params.
* The shm variant changes only what `send`/`recv` carry (`(block_id,
  shapes)` + a free-list back-queue) — the signatures above do not move,
  and `EndOfInput` being a message rather than `None` is what survives that
  variant.

## Per-module conventions

### Cache Checker — a pure decision

```python
def route(f: Path, index: int, ctx: CacheContext, pose_changed: bool = False) \
        -> PoseRenderTask | EmbedRenderTask | CachedHit | Redraw | Retired
```

`route` runs twice for a file that needed a fresh pose: once cold
(→ `PoseRenderTask`), and again on the Poser's `Resolved` — the pose store
is warm by then, so the same table serves the warm-`.npy` shortcut instead
of a re-embed (today's post-resolution check, `classify_stls.py:1148-1155`;
its loss was the regression B's reviewer escalated). `pose_changed` rides
on `Resolved` — the Poser knows the source it just recorded, so the driver
never re-derives it from the store (true when the fresh source is
`vlm`/`siglip`) — and the renders-wanted arm treats it like missing
renders: the redraw is forced even when the render set is complete.

`CacheContext` (shape in data_structures.md) bundles what today is closure
state: the pose store, embeds dir, render index, parsed args. `route`
**reads** caches and **never** writes, renders, or embeds — it is the one
function whose entire behaviour is the decision table in the proposal's
Cache Checker section, and it is trivially unit-testable because the
*whole* decision is in the return value: the redraw path returns
`Redraw(task, hit)` — both halves in one value, so a test of `route` covers
what the driver dispatches (I14). Under §P2.3 the hit carries
`retires=False`: it writes the row, and retirement comes from the child's
`Rendered` ack, so a redraw task that fails sends `Failure` *instead of*
the ack rather than double-retiring (J2). The `--skip-embed` warm paths:
nothing wanted → `Retired` directly (Q2); renders wanted → the plain
`EmbedRenderTask(needs_embed=False)`, which retires on its ack like every
other task (J1 — no `Redraw`-shaped special case, and no `CachedHit` doing
scoring work the flag exists to skip).

### Render child

```python
def run_child(tasks: Transport, results: Transport, cfg: RenderConfig) -> None
```

One loop: `recv` → dispatch on type → `send` result(s); `EndOfInput`
terminates it. Conventions:

* `PoseRenderTask` → `loader.get` → `up_axis_scores` → `renderer.pose_tiles`
  → `PoseTiles(geo_scores, tiles)`. The geometry evidence crosses with the
  tiles because the mesh does not.
* `EmbedRenderTask` → `renderer.views(lm, index, pose.up)` (`index` because
  residency and `Release` key on it, invariant 2; `lm=None` on a resident
  hit) — **`mesh.rotate` on a
  copy of the resident mesh, never in the camera** (I11 resolved
  2026-08-17, reversing this note's draft rule). Measured:
  `eval/views_camera_rotation.py` compared camera-carried rotation against
  `mesh.rotate` over the full view set (3 STLs × 6 ups × 16 views) and
  found max 75/255 differences on half the pixels under the production
  config, against a 2/255 repeat floor — the IBL fill light is
  world-fixed and this Open3D build cannot rotate it, so camera rotation
  changes what the fill illuminates and would shift embeddings under
  existing cache keys. Rotating a *copy* keeps every cache entry valid and
  keeps residency's real win — the parse+load is still saved; the revisit
  pays the ~275 ms re-upload (Spike 3's measurement). The resident
  original is never mutated. `pose_tiles` keeps the camera rotation: the
  pose path has always rendered that way in production, and the pose
  cache's provenance is camera-rotated tiles. `needs_embed=False` → save
  renders, then send
  **`Rendered(file, index)`** — the retirement ack (§P2.3), sent strictly
  **after** `save_renders` returns (K6): "quiescence means the child is
  idle" and the untimed join both rest on the ack being last. The child's
  contract is uniform: exactly one result per task, always.
* `Release(file, index)` clears a resident mesh's `in_flight` flag, dropping
  it to normal LRU eligibility; unknown or already-cleared indices are a
  no-op, which is what lets the parent send it unconditionally (K1).
* `EndOfInput` terminates the loop — and `run_child` then flushes
  stdout/stderr and exits with **`os._exit(0)`**, never by returning (K2):
  interpreter teardown would destroy the `OffscreenRenderer`, and teardown
  is the one thing this repo has a hard constraint about (CLAUDE.md:
  renderers live for the process lifetime, never destroyed — the abort is
  Filament throwing from a destructor). The stdio flush matters because
  `os._exit` skips it and the child's diagnostics are block-buffered on a
  pipe (L4). And note the load-bearing coupling: `os._exit` also skips the
  queue feeder's delivery guarantee, which is safe *only because*
  `EndOfInput` follows quiescence (I1) — by then every result is already
  received. Moving the sentinel earlier breaks both fixes at once.
* **Every exception between `recv` and `send` becomes
  `Failure(file, index, str(e))`** — one bad mesh must not end the run, and
  the file must retire. The child never crashes on a per-file error; it
  crashes only on protocol errors (which are bugs).
* `loader.get(file) -> LoadedMesh` raises on malformed input; the child
  loop is the boundary that converts. The Loader/Renderer seam stays a
  function call — multiple loader workers or a second live renderer attach
  behind these two signatures without the loop changing.

### Poser — never blocks, returns what to do next

```python
class Poser:
    parked: dict[int, ParkedFile]      # continuation state (data_structures.md),
                                       # written only here — the abort pair
                                       # fold_done()/settle() owns its emptying —
                                       # and READ by the driver (P4): the
                                       # quiescence loop and the M4/N1
                                       # subtractions need membership, which is
                                       # why this is an exposed attribute and
                                       # not a bool predicate
    def __init__(self, up_T, down_T, arbiter: Arbiter, record_pose, vlm_cfg): ...
    def on_tiles(self, m: PoseTiles) -> EmbedTilesRequest
    def on_tile_embeds(self, m: TileEmbeds) -> Resolved | None
    def poll(self) -> list[Resolved | Failure]                    # J3; O5
    def fold_done(self) -> int         # abort step 1: fold every ALREADY-resolved
                                       # future. No wait, so no exposure. What it
                                       # leaves in `parked` is the wait's size
    def drop(self, index) -> None      # forget a file: pops the tile stash and
                                       # parked (cancelling its future); no-op on
                                       # unknown. The driver's Failure arm calls
                                       # it so an embed error can't pin ~9 MB of
                                       # tiles forever (C-R1-5). NEVER called
                                       # from fail_outstanding: a parked file's
                                       # in-flight answer is already paid for
                                       # and must still fold (N3, C-R2-2)
    def settle(self, timeout) -> int   # abort step 2: wait out the in-flight calls
                                       # and fold them (I15). Returns how many were
                                       # abandoned to their ensemble pose — the
                                       # abort's closing line, when non-zero
```

* `on_tiles` stashes `(geo_scores, tiles-grid)` keyed by index and returns
  the embed request — it cannot finish without the Embedder.
* `on_tile_embeds` pulls the tensor off the GPU (the one
  `.float().cpu().numpy()`), runs `upright_scores` → `combine_up`, records
  the resolved `Pose` through `record_pose` — **`Done`'s write API, not a
  dict reference** (I9) — and returns `Resolved(file, index)`, or `None`
  having parked the file on a submitted arbiter `Future` (Q1 of the
  data-structures review: the `Future` *is* the transport). The driver
  re-routes every `Resolved` through `route` (the second-call rule there):
  that is where `EmbedRenderTask`, the warm-`.npy` `CachedHit`/`Redraw`,
  and Q2's `Retired` now come from — the Poser decides poses, never cache
  admission.
* `poll() -> list[Resolved | Failure]` is called every driver iteration:
  resolved futures are folded in via `apply_arbiter` semantics; each
  resumed file yields its `Resolved`, re-routed by the driver like any
  other. Failed *calls* keep the ensemble's pose; a fold that
  itself raises yields `Failure` for that file rather than ending the run
  (J3) — `poll` is its own error boundary, because a raise inside it cannot
  be attributed by the driver.
* `fold_done` and `settle` are `poll`'s abort-path siblings and reuse its
  fold, minus the dispatch: a parked file resolved during shutdown wants its
  `record_pose`, not the `Resolved` it would normally yield to a driver
  that is about to flush (C-R2-1). Both inherit `poll`'s error boundary (J3) — a
  fold that raises during abort costs that one file's answer, never the
  flush behind it. `settle` skips futures that `shutdown` already cancelled
  (they raise `CancelledError`); those were queued, never billed.
* **Abandonment is `settle`'s fallback, not the abort policy** (I15
  narrowed): a parked file that is still unanswered when `timeout` expires
  keeps the ensemble pose it recorded at park time. One cross-run
  consequence, accepted deliberately (C-R1-3): that park-time entry
  persists, so an escalation abandoned by an abort reads as sufficient on
  the next run and is not retried — today's unwritten park would have
  retried it. That is the floor, and
  before this ordering it was also the ceiling — see Shutdown for why every
  in-flight answer was being paid for and then discarded.
* The contact sheet is built here — `Image.fromarray` over the grid's first
  column — because arrays are what cross the boundary.

### Arbiter — a windowed pool, not an actor

```python
class Arbiter:
    def __init__(self, workers=8, min_interval=0.0,
                 wrap=None): ...        # wrap: the driver passes
                                        # instrument.arbiter_call (C-R1-2) —
                                        # applied on the worker after the pacing
                                        # sleep, so it times the call, not the
                                        # rate-limit wait
    def submit(self, call: Callable[[], int | None]) -> Future
    def shutdown(self) -> None         # wait=False, cancel_futures=True
```

Rate limiting and windowing live inside `submit`. `shutdown` cancels
**queued** futures (`cancel_futures=True`); calls already running are not
cancellable, and the pool's threads are **non-daemon, joined by
`concurrent.futures`' atexit hook regardless** — so Ctrl-C's residual wait
is up to one in-flight call (~24 s mean, 45 s p95) per worker, in parallel
(I7). Today's comment (`classify_stls.py:1238-1241`) draws exactly this
queued-vs-running distinction, and anyone building on "Ctrl-C is instant"
should know it is "instant except the in-flight calls".

**That atexit join is why the abort path folds rather than abandons.** The
wait is not a cost the design gets to decline — the interpreter pays it
whether or not anything reads the results — so waiting *deliberately*, on
the parked futures, is free in wall-clock and recovers up to
`--arbiter-workers` (default 8) answers already billed at ~$0.30 each. The
one thing `shutdown` must keep doing first is dropping the **queue**:
unbilled work, and left in place a free worker would pick up a new call
mid-abort and extend the very wait being spent. `wait=False` over
`wait=True` for the same reason the driver's constants exist — it hands the
timeout to the caller (`FOLD_S`) instead of surrendering it to a pool that
has no notion of the 300 s transport deadline.

### Embedder — synchronous, owns the GPU

```python
class Embedder:
    text_embeds: torch.Tensor                     # read-only after __init__
    up_T: np.ndarray; down_T: np.ndarray          # handed to Poser at wiring
    front_T: torch.Tensor; back_T: torch.Tensor   # handed to Done for
                                                  #   front_view (D-R1-2)
    def embed_tiles(self, m: EmbedTilesRequest) -> TileEmbeds
    def embed_views(self, m: EmbedViews) -> Embedded
```

Both methods block for the forward pass — in v1 that *is* the pipeline's
pacing, and torch releases the GIL, so the child renders on. `--compile`
wraps `get_image_features` here and nowhere else. If instrumentation ever
shows the parent starving the 4060, a queue goes in front of these two
methods and nothing else changes.

### Done — the only writer, and the owner of retirement

```python
class Done:
    def __init__(self, admission: Admission, text_embeds, cache_ctx,
                 tasks: Transport,             # for Release on retirement (K1)
                 *, categories=None,           # scoring inputs (E-R1-4): absent
                 front_embeds=None,            # under --skip-embed. The banks
                 back_embeds=None): ...        # are the Embedder's front_T/
                                               # back_T (§Embedder, D-R1-2);
                                               # here they take pose.py's
                                               # front_view_index param names
                                               # (E-R2-1)
    def record_pose(self, file: Path, index: int, pose: Pose) -> None
    def on(self, m: CachedHit | Embedded | Failure | Retired | Rendered) -> None
    def flush(self) -> None            # rows CSV + pose cache (temp+replace)
```

* **`Done` owns the canonical pose store** (I9). Concretely (J7): the CLI
  entry loads it via `load_pose_cache`, hands it to `Done` at construction,
  and `Done` owns it from then on — writes, `front_view`, and `flush`,
  which is where the still-open `save_pose_cache` atomicity fix
  (temp + `os.replace`) finally lands. `CacheContext.poses` is *the same
  object*, read-only by convention, so `route` sees this run's resolutions.
  `record_pose` takes `(file, index, pose)` — the Poser has no `root` and
  must not derive identities (Invariant 2); `Done` computes
  `file_identity(file, ctx.root)` itself (J6).
* **`Done` owns `Admission.retired`** (I10), and **retirement is
  idempotent** (J2): `Done` keeps `retired_ids: set[int]` and ignores a
  repeated index, so Invariant 1 is mechanical rather than a convention
  four modules must honour. Without it, a double retirement drives
  `in_flight()` negative and the quiescence loop exits with files still in
  flight — I1's hang inverted into a silent early finish with a
  complete-looking CSV. `admitted` is written only by the driver, `retired`
  only by `Done.on`. `Retired` and `Rendered` retire without a row; a
  `CachedHit` with `retires=False` writes its row without retiring (`rows`
  legitimately has holes — settled in the data-structures review's pass 1).
  On the redraw-failure path a later `Failure` **overwrites** the hit's row
  (K5) — parity with today, where a render failure reports `RENDER_ERROR`
  rather than the cached score.
* **`Done` sends `Release(file, index)` on every retirement** (K1) — it
  holds the `tasks` transport for exactly this. Unconditional, because
  `Done` does not track which files have an unanswered `PoseTiles`; the
  child's no-op on cleared indices makes that free, and FIFO ordering on
  `tasks` means a `Release` can never overtake the task it follows.
  Retirement was chosen as the sender precisely because "whoever retires
  it" — spread over three paths — is how the pinned-mesh leak went
  unnoticed.
* `on` loads the `.npy` for `CachedHit`, scores, resolves `front_view`,
  writes fresh embeddings to the cache. `flush` is **called by the driver
  on the main thread**, on both the drain and abort paths.

### Driver — the loop that is the v1 architecture

```python
@dataclass
class DriverState:                            # the container three passes asked
    admission: Admission                      # for one field at a time (M2, N4,
    admitted_files: dict[int, Path]           # O1): drain and fail_outstanding
    last_progress: float                      # write drv.* attributes, which
    child_failed: bool = False                # Python scoping cannot silently
                                              # rebind as locals. Lives in
                                              # src/driver.py — bookkeeping,
                                              # not a message.

def run(cfg) -> None:
    # As built (wave 2): the CLI does the constructing — it spawns the child
    # (driver.spawn_render_child), builds Done(Admission(), ...), and hands
    # the constructed world in via DriverConfig; run takes done.admission.
    # P2 holds by construction (there is no second source a copy could come
    # from), and a fake child/world is injectable for tests. The lines below
    # keep the original shape for the P2 argument's sake.
    child = spawn_render_child(cfg)
    admission = Admission()                   # ONE instance, handed to Done and
                                              # DriverState alike (P2): the
                                              # container holds the reference,
                                              # it does not own the object.
    drv = DriverState(admission, {},          # stall clock starts at spawn (N4),
                      last_progress=now())    # so the child's Filament/open3d
                                              # startup sits inside the first
                                              # interval instead of being measured
                                              # against a timestamp that does not
                                              # exist yet.
    for index, f in enumerate(walker):
        while not drv.child_failed and \
                drv.admission.in_flight() >= WINDOW:
            drain(block=True)                 # liveness lives inside drain
        if drv.child_failed: break            # (M1), so this gate — where the
                                              # child spends the whole run — is
                                              # covered too. The break sits AFTER
                                              # the gate (O3): fail_outstanding
                                              # fires inside it and releases it
                                              # by retiring, so a break checked
                                              # only at the loop top would admit
                                              # one more file to a known-dead
                                              # child. What a dead child does to
                                              # the walk is a stated trade — see
                                              # "the dead-child walk" below
                                              # (N3, P1).
        drv.admission.admitted += 1
        drv.admitted_files[index] = f
        try:
            dispatch(route(f, index, ctx))    # The warm path's error boundary
        except Exception as e:                # (J3): route stats files the walk
            done.on(Failure(f, index, str(e)))  # cache may list but that
        drain(block=False)                    # vanished, and done.on(hit) loads
                                              # the .npy. An improvement, not
                                              # parity — today a corrupt .npy
                                              # at cache-load ends the run too.
    while drv.admission.in_flight() > 0 \
            or poser.parked:                  # quiescence FIRST: the arbiter
        drain(block=True)                     # tail resolves files long after
                                              # the walker runs dry (I1). The
                                              # parked clause only differs after
                                              # fail_outstanding (N3): retirement
                                              # emptied in_flight, but each fold
                                              # still record_pose()s, so the
                                              # paid answers land before flush —
                                              # bounded by the arbiter's own
                                              # 300 s transport deadline
                                              # (src/pose.py:506, :429 — O4), not by
                                              # anything in this loop. A killed
                                              # child plus a just-submitted call
                                              # can mean five quiet minutes
                                              # before flush and the exit line;
                                              # that tail is not a hang.
    tasks.send(EndOfInput())                  # every Release precedes it (FIFO)
    child.join()                              # UNTIMED: quiescence means idle
    done.flush()                              # (§P2.3); timeout is abort's.
    if child.exitcode != 0:                   # AFTER flush, to stderr, never a
        print(f"child exit {child.exitcode}", # raise (M5, L2): on a clean walk
              file=sys.stderr)                # the run is complete and correct
                                              # here and the exitcode is only a
                                              # diagnostic about HOW the child
                                              # ended; on the N3 path this line
                                              # plus the truncated CSV IS the
                                              # crash report — lost work is
                                              # already Failure rows from
                                              # drain's check.
```

**The dead-child walk, described honestly** (N3, P1): `drv.child_failed`
is set only inside `fail_outstanding`, and the `outstanding()` guard
fires that only when something is owed — so death is *noticed* only when
it matters. Three clauses:

1. A run that needs nothing from the child — every file warm, `route()`
   serving `CachedHit`s — completes normally end to end, however and
   whenever the child died. A child that cannot even initialise Filament
   cannot damage a warm re-run.
2. Otherwise the walk stops within `WINDOW` files of the first file the
   dead child could not serve; the files admitted before the gate filled
   end as `Failure` rows.
3. The CSV therefore ends with up to `WINDOW` `Failure` rows and then
   stops — truncated, not complete — and the stderr exit line is the
   marker.

The front edge is inexact **on purpose**: noticing `child.exitcode`
unconditionally would make it exact at the cost of clause 1, breaking a
warm run on a dead child it never needed — and this project's product is
the caches, with the CSV a thin consumer, so warm-run survival outranks a
crisp edge. The guard is the trade, not an accident.

Driver state has a container: `DriverState` bundles `admission`,
`admitted_files`, the stall clock's `last_progress`, and `child_failed`,
and everything that mutates them does so through `drv.*` (O1). Three
passes found the same bug arriving one field at a time — state called
"driver state" in prose while the code showed the bare binding that makes
it a local of whatever closure writes it (M2, N4, O1) — and that
repetition is the signal the container was missing, not another
declaration: an attribute write cannot be shadowed by Python's scoping
the way a bare name can. `admission` is constructed once in `run` and
handed to both `Done` and `DriverState` — the same instance, never a
copy (P2): `admitted` stays the driver's field, `retired` stays `Done`'s
(I10), and a second `Admission` anywhere yields an `in_flight()` that
never decreases — I1's hang, reached through the container that was added
to stop a scoping bug. The map is **new bookkeeping, deliberately
unpruned** (M2): pruning would need `Done` to call back into the driver
on retirement, the coupling I10 avoided, and unpruned it holds one `Path`
per admitted file — ~1758 at the end of a full run, nothing. Two
subtractions read it, and they differ by exactly the parked set:

* `outstanding()` — the map minus `Done`'s `retired_ids`, **filtered to
  files that still need the child** (M4). The filter reads two things by
  name (N5): the Poser's `parked` dict (data_structures.md's continuation
  state — a citation, not new state) and `cfg.skip_embed`, which makes
  `outstanding()` mode-dependent. Under `--skip-embed` a parked file needs
  nothing further from the child — its ~$0.30 answer will fold and retire
  it as `Retired` on its own — so it is excluded; in every other mode a
  parked file's next step is an `EmbedRenderTask` a dead child cannot
  serve, and failing it is correct. This is **who to fail** when the
  child is gone.
* `child_owed()` — the map minus `retired_ids` minus **all** parked
  files, in every mode (N1). This is **what counts as evidence of child
  liveness**: a wedged child is always holding a task, so a wedge never
  empties it — and a healthy arbiter tail, where the child is idle by
  construction (I1), always does, so the stall clock cannot run against
  arbiter latency.

The hang table, honestly labelled: **death** is closed on all four causes —
the arbiter tail (I1), row-less files (I3/J1), double retirement (J2), a
child dying mid-run (L1/M1) — and the fifth cause, a child that **wedges**
without dying (a Filament stall, an amdgpu reset), is **bounded rather
than accepted** (M3): no progress on work the child owes, past a deadline,
is treated as death — kill the child first, so the untimed join stays
safe, then fail the outstanding files the same way. The deadline runs
against `child_owed()`, never `outstanding()` (N1): gated on
`outstanding()` it would count arbiter latency (24 s mean, **45 s p95** —
LEARNINGS, where a 7-hour run went) as child silence and fire on a
healthy run's quiescence tail — the one state this pipeline enters by
design. `STALL_S` is **~240 s** (N2, O4): the child's unit of work is
**3–28 s per model** (actors_proposal.md:196 — pass 5's 34 ms figure was
a resident re-show, not a model), and the error is one-sided — a wedge is
permanent, so a four-minute detection costs four minutes of a multi-hour
run exactly once, while a false positive kills a healthy child — so the
deadline sits ~8.5× above the documented top of range, a statement about
rendering, never about the network. And deliberately **not** 300 s: that
is the arbiter's transport deadline (src/pose.py:506), an unrelated number
`STALL_S` should not shadow. This repo's renderer has a documented
history of aborting rather than returning; one timestamp is cheap
insurance.

`dispatch` routes everything a decision can produce:
`CachedHit`/`Retired`/`Failure` → `done.on` (the `Failure` arm is what
`poser.poll` yields on a fold error — K3); tasks → `tasks.send`;
`Redraw(task, hit)` → both. `drain` is the routing
table, and **each arm is an error boundary** (I4): the parent runs four
things that raise (the embedder, the poser, future folding, the `.npy`
load), and today both halves of the pipeline convert those to error rows —
an unguarded drain would regress that, and a naive guard would admit
without retiring, which is I1's hang wearing a different hat.

```python
def drain(block):                   # driver state is written as drv.* (O1): an
                                    # attribute write cannot become a shadowing
                                    # local the way N4's bare name did
    while (m := results.recv(SHORT) if block else results.recv_nowait()) \
            is not None:                                   # not truthiness (J8)
        drv.last_progress = now()
        try:
            match m:
                case PoseTiles():
                    out = poser.on_tile_embeds(embedder.embed_tiles(poser.on_tiles(m)))
                    if out is not None: dispatch(out)      # task, or Retired
                case EmbedViews():          done.on(embedder.embed_views(m))
                case Rendered() | Failure(): done.on(m)
        except Exception as e:
            done.on(Failure(m.file, m.index, str(e)))      # retire, never crash
            poser.drop(m.index)                            # and unpin its tiles
                                                           # (C-R1-5; no-op if
                                                           # never stashed)
    for out in poser.poll():        # arbiter answers; poll is its own error
        dispatch(out)               # boundary, yields Failure per file (J3) —
                                    # but dispatch(Resolved) re-routes through
                                    # route(), which raises (vanished file), so
                                    # THIS dispatch gets the same Failure+drop
                                    # guard as the match arms (wave-2 addition).
                                    # NO progress bump here (O2): child_owed()
                                    # already silences the clock in the tail
                                    # (N1), so the bump N4 asked for protected
                                    # nothing — and it let a mid-run fold reset
                                    # a wedged child's deadline.
    if not child_owed():            # nothing owed: the clock has nothing to
        drv.last_progress = now()   # measure (W2-R1-1). Without this, an
                                    # arbiter tail ages the clock past STALL_S
                                    # and the child is killed the instant the
                                    # file un-parks — zero ms to answer the
                                    # task just sent. While the child owes
                                    # anything (the wedge O2 protected), owed()
                                    # is non-empty and no reset happens, so a
                                    # mid-run fold still cannot save a wedge.
    if outstanding() and (child.exitcode is not None or
            (child_owed() and now() - drv.last_progress > STALL_S)):
        fail_outstanding()          # LAST, after the recv loop AND poll (M1):
                                    # a result already in the pipe is consumed
                                    # first and never mis-blamed, and both
                                    # blocking loops route through drain. The
                                    # stall half watches child_owed() (N1); the
                                    # outer outstanding() guard keeps a dead
                                    # child's exitcode from re-firing over an
                                    # empty set every drain (N6), and reads as
                                    # what it is: nothing owed, nothing to do.
```

`fail_outstanding()`: if the child is wedged rather than dead,
`child.kill()` it first (the untimed join must never meet a live wedge —
M3); then `done.on(Failure(f, i, ...))` for each of `outstanding()` —
retirement is idempotent (J2), the `Release`s go to a dead child
harmlessly, and it runs at the end of `drain` where no caller is
mid-iteration over the same indices. It also sets `drv.child_failed`, the
flag that stops the walker admitting to a dead child (N3). Names, so
nobody hunts for them (N6): `STALL_S` is a driver constant beside
`WINDOW`, `SHORT`, and the abort path's `FOLD_S`, and `now()` is
`time.monotonic()`.

(`results.recv` with a short timeout when blocking — the `Transport`
protocol has no iterator, I16 — and every produced task flows through the
one `dispatch`, not two paths.)

## Sequence: one cold file, worst case (arbiter escalation)

```
driver          child                poser          embedder   arbiter    done
  │ route(f)=PoseRenderTask
  ├──send──────▶│ load, geo, tiles
  │             ├──PoseTiles(geo,tiles)──▶ (driver drain)
  │                                  │ on_tiles: stash
  │                                  ├─EmbedTilesRequest─▶│ forward
  │                                  │◀────TileEmbeds─────┤
  │                                  │ combine_up → margin < gate
  │                                  ├──record_pose──────────────────────────▶│
  │                                  ├─────submit(sheet call)────▶│ (24s, async)
  │                                  │ parked[index]=Future       │
  │   ...driver keeps admitting and draining other files...       │
  │                                  │ poll(): Future done ◀──────┘
  │                                  │ fold answer, record_pose ─────────────▶│
  │◀─EmbedRenderTask(pose)───────────┤
  ├──send──────▶│ views (rotated copy; resident hit)
  │             ├──EmbedViews────────▶ (driver drain)
  │                                             │ forward
  │                                        Embedded ──────────────────▶│ score,
  │                                                                    │ row, npy
  │                                                                    │ retired+1
```

The warm path is two hops: `route` → `CachedHit` → `done.on`. The redraw
path is `Redraw`: the hit writes the row (`retires=False`), the child saves
renders and its `Rendered` ack retires — one retirement, after the work
(§P2.3). The `--skip-embed` cold path ends at the Poser when no renders are
wanted: resolution recorded, `Retired` → `done.on`; with `--save-renders`
it continues as a plain `needs_embed=False` task retiring on its ack.

## Shutdown

**Drain** (input exhausted): the driver's epilogue above — quiescence
first, then `EndOfInput`, join, flush.

**Abort** (Ctrl-C): a `stopping` flag the driver checks per iteration.
Order matters and is fixed:

```
arbiter.shutdown()          # FIRST: drop QUEUED calls (unbilled) so no idle
                            #   worker starts new work while we fold (I7)
poser.fold_done()           # free: futures already resolved fold now, with no
                            #   wait between the Ctrl-C and their record_pose
done.flush()                # pose cache first (the artifact that costs money,
                            #   temp+os.replace), then partial rows CSV.
                            #   The run is durable from here on
poser.settle(FOLD_S)        # wait out the in-flight calls and fold them —
                            #   free in wall-clock (the atexit join blocks on
                            #   these same threads regardless). Whatever misses
                            #   FOLD_S abandons to its ensemble pose (I15)
done.flush()                # AGAIN: idempotent, picks up what settle recovered
tasks.close()               # cancel_join_thread: unflushed pickles must not
                            #   hold the process open (I6)
child: daemon, join(timeout) — abandoned renders are debug artifacts
```

**Why the wait is not a concession.** Cancelling the queue and abandoning
the in-flight calls — the obvious shape, and today's — pays for up to eight
VLM answers, waits ~24 s for them at interpreter exit, and then discards
every one: today a Ctrl-C jumps past the deferred-fold loop
(`classify_stls.py:1222-1236`) straight into the `finally`, and even
futures that were *already resolved* die there. Since the atexit join makes
the wait unavoidable, the choice is not "wait or don't" but "read the
results or throw them away". `FOLD_S` sits above the arbiter's 45 s p95 and
well under its 300 s transport deadline (`src/pose.py:506`) — a straggler past
it loses its answer either way, and `FOLD_S` is a driver constant beside
`WINDOW`, `SHORT`, and `STALL_S`.

**Flushing twice is what makes waiting safe.** A single flush behind
`settle` would put the pose cache — the only artifact whose loss costs
money — behind a minute of silence, so a second Ctrl-C would forfeit the
whole run's resolutions to recover eight. Flushing before the wait as well
costs one extra `os.replace` (`flush` is idempotent by contract) and bounds
the second Ctrl-C's damage to exactly the in-flight calls it interrupts.

**The wind-down narrates itself** — the driver does; no module narrates its
own shutdown. A stderr line before each step that can *take* time, which is
`settle` and nothing else: the others are milliseconds or bounded by their
own timeout. The line is printed before the wait, so its count is
`len(poser.parked)` once `fold_done` has run — the dict the driver already
reads by name (P4), no new plumbing:

```
saved 1203 rows + pose cache; folding 6 in-flight VLM calls already paid
for (up to 60s) — Ctrl-C again forfeits only those 6
```

`settle`'s return is the closing line, printed only when it is non-zero:
`N calls did not answer within 60s; those files keep their ensemble pose`.
Silence otherwise — a clean abort should not editorialise.

Load-bearing, not cosmetic, and specifically because the ordering above
already did the hard part: `settle` is the abort's one long silent window,
a quiet process reads as a hung one, and the reflex answer to a hang is the
second Ctrl-C. The line's job is to price that keystroke honestly — the run
is durable, the six are not — which turns it from a panic response into a
choice. A narration that could only say "please wait" would not be worth
the code.

Second Ctrl-C: hard exit. Before `done.flush()` it forfeits the run's
resolutions; after it, only whatever `settle` has not yet folded — which is
the whole point of flushing on both sides of the wait. `flush` is
idempotent and runs on the main thread in a `finally`, exactly as today's
nested-finally chain does.

## Invariants (the contract the tests pin)

1. **Every admitted index retires exactly once** — via `Embedded`, a
   retiring `CachedHit`, `Failure`, `Retired`, or `Rendered` reaching
   `done.on` — and the invariant is **mechanical**: `Done`'s
   `retired_ids` set ignores repeats (J2), so a double retirement cannot
   drive `in_flight()` negative and end the run early. The parent's drain
   arms, the walker loop, and `poser.poll` each convert their own
   exceptions to `Failure`, so this holds under errors, not only on the
   happy path.
2. **`index` is the identity everywhere**; no message or module keys on
   `Path` except the caches, which key on `file_identity`.
3. **Single-writer, with two named exceptions:** the pose store is owned by
   `Done`, written by the Poser only through `record_pose`; `Admission` is
   split `admitted`=driver / `retired`=`Done`, each field single-writer.
   Rows are `Done`'s alone; the scene and residency the Renderer's alone.
4. **The child sends exactly one result per task — no exceptions** (§P2.3):
   `PoseTiles`, `EmbedViews`, `Rendered`, or `Failure`. `Release` and
   `EndOfInput` are **control messages, not tasks** — no result is expected
   for them, which is why this invariant needs no qualification (K1). A
   task that raises sends `Failure` *instead of* its result — the exception
   never crosses raw, and the blanket rule no longer contradicts a
   carve-out.
5. **The parent never blocks on a send** (`tasks` is unbounded — Q1), and
   never does a blocking `recv` while it holds work it could dispatch. No
   module blocks on the Arbiter.

## What this note deliberately does not decide

* Thread placement in the successor (which seam gets a queue first) — that
  is instrumentation's call, and every seam above is already a
  message-shaped function boundary.
* The walker's cache format and `--rescan` plumbing — unchanged from today,
  wrapped behind `walker` as an iterable of paths.
* Whether `classify_stls.py` retains its full arg surface or `src/driver.py`
  grows its own — the runners invoke `classify_stls.py`, so it stays the
  entry point either way (migration notes in the proposal).
