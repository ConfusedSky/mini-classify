# Actor Refactor — Interfaces

Design note, 2026-08-14. Third of the set: [actors_proposal.md](actors_proposal.md)
argues the boundaries, [data_structures.md](data_structures.md) fixes the
shapes, this one fixes the **calling conventions** — who calls whom, with
what signature, what blocks, and who converts errors into `Failure`. Types
named here are data_structures.md's; where writing this note changed one
(`PoseTiles.geo_scores`), the change is recorded there with credit.

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
src/done.py          scoring, rows, cache writes, flush
src/driver.py        the sequential loop; owns Admission
pose.py              math + Pose + caches (unchanged home)
classify_stls.py     CLI entry: args, run-params, cache guards -> driver
```

Import rules, and why each is load-bearing:

| module | may import | must NOT import | because |
|---|---|---|---|
| child side (`loader`, `renderer`, `render_child`) | open3d, PIL, numpy, `messages`, pose | **torch** | SigLIP lives in the parent; a torch import in the child costs VRAM and startup for nothing |
| `poser` | torch (one conversion), numpy, pose, PIL (contact sheet) | open3d renderer calls | the Poser works on renders, never geometry |
| `embedder` | torch, transformers | — | the only owner of models |
| pose.py | numpy, open3d, PIL | torch, anything in src/ | the standing rule, unchanged |
| `messages` | pose, numpy, torch (annotations) | everything else | shapes only, no behaviour |

(pose.py imports open3d for `up_axis_scores`, so the parent transitively
imports open3d too — import only; no renderer is ever created parent-side.)

## The wiring

```
 PARENT PROCESS                                    CHILD PROCESS
┌────────────────────────────────────────┐        ┌─────────────────────────────┐
│ driver: sequential loop, Admission     │  tasks │ run_child loop              │
│                                        │ ─────▶ │                             │
│ walker ─▶ cache_checker.route(f, i)    │ Pose-  │  loader.get(file)           │
│    │           │            │          │ Render │      │ LoadedMesh          │
│    │      CachedHit    *RenderTask ────┼─Task / │      ▼                      │
│    │           │                       │ Embed- │  renderer.pose_tiles /      │
│    │           ▼                       │ Render │  renderer.views(up in cam)  │
│    │      done.on(msg) ◀───────────────┼─Task───│      │                      │
│    │        ▲    ▲                     │        │  save renders (child owns)  │
│    │        │    │                     │ results│      │                      │
│    │   Embedded  Failure ◀─────────────┼─◀───── │  resident LRU (bytes)       │
│    │        │             PoseTiles /  │ Pose-  └─────────────────────────────┘
│    │   embedder.embed_views  │         │ Tiles(geo,tiles) | EmbedViews | Failure
│    │        ▲                ▼         │
│    │        │      poser.on_tiles ──▶ embedder.embed_tiles
│    │        │           │                   │
│    │  EmbedViews   TileEmbeds ◀─────────────┘
│    │        │           ▼
│    │        │      poser.on_tile_embeds ──▶ EmbedRenderTask (back to tasks)
│    │        │           │ parked?
│    │        │           ▼
│    │        │      arbiter.submit(call) ── Future ──▶ poser.poll()
└────┴────────┴─────────────────────────────────────────────────────────────────┘
```

Everything inside the parent is a **function call** in v1. The two arrows
crossing the box edge are the only queues.

## The boundary protocol

```python
class Transport(Protocol):            # src/transport.py
    def send(self, msg) -> None       # blocks while the queue is full
    def recv_nowait(self):            # a message, or None if empty
    def recv(self, timeout: float):   # a message, or None on timeout
    def close(self) -> None
```

* Two transports, both `mp.Queue` at depth = the admission window:
  `tasks` (parent→child, carries `PoseRenderTask | EmbedRenderTask | None`)
  and `results` (child→parent, carries `PoseTiles | EmbedViews | Failure`).
* **Blocking convention:** `tasks.send` blocking is the *secondary* guard —
  the Admission window keeps in-flight work below the depth, so a blocked
  send means a bug in the window, not normal operation. The parent never
  blocks on `results.recv` while it has other work; the driver uses
  `recv_nowait` in its drain step and a short-timeout `recv` only when the
  window is full and nothing else can proceed.
* **Child lifecycle:** spawned once (`ctx.get_context("spawn")`, daemon),
  handed its config whole (`RenderConfig`: render size, views, elevations,
  save dir/format, `budget_bytes`, collection root) — the child never reads
  argv or run-params. `None` on the tasks queue is end-of-input: the child
  finishes its queue and exits. This sentinel is safe *here*, unlike
  pipeline-wide poison pills, because the parent owns admission and sends it
  only after the last task.
* The shm variant changes only what `send`/`recv` carry (`(block_id,
  shapes)` + a free-list back-queue) — the signatures above do not move.

## Per-module conventions

### Cache Checker — a pure decision

```python
def route(f: Path, index: int, ctx: CacheContext) \
        -> PoseRenderTask | EmbedRenderTask | CachedHit
```

`CacheContext` bundles what today is closure state: the pose dict, embeds
dir, render index, parsed args. `route` **reads** caches and **never**
writes, renders, or embeds — it is the one function whose entire behaviour
is the decision table in the proposal's Cache Checker section, and it is
trivially unit-testable for exactly that reason. The redrawn path returns
`EmbedRenderTask(needs_embed=False)` *and* the driver also sends the
`CachedHit` — `route` signals this by returning the task with
`needs_embed=False`; the driver derives the accompanying `CachedHit` from
the same `ctx` lookup rather than `route` returning two values.

### Render child

```python
def run_child(tasks: Transport, results: Transport, cfg: RenderConfig) -> None
```

One loop: `recv` → dispatch on type → `send` result(s). Conventions:

* `PoseRenderTask` → `loader.get` → `up_axis_scores` → `renderer.pose_tiles`
  → `PoseTiles(geo_scores, tiles)`. The geometry evidence crosses with the
  tiles because the mesh does not.
* `EmbedRenderTask` → `renderer.views(lm, pose.up)` — up-rotation in the
  camera (`R.T`), never `mesh.rotate`, or residency is worthless.
  `needs_embed=False` → save renders, send **nothing**.
* **Every exception between `recv` and `send` becomes
  `Failure(file, index, str(e))`** — one bad mesh must not end the run, and
  the Supervisor's counter needs the file to retire. The child never
  crashes on a per-file error; it crashes only on protocol errors (which
  are bugs).
* `loader.get(file) -> LoadedMesh` raises on malformed input; the child
  loop is the boundary that converts. The Loader/Renderer seam stays a
  function call — multiple loader workers or a second live renderer attach
  behind these two signatures without the loop changing.

### Poser — never blocks, returns what to do next

```python
class Poser:
    def __init__(self, up_T, down_T, arbiter: Arbiter, vlm_cfg): ...
    def on_tiles(self, m: PoseTiles) -> EmbedTilesRequest
    def on_tile_embeds(self, m: TileEmbeds) -> EmbedRenderTask | None
    def poll(self) -> list[EmbedRenderTask]
    def abandon(self) -> list[EmbedRenderTask]      # abort path: keep ensemble poses
```

* `on_tiles` stashes `(geo_scores, tiles-grid)` keyed by index and returns
  the embed request — it cannot finish without the Embedder.
* `on_tile_embeds` pulls the tensor off the GPU (the one
  `.float().cpu().numpy()`), runs `upright_scores` → `combine_up`, writes
  the resolved `Pose` into the canonical pose dict, and either returns the
  `EmbedRenderTask` (done) or **returns `None` having parked the file** —
  the margin gate fired, the arbiter call is submitted, and a `ParkedFile`
  holds the `Future`. Never blocks on it (Q1: the `Future` *is* the
  transport).
* `poll()` is called every driver iteration: resolved futures are folded in
  via `apply_arbiter` semantics and their `EmbedRenderTask`s returned.
  Failed futures keep the ensemble's pose (one bad call must not sink the
  file).
* The contact sheet is built here — `Image.fromarray` over the grid's first
  column — because arrays are what cross the boundary.

### Arbiter — a windowed pool, not an actor

```python
class Arbiter:
    def submit(self, call: Callable[[], int | None]) -> Future
    def shutdown(self) -> None        # cancel_futures, never joins the 24s tail
```

Rate limiting and windowing live inside `submit`. Network backends only.

### Embedder — synchronous, owns the GPU

```python
class Embedder:
    text_embeds: torch.Tensor                     # read-only after __init__
    up_T: np.ndarray; down_T: np.ndarray          # handed to Poser at wiring
    def embed_tiles(self, m: EmbedTilesRequest) -> TileEmbeds
    def embed_views(self, m: EmbedViews) -> Embedded
```

Both methods block for the forward pass — in v1 that *is* the pipeline's
pacing, and torch releases the GIL, so the child renders on. `--compile`
wraps `get_image_features` here and nowhere else. If instrumentation ever
shows the parent starving the 4060, a queue goes in front of these two
methods and nothing else changes.

### Done — the only writer

```python
class Done:
    def on(self, m: CachedHit | Embedded | Failure) -> None   # retires exactly once
    def flush(self) -> None            # rows CSV + pose cache (temp+replace)
```

`on` is where retirement happens: it increments `Admission.retired` for
every message, success or `Failure`. It loads the `.npy` for `CachedHit`,
scores, resolves `front_view` (writing through the canonical pose dict via
`replace`), writes fresh embeddings to the cache. `flush` is **called by
the driver on the main thread**, on both the drain and abort paths — the
durable-state-outside-the-actors rule.

### Driver — the loop that is the v1 architecture

```python
def run(cfg) -> None:
    child = spawn_render_child(cfg)
    for index, f in enumerate(walker):
        admission.wait_below(WINDOW)          # v1: drain until below
        admission.admitted += 1
        dispatch(cache_checker.route(f, index, ctx))
        drain(block=False)
    tasks.send(None)                          # end of input
    while admission.admitted > admission.retired:
        drain(block=True)                     # quiescence, not poison pills
    done.flush(); child.join(timeout)
```

`drain` is the whole routing table, and the only place message types meet
module calls:

```python
def drain(block):
    for m in results:                         # recv_nowait / short timeout
        match m:
            case PoseTiles():  dispatch(poser.on_tile_embeds(
                                   embedder.embed_tiles(poser.on_tiles(m))))
            case EmbedViews(): done.on(embedder.embed_views(m))
            case Failure():    done.on(m)
    for task in poser.poll():                 # arbiter answers
        tasks.send(task)
```

`dispatch(None)` is a no-op (a parked file). Note the chain in the
`PoseTiles` arm is three synchronous calls — the seams exist so the
threaded successor can cut any of them, not because v1 needs indirection.

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
  │                                  ├─────submit(sheet call)────▶│ (24s, async)
  │                                  │ parked[index]=Future       │
  │   ...driver keeps admitting and draining other files...       │
  │                                  │ poll(): Future done ◀──────┘
  │                                  │ apply answer → pose dict
  │◀─EmbedRenderTask(pose)───────────┤
  ├──send──────▶│ views (R.T, resident hit: 34ms)
  │             ├──EmbedViews────────▶ (driver drain)
  │                                             │ forward
  │                                        Embedded ──────────────────▶│ score,
  │                                                                    │ row, npy
  │                                                                    │ retired+1
```

The warm path is two hops: `route` → `CachedHit` → `done.on`. The redrawn
path is the warm path plus a fire-and-forget `EmbedRenderTask(needs_embed=
False)` whose renders the child saves silently.

## Shutdown

**Drain** (input exhausted): the driver's own epilogue above — stop
admitting, sentinel the child, drain to quiescence, `done.flush()`, join.

**Abort** (Ctrl-C): a `stopping` flag the driver checks per iteration.
Order matters and is fixed:

```
arbiter.shutdown()          # cancel futures; queued calls are ~24s each
poser.abandon()             # parked files keep their ensemble pose (already
                            #   in the pose dict — nothing to do but drop)
done.flush()                # pose cache first (the artifact that costs money,
                            #   temp+os.replace), then partial rows CSV
child: daemon, join(timeout) — abandoned renders are debug artifacts
```

Second Ctrl-C: hard exit. `flush` is idempotent and runs on the main
thread in a `finally`, exactly as today's nested-finally chain does.

## Invariants (the contract the tests pin)

1. **Every admitted index retires exactly once** — via `Embedded`,
   `CachedHit`, or `Failure` reaching `done.on`. No other path touches
   `retired`.
2. **`index` is the identity everywhere**; no message or module keys on
   `Path` except the caches, which key on `file_identity`.
3. **Single-writer rules:** the pose dict is written by Poser (resolution)
   and Done (`front_view`) only; rows by Done only; the scene and residency
   by the Renderer only; nothing writes another module's state.
4. **The child sends exactly one result per task**, except
   `needs_embed=False` (zero) and `None` (terminates). A task that raises
   sends `Failure` — the exception never crosses raw.
5. **No module blocks on the Arbiter**, and the parent never does a blocking
   `recv` while it holds work it could dispatch.

## What this note deliberately does not decide

* Thread placement in the successor (which seam gets a queue first) — that
  is instrumentation's call, and every seam above is already a
  message-shaped function boundary.
* The walker's cache format and `--rescan` plumbing — unchanged from today,
  wrapped behind `walker` as an iterable of paths.
* Whether `classify_stls.py` retains its full arg surface or `src/driver.py`
  grows its own — the runners invoke `classify_stls.py`, so it stays the
  entry point either way (migration notes in the proposal).
