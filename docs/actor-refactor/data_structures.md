# Actor Refactor — Data Structures

Design note, 2026-08-14. Companion to [actors_proposal.md](actors_proposal.md):
that document argues for the boundaries and records what the spikes measured;
this one fixes the shapes, and [interfaces.md](interfaces.md) fixes the
calling conventions between the modules that hold them. Revised the same day against
[docs/reviews/2026-08-14-data-structures.md](../reviews/2026-08-14-data-structures.md)
(findings D1–D15); the review's two gating questions are answered inline —
**Q1: the Arbiter stays a `Future` the Poser holds; Q2: the render child saves
renders itself.**

It describes the form the proposal's own postscript converged on — modules
under `src/`, frozen per-edge message types, a **sequential driver**, and
exactly one process boundary at the Renderer — since the overlap spike showed
that boundary captures essentially everything the nine-thread version could
reach.

Settled while designing these (each with the measurement that settled it):

* **v1 is a single renderer subprocess.** Build it, instrument it, and only
  then thread anything else. Nine threads buy work the card cannot absorb: the
  4060 is ~94% busy behind one child and thermally clamped above that, and the
  GIL spike showed the Python-level stages cannot overlap a render in-process
  anyway.
* **The Embedder stays in the parent.** A process boundary buys escape from a
  GIL-holder, and the renderer was the GIL-holder (85–92% held during
  `render_to_image`). Torch releases the GIL during SigLIP's forward, so
  co-residency with the driver costs nothing measurable. Revisit only if
  instrumenting the built pipeline shows the parent starving the 4060 — the
  module boundary makes putting a queue in front of it a local change.
* **`Pose` lives in `pose.py`**, which already owns `POSE_CACHE_VERSION` and
  the sufficiency rules, and deliberately imports nothing from the pipeline.
* **Transport is `mp.Queue` for v1**, behind a small interface so the measured
  shared-memory variant can drop in. Numbers in [Transport](#transport).

## What is v1 and what is the threaded successor (D15)

The preamble says sequential, so the note must not quietly specify threads.
**v1** is: the message types below, the `Pose`/`ResultRow` shapes, `parked`
(the arbiter deferral is already threaded today via `ThreadPoolExecutor` —
that carries over unchanged), the child's resident dict, and the two boundary
`mp.Queue`s. The driver is a loop; in-process "edges" are function calls.

**Threaded successor only**: the Supervisor's condition variable, bounded
in-process forward edges, and the unbounded-back-edge deadlock rule. They are
kept in this note because the message types are designed not to foreclose
them, and are marked where they appear. Do not build them for v1 — a
sequential loop cannot contend.

## Messages: frozen dataclasses per edge, no `kind` field

The proposal sketched one envelope dict with a `"kind"` discriminator. Split
by type instead: `kind: "embed"` with no pose is an illegal state, and hard
boundaries are the point of the refactor. Routing is `match`/`isinstance`.

`index` is the file's position in the Walker's list. It is what `Done` sorts
by and what the Supervisor counts — the identity, everywhere.

An earlier draft had two field-identical task pairs (`PoseTask`/`EmbedTask`
upstream of `PoseRenderTask`/`EmbedRenderTask`); with the Loader inside the
child there is no *wire* edge between them, so the duplicate pair is
collapsed (D12) — the Cache Checker sends render tasks directly. The Loader
keeps a module seam inside the child regardless; see
[the Loader/Renderer seam](#inside-the-child-the-loaderrenderer-seam).

### Parent → child

```python
@dataclass(frozen=True)
class PoseRenderTask:              # pose unknown → render candidate tiles
    file: Path
    index: int

@dataclass(frozen=True)
class EmbedRenderTask:             # pose resolved → render classification views
    file: Path
    index: int
    pose: Pose
    needs_embed: bool              # False: the redrawn path (D8) — embedding
                                   # cached but a saved render is missing/stale;
                                   # the child renders, saves, returns nothing

@dataclass(frozen=True)
class Release:                     # control message (not a task — no result):
    file: Path                     # clears a resident mesh's in_flight flag.
    index: int                     # Done sends one per retirement,
                                   # unconditionally; the child no-ops on
                                   # cleared or unknown indices (K1)

@dataclass(frozen=True)
class EndOfInput:                  # terminates the child. A message, not None:
    pass                           # recv's None means "nothing arrived yet",
                                   # and a value meaning two things would make
                                   # the child exit on its first idle window
                                   # (interfaces review I5). The child then
                                   # flushes stdio and exits via os._exit(0)
                                   # — interpreter teardown would destroy the
                                   # renderer, the one hard-constraint abort
                                   # (K2/L4)
```

The child owns saving renders in every case (Q2): the pixels are already in
its memory, and `--save-renders` config is passed once at child startup. On
the `needs_embed=False` path the row comes from the accompanying `CachedHit`
(`retires=False`, below) and **retirement from the child's `Rendered` ack**
— the uniform contract from the interfaces review's §P2.3, which is what
keeps admission bounding the child's backlog. Render work is abandoned only
on abort (debug artifacts).

### Child → parent

```python
@dataclass(frozen=True)
class PoseTiles:                   # → Poser
    file: Path
    index: int
    geo_scores: np.ndarray         # up_axis_scores from the child's mesh: the
                                   # mesh never crosses the boundary, so its
                                   # geometry evidence must — the ensemble is
                                   # combine_up(geo, sig) and the Poser holds
                                   # no mesh (found writing interfaces.md; the
                                   # roundtrip spike's child already sent it)
    tiles: list[list[np.ndarray]]  # [candidate][azimuth] — the grid, not a
                                   # flat list (D7): the ensemble reshapes by
                                   # candidate, and n_az just changed 4 → 2

@dataclass(frozen=True)
class EmbedViews:                  # → Embedder
    file: Path
    index: int
    pose: Pose                     # read-only echo for the row's pose columns;
                                   # Done writes front_view through the
                                   # canonical pose dict, not this copy (D9)
    views: list[np.ndarray]

@dataclass(frozen=True)
class Rendered:                    # → Done: the needs_embed=False ack. The
    file: Path                     # child always sends exactly one result per
    index: int                     # task (interfaces pass 2 §P2.3) — this is
                                   # what retires a render-only file, which is
                                   # what keeps admission bounding the child's
                                   # backlog (J1/J2/J4)
```

The arbiter path needs the six first-column tiles as PIL Images for
`make_contact_sheet` (`pose.py:300`): **the Poser converts** with
`Image.fromarray` at sheet-build time — arrays are what cross the boundary.

### Poser ↔ Embedder (the ensemble) — D5

```python
@dataclass(frozen=True)
class EmbedTilesRequest:           # Poser → Embedder
    file: Path
    index: int
    tiles: np.ndarray              # stacked, order-preserving; the Poser keeps
                                   # the [candidate][azimuth] grouping

@dataclass(frozen=True)
class TileEmbeds:                  # Embedder → Poser (back-edge)
    file: Path
    index: int
    embeds: torch.Tensor           # on device; the Poser pulls it off the GPU
```

### Into `Done` — D5

```python
@dataclass(frozen=True)
class CachedHit:                   # Cache Checker → Done: embedding cache hit
    file: Path
    index: int
    pose: Pose
    cache_file: Path               # Done loads the .npy (today's cache-load)
    retires: bool = True           # False on the redraw path: the row comes
                                   # from here, retirement from the child's
                                   # Rendered ack (§P2.3)

@dataclass(frozen=True)
class Embedded:                    # Embedder → Done: fresh embeddings to score
    file: Path
    index: int
    pose: Pose
    embeds: torch.Tensor           # stays on device: Done's scoring matmul
                                   # against the text embeddings runs on the GPU

@dataclass(frozen=True)
class Failure:                     # any stage → Done; becomes a RENDER_ERROR row
    file: Path
    index: int
    error: str

@dataclass(frozen=True)
class Retired:                     # → Done: retire with no row (interfaces
    file: Path                     # review I3/Q2) — the --skip-embed paths,
    index: int                     # where pose resolution was the whole job
```

`Done` holds the category text embeddings (computed once by the Embedder at
startup, read-only thereafter) and does the scoring — `pool_sims`, top-3,
`front_view` resolution.

**The Embedder always returns `torch.Tensor`.** An earlier revision typed the
two messages by consumer (`TileEmbeds` as numpy, `Embedded` as tensor); the
uniform contract won: the Embedder returns what it computes, and conversion
is the consumer's business. `Done` keeps the tensor on device for its
`img_embeds @ text_embeds.T`; the **Poser** does the one
`.float().cpu().numpy()` before handing tiles to the ensemble math. The torch
import that requires lives in `src/poser.py` — `src/pose.py` itself stays
torch-free, receiving plain arrays as it always has (siblings once the
refactor moves it; the no-torch rule was never about the directory). Both
messages live entirely in the parent process, so nothing here is ever
pickled and transport plays no part in the choice. Nor is "on device" a
residency claim for `Done` (R5): it still lands on numpy for
`front_view_index` and the `.npy` cache write — the tensor stays put for the
scoring matmul, not forever.

`Failure` is load-bearing, not a convenience: the Supervisor's
`admitted − retired` counter only reaches zero if errors *retire* files
exactly like successes, so an error must be a message that arrives at `Done`,
not a log line. Today errors are `rows.append({"top1": f"RENDER_ERROR: ..."})`
(`classify_stls.py:1024`, `:1066`) — a malformed row shape that survives only
because `DictWriter` fills missing keys.

### The Arbiter is not a queue-fed actor (Q1, D6)

An earlier draft specified `Arbiter → Poser` both as an unbounded back-edge
queue and as `ParkedFile.future` — incompatibly. The `Future` wins: it is
today's working mechanism (`ThreadPoolExecutor`, `classify_stls.py:976`), the
cheaper build, and the v1 reality. The Arbiter module is a windowed,
rate-limited pool the Poser holds; the back-edge rule in [Queues](#queues)
does not apply to it.

### Driver-side shapes (never cross a queue)

```python
@dataclass(frozen=True)
class Redraw:                      # route's redraw return: both halves of the
    task: EmbedRenderTask          # decision in one value, so a test of route
    hit: CachedHit                 # covers what the driver dispatches (I14)

@dataclass(frozen=True)
class RenderConfig:                # handed whole to the child at spawn — it
    render_size: int               # crosses the spawn boundary, so everything
    views: int                     # here must stay picklable (I13)
    elevations: tuple[float, ...]
    save_renders_dir: Path | None
    render_format: str
    budget_bytes: int
    collection_root: Path

@dataclass
class CacheContext:                # route()'s read-only world: the pose store
    poses: dict                    # (THE object Done owns, not a copy — route
    embeds_dir: Path | None        # must see this run's resolutions), the
    render_index: dict             # render index, and the parsed args the
    args: argparse.Namespace       # cache keys derive from
    root: Path                     # the collection anchor — Done derives
                                   # file_identity from it, so the Poser
                                   # never has to (J6). The ONLY sanctioned
                                   # parent-side root: never re-derive it
                                   # from args (K7). RenderConfig carries
                                   # its own copy because it crosses spawn.
```

## Inside the child: the Loader/Renderer seam

Same process, still a module boundary. `src/loader.py` and `src/renderer.py`
are separate modules with a typed seam between them, even though v1 crosses
it with a plain function call:

```python
@dataclass
class LoadedMesh:
    file: Path
    mesh: o3d.geometry.TriangleMesh
    nbytes: int                    # feeds ResidentMesh accounting

# loader.get(file) -> LoadedMesh   (prefetch, if any, hidden behind it)
# renderer consumes LoadedMesh and owns upload, framing, residency
```

Collapsing the duplicate *message* pair (D12) removed a dead wire format, not
this seam. Keeping the seam is what stays cheap now and pays later: multiple
loader workers attach behind `get()` without the renderer noticing, and a
second live renderer (at another size, say — possible now that the abort is
known to be teardown-only) attaches beside the first without touching the
Loader. Removing it would lock the child into exactly one of each.

## `Pose`

A **frozen** dataclass in `pose.py`, replacing the raw cache-entry dict in
flight. An earlier draft left it mutable "so `Done` can write `front_view`" —
but the copy crossing the process boundary is a copy either way, so mutating
the echo updates nothing that reaches `save_pose_cache` (D9). `Done` writes
through the canonical pose dict with `dataclasses.replace`; freezing costs
nothing the boundary was not already costing.

```python
@dataclass(frozen=True)
class Pose:
    up: tuple[float, float, float]
    confidence: float
    source: str                    # "forced" | "geometry" | "siglip" | "vlm"
    v: int                         # no default: from_cache carries it through,
                                   # fresh resolutions pass POSE_CACHE_VERSION
                                   # explicitly (D10)
    margin: float | None = None
    front_view: dict[str, int] = field(default_factory=dict)   # view_cfg -> index

    @classmethod
    def from_cache(cls, d: dict) -> "Pose": ...   # absorbs legacy *shapes*
    def to_cache(self) -> dict: ...
```

* **The vocabulary is `forced | geometry | siglip | vlm` — decided in the
  review's pass 2 (P2.3-A) and already live in today's code.** What `source`
  records, stated plainly so it is never re-derived: **which tier moved the
  answer**, not which ran. The ensemble runs on every model, so the old
  `"ensemble"` could not mean "the ensemble decided" — it meant "the combined
  pick differed from geometry's"; likewise a paid arbiter call that
  *confirms* the pose leaves the label alone, so `pose_source` undercounts
  arbiter usage. `forced` = the user's `--up-axis`; `geometry` = geometry's
  pick stood; `siglip` = SigLIP moved it off that pick; `vlm` = the arbiter
  moved it off the ensemble's conclusion. Whether the ensemble ran at all
  lives where it always has: `margin is not None` (`pose_is_sufficient`).
  Chosen over the old open question's `"confirmed"` because it keeps one axis
  and stays true for `--no-up-ensemble`. Two carried caveats: `siglip`
  slightly overclaims (a compromise candidate ranked second by both arms can
  win the combined argmax — rare, unmeasured, and the component scores are
  not stored), and the agreement label with `margin: None` is a latent
  overload that is not live (0 of 4092 entries across both caches).
  Mechanically the rename maps old spellings in `load_pose_cache` — the
  refactor's `from_cache` inherits that — with **no version bump**, because a
  bump would re-resolve (and re-bill) unchanged poses.
* **The freeze is shallow, and `Pose` is unhashable** (R4): `front_view` is a
  dict, so `hash(pose)` raises and mutation through the field is still
  possible. Nothing may key on a `Pose`; `index` is the identity, everywhere.
* **`from_cache` absorbs legacy shapes, not versions.** The shapes are real on
  disk: `embed-cache3` holds bare-int `front_view: 0` entries beside
  per-config dicts — today merged at the *write* site
  (`classify_stls.py:1100-1103`, D3) — and `margin` is absent from older
  entries. Version filtering already has a home and keeps it:
  `load_pose_cache` drops mismatched `v` before construction (`pose.py:126`).
  `from_cache` therefore carries `v` through rather than defaulting it — a
  field default of `POSE_CACHE_VERSION` would stamp unversioned entries as
  freshly resolved and silently defeat that drop rule (D10).
* **`pose_is_sufficient` stays a module function over `Pose | None`** (D11):
  it is the miss test, called with a possibly-absent entry
  (`classify_stls.py:964`, `:1003`), and `None → False` is load-bearing.
  Absence is the Cache Checker's dict lookup, not `Pose`'s job.
  `embed_cache_token` can become a method.
* The on-disk pose cache stays JSON dicts; `from_cache`/`to_cache` are the
  only crossing points.

### The embedding token is the up vector (P2.3-B)

`embed_cache_token` is `up_str(pose.up)` for every pose — shipped, with both
live caches migrated. The cache was never keyed on *source*; it is keyed on
what changes the pixels, and only `up` does. The old elision — deterministic
poses keyed as the literal `--up-axis` string — existed to keep a
pre-pose-pipeline cache valid, and it cost real duplication: a forced
`--up-axis z` and an auto run whose geometry resolved to `[0,0,1]` rendered
identical pixels under two keys. Under the honest token they are one entry, a
pose that changes *label* without changing *axis* stops re-embedding, and the
source string leaves the cache key — which is what made P2.3-A a plain rename
instead of a 1531-model re-embed.

The key scheme is stamped in `cache-meta.json` (`CACHE_VERSION`); readers
refuse a cache they cannot read instead of silently missing on every key, and
`migrate_cache_keys.py` re-keys old caches — a rename of `.npy` files, never
a re-embed, stamped only after every move succeeds.

## Rows

```python
@dataclass(frozen=True)
class ResultRow:
    index: int
    file: str
    up: str
    pose_conf: float
    pose_source: str
    front_view: int
    top: tuple[tuple[str, float], ...]     # up to 3 of (category, score)
    def to_csv(self) -> dict: ...

# Failure arrived at Done doubles as the error row — one type, plus to_csv()
```

`Done` holds `rows: dict[int, ResultRow | Failure]` — keyed by `index`, not a
list — and flushes `writerows(rows[i].to_csv() for i in sorted(rows))`.
Deterministic output order regardless of completion order, and partial flush
on abort is the same code path.

**`rows` is the output record, not the retirement record**: under
`--skip-embed` a file retires with no row (`classify_stls.py:1094`), so
`rows` legitimately has holes while `admitted == retired` holds — never
assert `len(rows) == admitted`.

Per the proposal's shutdown rule, `rows` and the pose dict are the durable
state the main thread reads after joining actors with a timeout, so a wedged
actor cannot take the flush down with it.

## Poser continuation state

Replaces the `deferred` list + single-slot `pending_box` (and its
`assert not pending_box` — the dict makes concurrent parks natural, which is
the actual win). The Poser never blocks on the Arbiter: it parks the file and
keeps going; the answer resumes it.

```python
parked: dict[int, ParkedFile]

@dataclass
class ParkedFile:
    file: Path
    resolved: tuple                # (up, ratio, source, margin) — ensemble's answer
    future: Future                 # the in-flight arbiter call (Q1: this IS the
                                   # Arbiter → Poser transport)
```

Pose resolution is never repeated on resume (measured: re-running cost more
than the overlap saved, and two arbiter calls disagreed on three models).

## Supervisor accounting

One counter doing three jobs — admission, quiescence, and bounding the
in-flight window:

```python
@dataclass                         # NOT frozen — the one mutable shape here —
class Admission:                   # with defaults, so run's Admission()
    admitted: int = 0              # constructs (sign-off carry-over).
                                   # admitted: written only by the driver
    retired: int = 0               # written only by Done — once per index,
                                   # whichever of Embedded / CachedHit(retires)
                                   # / Failure / Retired / Rendered arrives;
                                   # repeats ignored via retired_ids (J2)
    def in_flight(self) -> int:
        return self.admitted - self.retired
```

Quiescence — the shutdown signal — is: Walker exhausted **and**
`admitted == retired` **and** no file parked on an arbiter `Future`
(amended per interfaces pass 8, P3). No poison pills; a file can still be
parked on an arbiter call long after the input is exhausted. The counter
and the parked set differ only after the interfaces note's
`fail_outstanding` (its N3 path), where retirement empties `in_flight()`
while paid arbiter answers are still in the air — the parked clause is
what lets each answer fold through `record_pose` before `flush`. On every
healthy run a parked file is un-retired, so the counter alone covers it.

In v1 the counter is a field of `DriverState`, the driver's bookkeeping
container (interfaces.md §Driver, its O1): `admission`, `admitted_files:
dict[int, Path]`, the stall clock's `last_progress`, and `child_failed`,
written as `drv.*` attributes. One `Admission` instance is handed to both
`DriverState` and `Done` — the same object, never a copy (P2). The
`threading.Condition` that blocks `admit` belongs to the threaded successor
(D15).

**One window, two consumers** (D15, revised with interfaces pass 2): the
admission limit is the single knob; the `results` queue depth and the
residency exemption both derive from it. The task queue is unbounded (I2)
and the child's backlog is bounded by **admission itself**: under the
uniform child contract (§P2.3), every task — `needs_embed=False` included —
holds its admission slot until its result or `Rendered` ack returns, so
quiescence genuinely means the child is idle. (The old R7 nuance, where
redraw files retired while their render work was still queued, dissolved
with the carve-out that created it — J5.)

## Renderer-child mesh residency

The proposal's device tier targets the **upload**, and that is still the
number that justifies it (D13): `_upload` is 275 ms on an 800k-triangle
collection STL against 34 ms to re-show a hidden geometry — a miss is ~8× a
hit. What got cheap is only the parse (11–66 ms, numpy parser). The reason the
structure can still be small is the access pattern, not the miss price: the
only revisit in a run is the pose → embed round trip, it exists only on cold
runs (a cached pose sends the file through in one pass), and the roundtrip
spike resolved it to three resident meshes at 88% busy.

```python
@dataclass
class ResidentMesh:
    center: np.ndarray             # framing from _upload, kept to avoid recompute
    radius: float
    nbytes: int
    in_flight: bool                # awaiting a pose answer — exempt from eviction

resident: OrderedDict[str, ResidentMesh]   # keyed by scene name
budget_bytes: int
```

Eviction (D14): LRU — `move_to_end` on every touch, evict from the front
until under `budget_bytes`, matching the proposal's rule (an earlier draft
said FIFO, which with a round trip in flight can drop exactly the mesh about
to come back). `in_flight` entries are never evicted, which makes
`budget_bytes` a *soft* bound: the hard worst case is the admission window ×
the heaviest mesh (~450 MB at 3 × 150 MB) — this is the proposal's
residency-depth-follows-the-admission-window link, restored (D14).

`in_flight` has **two** clears, and both are required (interfaces pass 3
K1): the `EmbedRenderTask` that consumes the mesh, and the `Release`
control message for the paths where no embed render ever follows — a
`Retired` under `--skip-embed` (every file, on that mode), a fold `Failure`
from `poser.poll`, or a drain-arm exception on the pose path (CUDA OOM
being the realistic one, which would otherwise pin a mesh exactly when
memory is tight). Without `Release`, those meshes are exempt from eviction
for the process lifetime and `budget_bytes` cannot reclaim them.

The rotate-into-the-camera rule still applies — residency only pays if the
resident geometry is reusable as-is — and it is a **precondition to verify,
not a settled fact** (interfaces review I11): `R.T` is proven
pixel-identical only for the tile grid at one elevation, the roundtrip
spike that produced the residency numbers *rotated held meshes*, and the
classification views span 8 azimuths × 2 elevations. Building
`renderer.views` includes a pixel-identity check against `mesh.rotate`
across the full view set; residency is inert until that check passes.

## Queues

* v1: two boundary `mp.Queue`s — **`tasks` (parent→child) unbounded,
  `results` (child→parent) bounded at the admission window** (interfaces
  review I2/Q1). An earlier revision bounded both, which closes the
  `Renderer → Poser → Renderer` cycle the deadlock rule exists for:
  `Poser → Renderer` is a listed back-edge, and the overlap spike left it
  unbounded for exactly that reason. Admission is the only forward
  pressure; the parent never blocks on a send; the bounded `results` queue
  is what paces the child. In-process edges are function calls in the
  sequential driver.
* Threaded successor: forward edges bounded, back-edges (`Embedder → Poser`,
  `Poser → Renderer`) **unbounded** — the deadlock rule. Pressure is applied
  only at admission. (`Arbiter → Poser` left this list with Q1: a `Future` is
  not a queue.)

### Transport

`eval/ipc_spike.py` isolated the boundary's transport cost from render-wait by
blasting **the overlap spike's payload** — 24 tiles + 16 views uint8 at
384 px, 17.7 MB/model — with zero render time. Production today is lighter
(D4): `UP_TILE_AZIMUTHS = 2` makes a model 12 tiles + 16 views ≈ 12.4 MB,
30% less, which only strengthens the conclusion.

| transport | per model (spike payload) | throughput |
|---|---|---|
| `mp.Queue` | 13.5–14.3 ms | 1.3 GB/s |
| shm pool, consumed in place | 2.8–3.5 ms | 5.1–6.3 GB/s |
| shm pool, copy-out | 4.7–5.5 ms | 3.2–3.8 GB/s |
| raw pickle, no pipe | 4.1 ms | 4.3 GB/s |

The pipe is the tax, not pickle. But of the overlap spike's 6–8 s parent wait,
only ~0.85 s was transport — the rest is genuine render-wait no transport
removes — so the addressable win is ~0.5% of a cold run. **v1 ships on
`mp.Queue`.** Under the shm variant, `PoseTiles`/`EmbedViews` carry
`(block_id, shapes)` instead of arrays, one free-list queue returns blocks to
the child (a back-edge, so unbounded), and abort must `unlink` the pool.
Triggers for the swap: instrumentation shows the parent's `get()` starving the
4060, or renders move to 2048 px (28.4× the bytes, same 4–5×).

## Unchanged

Pose cache (`file_identity → entry` on disk), embedding cache (`.npy` per
`cache_key`), render index, walk cache — shapes untouched; the refactor moves
who reads and writes them (Cache Checker reads, Done writes). The embedding
*key* did change with P2.3-B above, and the cache root gained
`cache-meta.json`, but both landed in current code before the refactor — the
`src/` split inherits them.

Of the proposal's three atomicity defects, **two have since been fixed in the
current code** (D2): the CSV now flushes inside the `finally` chain that
attempts all three artifacts (`classify_stls.py:1134-1169`), and a torn
`.npy` unlinks itself on `BaseException` so it cannot read as a hit next run
(`classify_stls.py:1086-1092`) — temp + `os.replace` would still be stronger
against SIGKILL, but it is a hardening, not an open hole. What remains open is
the one whose loss costs money: **`save_pose_cache` is a bare `write_text`**
(`pose.py:133-138`); write temp + `os.replace` when `Done` takes it over.
