# Actor Refactor — Data Structures

Design note, 2026-08-14. Companion to [actors_proposal.md](actors_proposal.md):
that document argues for the boundaries and records what the spikes measured;
this one fixes the shapes. Revised the same day against
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
child there is no edge between them, so the pair is collapsed (D12) — the
Cache Checker sends render tasks directly.

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
```

The child owns saving renders in every case (Q2): the pixels are already in
its memory, and `--save-renders` config is passed once at child startup. On
the `needs_embed=False` path the file's CSV row comes from the `CachedHit`
sent to `Done` alongside (below), which is also what retires it — the child's
outstanding render work is drained by the child join at shutdown, and
abandoned on abort (debug artifacts only).

### Child → parent

```python
@dataclass(frozen=True)
class PoseTiles:                   # → Poser
    file: Path
    index: int
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
    embeds: np.ndarray
```

### Into `Done` — D5

```python
@dataclass(frozen=True)
class CachedHit:                   # Cache Checker → Done: embedding cache hit
    file: Path
    index: int
    pose: Pose
    cache_file: Path               # Done loads the .npy (today's cache-load)

@dataclass(frozen=True)
class Embedded:                    # Embedder → Done: fresh embeddings to score
    file: Path
    index: int
    pose: Pose
    embeds: torch.Tensor

@dataclass(frozen=True)
class Failure:                     # any stage → Done; becomes a RENDER_ERROR row
    file: Path
    index: int
    error: str
```

`Done` holds the category text embeddings (computed once by the Embedder at
startup, read-only thereafter) and does the scoring — `pool_sims`, top-3,
`front_view` resolution.

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
    source: str                    # "forced" | "heuristic" | "ensemble" | "vlm"
    v: int                         # no default: from_cache carries it through,
                                   # fresh resolutions pass POSE_CACHE_VERSION
                                   # explicitly (D10)
    margin: float | None = None
    front_view: dict[str, int] = field(default_factory=dict)   # view_cfg -> index

    @classmethod
    def from_cache(cls, d: dict) -> "Pose": ...   # absorbs legacy *shapes*
    def to_cache(self) -> dict: ...
```

* The `source` literals are the ones the code writes: `resolve_up` initialises
  `"heuristic"` and moves it to `"ensemble"` or `"vlm"`; the forced path
  writes `"forced"` (D1 — an earlier draft listed a `"geometry"` value that
  appears in neither cache on disk, and would have made `from_cache` reject
  every entry).
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
class Admission:
    admitted: int
    retired: int                   # incremented by Done: success OR Failure
```

Quiescence — the shutdown signal — is: Walker exhausted **and**
`admitted == retired`. No poison pills; a file can still be parked on an
arbiter call long after the input is exhausted.

In v1 this is a plain counter the driver loop consults; the
`threading.Condition` that blocks `admit` belongs to the threaded successor
(D15).

**One window, three consumers** (D15): the admission limit is the single
knob. The child's task-queue depth and the residency exemption below both
derive from it — the roundtrip spike ran the same window as `--inflight 3`
with queue depth 4, and nothing was learned from them differing.

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

The rotate-into-the-camera precondition from the proposal still applies:
residency only pays if the resident geometry is reusable as-is, so the embed
render must carry the up-rotation in the camera (`render_up_candidate_grid`'s
proven `R.T` pattern), not by mutating vertices.

## Queues

* v1: the two boundary `mp.Queue`s (both directions, depth = the admission
  window). In-process edges are function calls in the sequential driver.
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
who reads and writes them (Cache Checker reads, Done writes).

Of the proposal's three atomicity defects, **two have since been fixed in the
current code** (D2): the CSV now flushes inside the `finally` chain that
attempts all three artifacts (`classify_stls.py:1134-1169`), and a torn
`.npy` unlinks itself on `BaseException` so it cannot read as a hit next run
(`classify_stls.py:1086-1092`) — temp + `os.replace` would still be stronger
against SIGKILL, but it is a hardening, not an open hole. What remains open is
the one whose loss costs money: **`save_pose_cache` is a bare `write_text`**
(`pose.py:133-138`); write temp + `os.replace` when `Done` takes it over.
