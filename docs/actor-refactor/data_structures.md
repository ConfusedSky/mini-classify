# Actor Refactor — Data Structures

Design note, 2026-08-14. Companion to [actors_proposal.md](actors_proposal.md):
that document argues for the boundaries and records what the spikes measured;
this one fixes the shapes. It describes the form the proposal's own postscript
converged on — modules under `src/`, frozen per-edge message types, a
**sequential driver**, and exactly one process boundary at the Renderer — since
the overlap spike showed that boundary captures essentially everything the
nine-thread version could reach.

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

## Messages: frozen dataclasses per edge, no `kind` field

The proposal sketched one envelope dict with a `"kind"` discriminator. Split it
instead: `kind: "embed"` with `pose: None` is an illegal state, and hard
boundaries are the point of the refactor. Routing is `match`/`isinstance` on
type.

`index` is the file's position in the Walker's list. It is what `Done` sorts
by and what the Supervisor counts — the identity, everywhere.

```python
# src/messages.py
@dataclass(frozen=True)
class PoseTask:                    # pose unknown → needs candidate tiles
    file: Path
    index: int

@dataclass(frozen=True)
class EmbedTask:                   # pose resolved → needs classification views
    file: Path
    index: int
    pose: Pose                     # non-optional, by construction

@dataclass(frozen=True)
class Failure:                     # any stage → Done; becomes a RENDER_ERROR row
    file: Path
    index: int
    error: str
```

`Failure` is load-bearing, not a convenience: the Supervisor's
`admitted − retired` counter only reaches zero if errors *retire* files exactly
like successes, so an error must be a message that arrives at `Done`, not a log
line. Today errors are `rows.append({"top1": f"RENDER_ERROR: ..."})` scattered
through `process()` — a malformed row shape that only works because
`DictWriter` tolerates missing keys.

### Across the process boundary

Meshes never cross it. The child receives a path (~100 bytes), loads the mesh
itself, and ships back only pixels — ~440 KB per view at production 384 px
against ~150 MB for a heavy mesh. The Loader is therefore not an inter-actor
edge at all; it is a prefetch inside the child (and mostly vestigial since the
numpy parser took the load to ~10 ms).

```python
# parent → child
@dataclass(frozen=True)
class PoseRenderTask:
    file: Path
    index: int

@dataclass(frozen=True)
class EmbedRenderTask:
    file: Path
    index: int
    pose: Pose

# child → parent
@dataclass(frozen=True)
class PoseTiles:                   # → Poser
    file: Path
    index: int
    tiles: list[np.ndarray]

@dataclass(frozen=True)
class EmbedViews:                  # → Embedder
    file: Path
    index: int
    pose: Pose                     # echoed through so Done needn't look it up
    views: list[np.ndarray]
```

## `Pose`

A dataclass in `pose.py`, replacing the raw cache-entry dict in flight. The
argument that decided it: the legacy-shape handling gets one home. Today the
bare-int `front_view` fallback is inlined at its read site
(`classify_stls.py:1101`), `pose_is_sufficient` inspects `v` and `source` on a
raw dict, and every reader does `.get("margin")` defensively. `from_cache`
becomes the single place old shapes are absorbed; everything downstream sees
one guaranteed shape.

```python
@dataclass
class Pose:
    up: tuple[float, float, float]
    confidence: float
    source: str                    # "forced" | "geometry" | "vlm" | "ensemble"
    margin: float | None = None
    v: int = POSE_CACHE_VERSION
    front_view: dict[str, int] = field(default_factory=dict)   # view_cfg -> index

    @classmethod
    def from_cache(cls, d: dict) -> "Pose": ...   # absorbs legacy shapes here
    def to_cache(self) -> dict: ...
```

* **Not frozen.** `Done` writes `front_view` after scoring; exactly one stage
  writes poses by design, and freezing would force a copy on the hottest write
  path for no safety.
* The on-disk pose cache stays JSON dicts; `from_cache`/`to_cache` are the only
  crossing points. `pose_is_sufficient` and `embed_cache_token` move to
  methods (or take `Pose`).

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

# ErrorRow is Failure arrived at Done — same type, plus to_csv()
```

`Done` holds `rows: dict[int, ResultRow | Failure]` — keyed by `index`, not a
list — and flushes `writerows(rows[i].to_csv() for i in sorted(rows))`.
Deterministic output order regardless of completion order, and partial flush on
abort is the same code path. Per the proposal's shutdown rule, `rows` and the
pose dict are the durable state the main thread reads after joining actors with
a timeout, so a wedged actor cannot take the flush down with it.

## Poser continuation state

Replaces the `deferred` list + single-slot `pending_box` (and its
`assert not pending_box` — the dict makes concurrent parks natural, which is
the actual win). The Poser never blocks on the Arbiter: it parks the file and
keeps consuming its queue; the answer resumes it.

```python
parked: dict[int, ParkedFile]

@dataclass
class ParkedFile:
    file: Path
    resolved: tuple                # (up, ratio, source, margin) — ensemble's answer
    future: Future                 # the in-flight arbiter call
```

Pose resolution is never repeated on resume (measured: re-running cost more
than the overlap saved, and two arbiter calls disagreed on three models).

## Supervisor accounting

One counter doing three jobs — admission, quiescence, and bounding how much
work is in flight:

```python
class Admission:
    admitted: int
    retired: int                   # incremented by Done: success OR Failure
    cv: threading.Condition        # admit blocks while admitted - retired >= limit
```

Quiescence — the shutdown signal — is: Walker exhausted **and**
`admitted == retired`. No poison pills; a file can still be circulating in
`Poser → Arbiter → Poser` long after the input is exhausted.

## Renderer-child mesh residency

The proposal's byte-budgeted device-tier LRU with framing capture, depth
analysis, and a host tier was resolved smaller by the roundtrip spike: three
resident meshes in a dict held 88% busy. But a bare count cap has wildly
variable footprint — three meshes can be single-digit MiB or ~450 MiB
(a 4M-triangle mesh is ~150 MB) — so bound the thing actually being spent:

```python
@dataclass
class ResidentMesh:
    name: str
    center: np.ndarray             # framing from _upload, kept to avoid recompute
    radius: float
    nbytes: int

resident: OrderedDict[str, ResidentMesh]   # FIFO: evict from the front until
budget_bytes: int                          # under budget; never evict in-flight
```

Same five lines of logic as the count-3 dict. A miss is cheap now anyway —
~10 ms parse plus re-upload — so nothing more elaborate pays. The only revisit
in a run is the pose → embed round trip, and only on cold runs; a warm run's
cached pose sends the file through in a single pass.

The rotate-into-the-camera precondition from the proposal still applies:
residency only pays if the resident geometry is reusable as-is, so the embed
render must carry the up-rotation in the camera (`render_up_candidate_grid`'s
proven `R.T` pattern), not by mutating vertices.

## Queues

* Forward edges: bounded. Back-edges (`Embedder → Poser`, `Arbiter → Poser`,
  `Poser → Renderer`): **unbounded** — this is the deadlock rule. Pressure is
  applied only at admission.
* The process boundary: bounded both ways at depth 4 (the spike's config),
  behind a transport interface.

### Transport

`eval/ipc_spike.py` isolated the boundary's transport cost from render-wait by
blasting the real payload (24 tiles + 16 views uint8 at 384 px, 17.7 MB/model)
with zero render time:

| transport | per model | throughput |
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
4060, or renders move to 2048 px (29× the bytes, same 4–5×).

## Unchanged

Pose cache (`file_identity → entry` on disk), embedding cache (`.npy` per
`cache_key`), render index, walk cache — shapes untouched; the refactor moves
who reads and writes them (Cache Checker reads, Done writes) and adds the three
atomicity fixes from the proposal (temp + `os.replace` for the pose cache and
`.npy` writes, partial CSV flush on Ctrl-C).
