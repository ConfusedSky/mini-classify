# Actor Proposal

## Proposal

Right now there is a lot of things that can be done at the same time that we
currently aren't doing. I propose that we build out a series of 9 actors that
each handle a stage of the process. This also should help simplify things
because right now there is a huge mess of intertwined processes and
conditionals, without many **hard** boundaries to break things up.

The RTX 4060 sits at ~30% utilization and the CPU lower still, so there is real
idle time to reclaim. Note that `nvidia-smi` utilization is
percent-of-time-a-kernel-is-resident, not occupancy — 30% means the card is idle
70% of the time.

That idle is **not** render and embed contending for one card. Measured: Open3D
brings up EGL headless on `/dev/dri/renderD129`, which is `amdgpu` — the Phoenix1
integrated GPU. The 4060 is `renderD128`/`nvidia`. Rendering is hardware
accelerated, just on the iGPU, and the 30% on the 4060 is SigLIP by itself. See
[Devices](#devices). Mesh loading is not part of the gap either: at the time of
writing `MeshPrefetcher` overlapped it (see Loader below). *(As built: the
prefetcher is gone. The numpy STL parser took the parse to 11–66 ms and the
render child loads each mesh inline — see [What the spikes
changed](#what-the-spikes-changed) and the migration notes.)*

The structural win stands on its own. Hard stage boundaries are what make it
possible to swap SigLIP for another embedder, or Filament for something we
control, without touching the rest of the chain. If the spikes come back flat on
wall-clock we keep the boundaries and drop the threads — see [Fallback](#fallback).

## Design

Each of these actors should have their own message queues and be implemented in
their own files. These new files should live under src/ instead of just being
loose files. Each actor should also be it's own thread, it's possible it might
need to even be it's own process, dependent on followup spikes (see
[Spike 4](#spike-4-gil-behaviour)).

The nine are eight pipeline stages plus a `Supervisor`. The original list had
eight because scoring and CSV writing had no home; those now belong to `Done`,
and the `Supervisor` owns admission, quiescence and shutdown.

Messages carry the file and everything resolved about it so far, so no stage
reaches back into another's state:

```
{"file": Path, "kind": "pose" | "embed", "index": 42,
 "pose": {"up": [0,0,1], "confidence": 0.13, "source": "siglip",
          "margin": 0.81, "v": 4} | None,
 "mesh": <handle> | None}
```

`index` is the file's position in the Walker's list. It is what `Done` sorts by,
and it is the identity the `Supervisor` counts.

### Walker

* If the walk cache exists and `--rescan` was not passed, load it, prune the
  files that have vanished since the scan, and send them through to `Cache
  Checker's` queue.
* Otherwise walk the directories, sending each file through to `Cache Checker's`
  queue as it is discovered, and write the walk cache when the walk completes.

Streaming-as-discovered is the only real difference between the two paths — once
it is a queue, "all at once" and "one at a time" are the same thing.

### Cache Checker

Order matters here. The embedding cache key includes the resolved pose token
(`cache_key` takes `up_token`, from `pose.embed_cache_token`), so under
`--up-axis auto` the embedding cache **cannot be consulted until the pose is
known**. Only a forced `--up-axis z|y` can go straight to the embedding key.

* When a file comes in through the queue:
  * Check the pose cache first. On a miss, send the file with `"kind": "pose"` to
    the `Loader's` queue and stop.
  * On a hit, compute the embedding cache key from the resolved pose token, then
    check the embedding cache.
    * Embedding cached and no renders needed → send **straight to `Done`** with
      the cached embedding. It is not dropped: a cache hit still produces a CSV
      row, and "reruns with new categories skip rendering/embedding entirely" is
      the whole point of that cache.
    * Otherwise send the file with `"kind": "embed"` plus the pose metadata to
      the `Loader's` queue.

"Renders needed" is the existing `need_renders` rule: `--save-renders` is set and
either the pose just changed or some `<render_key>_view<i>` is missing from the
render index.

### Loader

*(As built, none of this section's knobs exist: the render child loads each
mesh inline and keeps a byte-budgeted resident LRU — one number,
`RenderConfig.budget_bytes`. Read the rest of the section as the reasoning
that produced that, not as the shape of the code; the migration notes carry
the mapping.)*

* Uses up to [loader_worker_count=4] workers to load meshes.
* Passes along kind and pose metadata to the `Renderer`.
* Two tiers:
  * **Host tier** — up to [loader_host_cache=4GiB] of decoded meshes in RAM.
  * **Device tier** — up to [loader_device_cache=2GiB] of meshes already uploaded
    to the renderer, kept resident by name and hidden rather than destroyed.

The device tier is the interesting half — though its premise shrank after this
was written. "`_upload` is recorded as the single largest per-model cost"
rested on a 15 s figure that `19e5033` corrected to 275 ms, fifteen minutes
after this document's last edit (`_upload` was never a separately instrumented
stage). What survives is the 34 ms re-show against 369 ms remove+re-add below —
real, but a much smaller prize than the one this section was designed around.

**It is expressible against Open3D**, which is a correction to an earlier draft
of this document. `show_geometry(name, False)` hides *without* freeing —
`clear_geometry()` and `remove_geometry()` both destroy the buffers, but a hidden
geometry stays registered (`has_geometry()` is still True,
`geometry_is_visible()` is False). Measured:

| operation                                 | time   |
|-------------------------------------------|--------|
| `show_geometry(True)` + render            | 34 ms  |
| `remove_geometry` + `add_geometry` + render| 369 ms |

So the device tier is: add each mesh under a unique name, hide instead of
clearing, and keep an LRU of resident names. ~11× on any mesh revisited, with no
renderer of our own required.

**The only revisit in a run is the pose → embed round trip.** A file renders its
candidate tiles, gets posed, and comes back for the embed renders. That means the
LRU pays on *cold* runs — the ones that take hours — and buys nothing on a warm
one, where a cached pose sends the file straight through as `"kind": "embed"` in
a single pass. Worth being honest about the population rather than quoting 11×
against the whole run.

An earlier draft argued the loader tail from "33% of a median model and 55% of
a p99 one (0.83 s / 15.4 s)" — the exact sentence `19e5033` deleted from
`MeshPrefetcher`'s docstring as wrong in both directions. With the numpy parser
a mesh loads in 11–66 ms and `mesh-wait` is 0.4% of wall, so the
`loader_worker_count` case is dead; see
[What the spikes changed](#what-the-spikes-changed).

**The rotation is the blocker, and it is now the whole design risk.** Hiding and
re-showing only helps if the resident geometry is reusable *as-is*, and today it
is not: the embed render mutates vertices via
`mesh.rotate(rotation_to_z_up(...))` before `render_views`, so the geometry left
resident by the pose pass is the *unrotated* mesh. Re-showing it renders the
wrong pose and saves nothing. The fix is already proven in this codebase —
`render_up_candidate_grid` does one upload and carries the rotation into the
camera with `R.T` rather than rotating the mesh six times, verified
pixel-identical. Move the rotation to the camera and the revisit costs 34 ms
instead of 369 ms *plus* a mesh reload. Skip it and the LRU is inert on precisely
the path that would use it.

Two things remain unresolved and gate on
[Spike 3](#spike-3-mesh-ownership-and-the-two-tier-loader):

1. **How deep the LRU has to be.** For a non-escalating file the revisit distance
   is short. For the ~20% that hit the arbiter it is ~24 s of other work, so the
   LRU must still hold those meshes when they come back — otherwise the escalated
   files, already the slow ones, also pay full re-upload. Depth is therefore a
   function of how many files the `Supervisor` admits into the arbiter window:
   the same counter as admission and quiescence, for the third time. Each entry
   also carries the framing (`center`, `radius`) that `_upload` returns, since
   that is computed from the mesh and would otherwise have to be recomputed.
2. **Whether 4GiB of host tier is even reachable.** A 4M-triangle mesh is
   roughly 150 MB in Open3D's representation (double-precision points and
   normals, int32 triangles — to be confirmed by the spike), so 4GiB is ~27 heavy
   meshes, and four workers cannot usefully run that far ahead of a consumer that
   eats one at a time. Read the cap as a memory *bound* rather than a tuning
   knob, the same way `MeshPrefetcher`'s depth was; the useful depth is set by p99
   load time against downstream work, not by bytes. *(As built, this is the one
   surviving knob and it reads exactly that way: `RenderConfig.budget_bytes`, a
   soft byte bound on the child's resident meshes, with the hard worst case set
   by the admission window instead.)*

### Renderer

* Receives a loaded mesh plus metadata signifying what kind of render this is
* If the `"kind": "pose"` then each image required to handle posing is rendered
* If the `"kind": "embed"` then each of the images according to the pose data
  and the input parameters are generated.
* At this point if `--save-renders` is specified the renders are saved to disk
* Pose renders are then sent to `Poser` and embed renders are sent to `Embedder`.

Single thread, single `OffscreenRenderer`, created on the Renderer's own thread
rather than lazily on main. An earlier draft called one renderer per process a
hard limit; the review measured otherwise (`docs/reviews/2026-08-13.md` §3.1):
four renderers at four sizes were created and used correctly in one process,
and the abort is **teardown only** — Filament throws from a destructor when a
renderer is destroyed, which is why `eval/tile_and_vlm.py` keeps its four
alive deliberately. So the render stage is serial by *choice* — one renderer,
kept for the process lifetime, never destroyed — not by construction, and
mixed render sizes in one process only need a second live renderer.

The Renderer owns the device-tier LRU, since the resident names live in its
scene. The Loader decides what should be resident; the Renderer is what holds it.

### Poser

* Handles posing the way it currently works, if needs arbitration the results
  are sent to `Arbiter`
* Request embedding from the `Embedder` to handle the ensemble.
* Once posing is complete send back to `Renderer` with `"kind": "embed"`.

**The Poser never blocks on the Arbiter or the Embedder.** It keeps
continuation state keyed by file and goes on consuming its queue; an answer
arriving on its inbox resumes that file. Blocking per-message would throw away
the entire overlap the deferral was built for — a network arbiter averages 24 s
against 3-28 s of local work for a whole model.

Pose resolution is never repeated when the arbiter answer comes back. Re-running
it would redo the mesh load, the 24 candidate renders and their embeddings, and
ask the arbiter a second time; that measured worse than the overlap saved, and
two independent calls disagreed on three models.

### Arbiter

* Handles queuing up async requests to the arbiter, appropriately windows and
  times requests to not be rate limited
* Once the arbitration is complete the result is passed back to the `Poser`.

Network backends only — gemini and claude. **ollama is out of scope**; it has
caused too many problems and it contends with the Embedder for the card
(measured: 10.1 s SigLIP reload against 0.49 s of inference). `--pose-vlm ollama`
should either be dropped or run in a serialized mode that never overlaps the
Embedder.

"Windows" rather than "batches" on purpose: no backend accepts more than one
contact sheet per call.

### Embedder

* Embeds an image then passes it either back to `Poser` to complete the ensemble
  or sends it to `Done`

Owns SigLIP and the category text embeddings, which it computes once at startup.

### Done

*Altitude note (2026-08-13): in practice the pipeline's product is the caches
— querying happens interactively in `test_categories.py`, and the CSV scoring
here is an afterthought. The `src/` split should treat scoring as one thin
consumer beside `test_categories`, not as logic that lives inside a
first-class pipeline stage; `Done`'s load-bearing jobs are the cache writes,
row collection and shutdown flush.*

* Scores the embeddings against the category text embeddings (`pool_sims`,
  top-3), resolves `front_view`, and builds the row
* Saves the embedding to the `.npy` cache and the resolved pose to the pose cache
* Receives error messages from any stage and turns them into `RENDER_ERROR` rows
* Sorts rows by `index` and writes `results.csv`

Sorting is required — and is a fix, not just parity: with deferral on (the
default) the ~20% of files that hit the arbiter are processed after the main
loop, so their rows already land out of file order at the end of today's CSV.
Eight concurrent actors would merely make the disorder total.

### Supervisor

* Owns admission: how many files may be in the pipeline at once
* Counts `admitted − retired`, where a file retires by reaching `Done` — by
  success **or** by error
* Owns shutdown (below)

## Cycles and deadlock

The graph has two cycles: `Poser ↔ Embedder` (the ensemble) and
`Renderer → Poser → Renderer` (the `"kind": "embed"` return). With bounded queues
and no reservation this deadlocks: Renderer blocks pushing to Embedder, Embedder
blocks pushing to Poser, Poser blocks pushing to Renderer.

The rule: **back-edges are never blocked.** `Embedder → Poser`,
`Arbiter → Poser` and `Poser → Renderer` are unbounded, or re-entrant work
takes priority over new work. Forward pressure is applied at the front instead —
the `Supervisor` only admits a file when there is room for it, which is the same
counter that detects quiescence.

## Devices

There are **two** GPUs here, and the pipeline already straddles both:

| stage      | device                          | memory                       |
|------------|---------------------------------|------------------------------|
| Renderer   | AMD Phoenix1 iGPU (`renderD129`)| shared system RAM            |
| Embedder   | RTX 4060 (CUDA)                 | 2.5 GB peak of 7.8 GB VRAM   |
| Arbiter    | network                         | —                            |

(The 4060 is also `renderD128`, which is how the EGL probe identified the
cards — but the Embedder reaches it through CUDA (`/dev/nvidia*`), not the DRM
render node. 2.2 GiB is the weights alone; 2.5 GB is measured peak allocated.)

Consequences:

* **Render and embed do not contend.** Different devices entirely, so overlapping
  them is close to free. This is the strongest argument in the proposal and it
  was discovered by accident.
* **The 4060 holds SigLIP and nothing else.** The VRAM budget is not tight; there
  is no renderer allocation on that card to account for.
* **The device tier is host memory under another name.** The iGPU draws from
  shared system RAM, so `loader_device_cache` and `loader_host_cache` come out of
  the same pool and compete for the same bandwidth. Size them together, not
  independently.

Moving rendering onto the 4060 is **not** a flag flip. Forcing the NVIDIA EGL
vendor library fails at `eglInitialize` and then core-dumps — NVIDIA headless
wants `eglQueryDevicesEXT` + `EGL_PLATFORM_DEVICE_EXT` rather than a default
display, and Filament's GL backend does not ask that way. Reaching the discrete
card means Filament's Vulkan backend or a renderer of our own, which folds into
the roll-our-own question rather than standing apart from it.

And it is genuinely unclear that we *want* to. iGPU-render + dGPU-embed is free
cross-device parallelism; consolidating onto the 4060 reintroduces exactly the
contention we turn out not to have.
[Spike 2](#spike-2-the-device-split) measures both rather than assuming the
discrete card wins.

## Shutdown

**Poison pills do not work here.** A file can still be circulating in
`Poser → Arbiter → Poser` long after the Walker is exhausted, so an
end-of-stream token flowing downstream declares the pipeline finished while work
is in flight. End-of-stream is *quiescence*: the Walker is exhausted **and**
`admitted − retired == 0`. Same counter as admission control — one mechanism
solves both problems.

Two paths:

**Drain** (input exhausted): stop admitting → let the cycles empty → flush →
write → join.

**Abort** (Ctrl-C, fatal error): a shared `stopping` event that every actor
checks between messages. No new work starts, and the `Arbiter` **drops** its
queue rather than draining it — queued calls are ~24 s each, so joining them
hangs Ctrl-C for minutes with nothing to show for it. In-flight calls are a
different matter: they are billed already, and their non-daemon threads are
joined at interpreter exit whether or not anyone reads the results, so the
abort folds them instead of abandoning them — flush, wait out the calls,
flush again. Files still unanswered at the deadline keep the ensemble's pose.
A second Ctrl-C is a hard exit, and after the first flush it costs only the
calls being waited on (the ordering is fixed in interfaces.md § Shutdown).

**Flush runs on the main thread, not in the actors.** Durable state lives in
structures the `Supervisor` can read (the pose dict under a lock, the row list);
actors are daemon threads joined with a timeout, so one wedged actor cannot take
the pose cache down with it. This preserves today's guarantee that an interrupted
run keeps its (expensive) pose resolutions.

Three defects were listed here; all three are now closed:

* ~~Ctrl-C loses the CSV entirely~~ — **fixed**: the write now runs inside the
  `finally` chain that attempts all three artifacts even when another's write
  raises (`main:classify_stls.py:1134-1169`, and `Done.flush` on this branch).
* ~~**`save_pose_cache` is a bare `write_text`**, and it runs on every shutdown
  including Ctrl-C. A kill mid-write corrupts the most expensive artifact we
  have — re-resolving it means 24 candidate renders per model plus ~$0.30 of
  Gemini calls.~~ — **landed** (`src/done.py`, `Done.flush`): the pose cache is
  written to `pose-cache.json.tmp` and `os.replace`d, first of the two writes,
  with a `finally` that unlinks the temp so a failed replace strands nothing
  (E-R1-2/E-R1-3). `pose.save_pose_cache` still exists for the evals and is
  still a bare `write_text`; the pipeline no longer goes through it.
* ~~`np.save` to the embedding cache is non-atomic~~ — **handled**: the write
  unlinks its file on `BaseException`, so a truncated `.npy` cannot pass the
  `.exists()` check next run (`classify_stls.py:1086-1092`). Temp +
  `os.replace` would still be stronger against SIGKILL.

## Spikes

In the order that de-risks fastest. **Spikes 1 and 4 are done** — results below,
full write-ups in `LEARNINGS.md`. What they found changes two of this document's
conclusions; see [What the spikes changed](#what-the-spikes-changed).

### Spike 1: baseline instrumentation — DONE

`--instrument PATH` (`instrument.py`) records exclusive per-stage timing plus
CPU / NVIDIA / amdgpu utilization. Full collection, 602 models at 384px, 2121 s:

*(As built, this table spans two processes. `mesh-load`, `pose-geometry`,
`pose-render`, `view-render` and `save-renders` are timed in the render child
and shipped back on `EndOfInput` — the child times, the parent samples, and the
report prints the child's stages as their own table because a separate process
overlaps the parent's wall clock rather than consuming it. `mesh-wait` and
`arbiter-wait` have no successors: the first was the prefetcher's, the second
is `results-wait` now. F-7.)*

| stage | % wall | | after the STL parser landed (104-model subset) |
|---|---|---|---|
| pose-embed | 29.3% | | 41.7% |
| pose-render | 18.7% | | 15.9% |
| mesh-wait | **18.6%** | | **0.4%** |
| cache-save (SigLIP forward) | 16.8% | | 24.4% |
| view-render | 13.7% | | 12.0% |
| embed (preprocessing) | 1.8% | | 2.4% |

Four things fell out of it:

* **Nothing was ever saturated.** CPU 15%, NVIDIA 44%, AMD 26% on the baseline —
  which is the strongest version of this proposal's premise, and it survived.
* **`mesh-wait` was 18.6%, not the 3.2% an 8-model sample suggested.** One
  prefetch thread at depth 2 does not keep up with the real collection.
* **The numpy STL parser removed it** — 746 ms → 10 ms per model — which is most
  of what `loader_worker_count` was for, without any actor machinery.
* **The bottleneck is now SigLIP on the 4060**, 68.5% of the run against 27.9%
  for rendering, and `pose-embed` alone costs 1.6× what embedding the
  classification views costs.

### Spike 2: the device split — LARGELY ANSWERED

Coresidency is answered — the two stages are on different GPUs and do not
contend — and the overlap spike (`eval/overlap_spike.py`, LEARNINGS
2026-08-13) answered the half that mattered: keep the split. Overlapping the
two devices is worth a real 1.17–1.21×, and consolidating onto the 4060 would
now be doubly wrong — it reintroduces the contention we don't have *and* the
card is thermally clamped with no headroom to absorb a second workload. What
is left, if ever needed, is render throughput iGPU-vs-4060 at the real run
config (`--render-size 2048 --views 8 --elevations 20,-20`):

* Render throughput on the Phoenix1 iGPU versus the RTX 4060, if Filament can be
  reached on the 4060 at all (Vulkan backend, or our own renderer).
* What iGPU rendering costs in system memory bandwidth, since the Loader's host
  tier is competing for it.
* Whether consolidating onto the 4060 is a net loss once SigLIP has to share it —
  the split we have may already be the right answer.

### Spike 3: mesh ownership and the two-tier loader

The named-LRU device tier, on real STLs rather than a synthetic mesh. Measure:

* **Whether the rotation moves to the camera cleanly.** Everything else here is
  worthless if it does not. `render_up_candidate_grid` is the precedent; confirm
  it holds for the embed render across all six candidate ups.
* **Whether render time degrades as hidden geometry accumulates.** The 34 ms
  figure is at small depth. If Filament walks the scene graph per frame there
  may be a per-resident cost that only shows up at depth 20 with 4M-triangle
  meshes — that would quietly undo the win, and it is a ten-minute measurement.
* **Bytes per mesh** across the collection's size distribution, host-side and
  resident, to confirm what 4GiB and 2GiB actually hold.
* **Re-measure `_upload` — DONE, same session.** The 15 s figure did not
  survive: `19e5033` measured 275 ms on a real collection mesh, in line with
  what the 159k-triangle sphere extrapolated (0.01 s upload, 8 ms/view at
  512 px → ~0.25 s). Nothing should be designed around upload cost.
* **Peak RSS** holding N meshes at 4 workers, and the hold-vs-reload tradeoff
  across the pose round trip now that re-showing is 34 ms.

This is the most expensive spike of the four and the one with the largest
upside.

### Spike 4: GIL behaviour — DONE

The question was whether Filament's render call releases the GIL. **It does
not.** `py-spy record --gil` on a real run puts 99% of GIL-held time on the main
thread, with `_shoot`'s line 139 alone at 62% of it; splitting that line
(`eval/renderer_gil.py`) shows `render_to_image` holding the GIL for ~85–92% of
its 36–61 ms, while `np.asarray` is free and `Image.fromarray` releases.

Open3D's mesh reader *does* release it — the prefetch thread took 1% of GIL time
while being 33% of all samples, which is exactly why `MeshPrefetcher` worked.
(It is gone as built: with the parse at 11–66 ms there was nothing left to hide,
and the child loads inline.)

So the split is: actors doing native GIL-free work (mesh loading, torch, the
network arbiter) overlap fine; actors doing **Python-level** work — Cache
Checker, Poser bookkeeping, `Done`'s scoring, `Supervisor` accounting — are
starved while a render is in flight. At ~40 renders per model that was ~1.25 s
per model of held GIL.

ModernGL holds it too, ~3–5× less (~4–9 ms per view against ~21–26 ms), so
"several contexts per process" is **not** parallel rendering — see
`docs/actor-refactor/renderer_alternatives.md`.

## What the spikes changed

Two conclusions in this document moved, in opposite directions.

**The Loader shrank.** `loader_worker_count` and the host tier were sized against
a parse that turned out to be replaceable: the numpy STL parser took `mesh-wait`
from 18.6% of wall to 0.4%, which is most of what multiple loader threads were
for. The two-tier design still stands on the *device* tier — the resident-mesh
LRU, which targets `_upload`, not the parse — but the host tier is now a memory
bound with little left to buy.

**"The Renderer must be its own process" became optional rather than forced.**
Three findings were converging on it; the review then voided two — the
one-renderer-per-process abort is teardown-only, and 384 pose tiles *can*
coexist with 2048 view renders via a second live renderer (§3.1) — leaving the
GIL result as the one that stands. ModernGL loosens that too (3–5× less GIL),
which turns a forced architectural move into a budgeting question. The budgeting then resolved in
the boundary's favour: 12.6 MB per view was a 2048 px number, and at the
production 384 px a view is ~440 KB — the overlap spike measured the parent
waiting on the queue just 6–8 s in a ~2-minute run, so the boundary is nearly
free.

**What did not change:** nothing is saturated, so overlap still has room
everywhere. But the target moved. The proposal was written against a run where
rendering and loading dominated; after the parser it is 68.5% SigLIP on the
4060, and the largest single item — `pose-embed`, 24 tiles per model through a
1.1B-param tower over near-silhouettes — is an eval question rather than an
architecture one (`OPEN_QUESTIONS.md`). Fix that first and the pipeline this
document describes is arranging a much smaller amount of work.

**The overlap was then measured, and it caps this document** (2026-08-13,
`eval/overlap_spike.py` + `eval/siglip_bench.py`, LEARNINGS "Overlap and the
thermal ceiling"). One renderer child process feeding SigLIP through a bounded
queue — the minimal version of the Renderer boundary proposed here — takes the
4060 from ~57% to ~94% busy and delivers a true 1.17–1.21× on cold-run
wall-clock. The gap to the Amdahl 1.45× is thermal, not architectural: the
80 W card trades clocks for duty cycle (2250 → ~1400 MHz saturated), so the
same embed work runs 1.4–1.6× slower once it is back-to-back. SigLIP itself
has no software headroom (batch-size-flat, `torch.compile` disqualified by
drift against the closest decision margin). The consequence for this proposal:
**the one process boundary captures essentially everything the nine-actor
version could reach for throughput.** The remaining levers are cutting
`pose-embed`'s tile count (an eval question) and cooling/power (a hardware
question). The structural case in [Fallback](#fallback) — modules, message
types, a sequential driver, plus exactly this render boundary — is now the
whole recommendation rather than the consolation prize.

The remaining unknown — the pose → embed cycle across the boundary — was then
measured (`--modes roundtrip`, LEARNINGS 2026-08-13): 1.11× against overlap's
1.21× in the all-cold worst case, 88% busy, with the Loader/Poser residency
question resolving to a three-mesh dict in the child and the unbounded
back-edge deadlocking nothing. Warm runs skip the cycle for every pose-cached
file, so the real number sits between the two. Nothing architectural remains
to de-risk.

## Fallback

If wall-clock comes back flat, the structural win is still worth having, in a
much smaller form: split the stages into modules under `src/` with explicit
message types and a **sequential** driver, no threads. Same boundaries, same
testability, none of the deadlock or shutdown complexity — then thread only the
stages Spike 1 says pay for it. This does not foreclose the full version.

## Migration notes

* `run_classify.sh` and the other runners invoke `python classify_stls.py`
  directly and **need no updating** — verified 2026-08-18 against the built
  branch: `classify_stls.py` is still the entry point, now the CLI rather than
  the pipeline, and every flag the runners pass (`--out`, `--cache-dir`,
  `--elevations`, `--render-size`, `--views`, `--pose-vlm`, `--save-renders`)
  is still parsed there. The `src/` layout is behind that door, which is the
  point of keeping the door.
* `pose.py` deliberately never imports `classify_stls` — no rendering or model
  code in it. It moved to `src/pose.py`; the rule held (CLAUDE.md).
* Existing flags to map: `--arbiter-workers` becomes the `Arbiter's` window
  (it did — `Arbiter(workers=...)`); `--no-defer-arbiter` loses its meaning
  entirely, since non-blocking deferral is structural here (retired, C-R1-4).
  `--prefetch` was to become `loader_worker_count` / `loader_host_cache` /
  `loader_device_cache`; **none of those exist as built.** The Loader shrank to
  an inline `loader.get` in the render child (below), so there are no loader
  workers to count and no separate host tier to bound, and the one residency
  knob left is `RenderConfig.budget_bytes` — which is not a flag either: it
  follows the admission window (`classify_stls.RESIDENT_BUDGET_BYTES`,
  data_structures.md §residency). `--prefetch` is accepted and inert, kept so
  the runners' command lines keep parsing.
* Retired outright (2026-08-17): `--no-up-ensemble` and `--up-conf` — the
  ensemble always runs; C-R1-1 showed the off-mode had no home in the new
  protocol (no message carries the bit, the geometry-only `needs_arbiter`
  gate had nowhere to live) and the flag was chosen for retirement over
  implementation. `--pose-vlm ollama` — `VlmConfig` rejects it at
  construction; the pool has no inline arm and ollama+SigLIP never share
  the 4060 (CLAUDE.md). A serialized-inline ollama mode can return as its
  own design item.
* `--up-axis z|y` is the one path that can skip the pose cache lookup, and the
  Cache Checker should keep that shortcut.
