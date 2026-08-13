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
[Devices](#devices). Mesh loading is not part of the gap either:
`MeshPrefetcher` already overlaps it (see Loader below).

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
 "pose": {"up": [0,0,1], "confidence": 0.13, "source": "ensemble",
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
either the pose just changed or some `<stem>_view<i>` is missing from the render
index.

### Loader

* Uses up to [loader_worker_count=4] workers to load meshes.
* Passes along kind and pose metadata to the `Renderer`.
* Two tiers:
  * **Host tier** — up to [loader_host_cache=4GiB] of decoded meshes in RAM.
  * **Device tier** — up to [loader_device_cache=2GiB] of meshes already uploaded
    to the renderer, kept resident by name and hidden rather than destroyed.

The device tier is the interesting half. `_upload` is recorded as the single
largest per-model cost, and it sits squarely on the Renderer's critical path —
though how large is itself in question now (see
[Spike 3](#spike-3-mesh-ownership-and-the-two-tier-loader)).

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

Loading is 33% of a median model and 55% of a p99 one (0.83 s / 15.4 s), it is
disk+CPU while everything after it is GPU, and Open3D's reader releases the GIL.
`MeshPrefetcher` already overlaps it — but with a *single* thread at depth 2. On
a median model that hides the load completely (0.83 s against ~1.7 s of
downstream work); on a p99 model the 15.4 s load exceeds the ~12.6 s behind it,
so a run of heavy meshes outruns one loader. That tail is what
`loader_worker_count` is for, not a claim that disk is an unmeasured bottleneck.

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
   knob, the same way `MeshPrefetcher`'s depth is; the useful depth is set by p99
   load time against downstream work, not by bytes.

### Renderer

* Receives a loaded mesh plus metadata signifying what kind of render this is
* If the `"kind": "pose"` then each image required to handle posing is rendered
* If the `"kind": "embed"` then each of the images according to the pose data
  and the input parameters are generated.
* At this point if `--save-renders` is specified the renders are saved to disk
* Pose renders are then sent to `Poser` and embed renders are sent to `Embedder`.

Single thread, single `OffscreenRenderer`, created on the Renderer's own thread
rather than lazily on main. This is not a style choice: **one
`OffscreenRenderer` per process** is a hard limit — a second one does not fail
politely, Filament's resource manager throws from a destructor and the
interpreter aborts (LEARNINGS). Rendering cannot be threaded *or*
multi-instanced within a run, only split across processes. So the render stage
is a serial resource by construction, and everything else is arranged around it.

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

* Scores the embeddings against the category text embeddings (`pool_sims`,
  top-3), resolves `front_view`, and builds the row
* Saves the embedding to the `.npy` cache and the resolved pose to the pose cache
* Receives error messages from any stage and turns them into `RENDER_ERROR` rows
* Sorts rows by `index` and writes `results.csv`

Sorting is required: today rows come out in file order, and eight concurrent
actors would otherwise produce an arbitrary order run to run.

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

| stage      | device                          | memory                    |
|------------|---------------------------------|---------------------------|
| Renderer   | AMD Phoenix1 iGPU (`renderD129`)| shared system RAM         |
| Embedder   | RTX 4060 (`renderD128`)         | ~2.2 GiB of 8 GiB VRAM    |
| Arbiter    | network                         | —                         |

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
checks between messages. No new work starts, in-flight work is abandoned, and the
`Arbiter` **drops** its queue rather than draining it — queued calls are ~24 s
each, so joining them hangs Ctrl-C for minutes with nothing to show for it. Their
files keep the ensemble's pose. A second Ctrl-C is a hard exit.

**Flush runs on the main thread, not in the actors.** Durable state lives in
structures the `Supervisor` can read (the pose dict under a lock, the row list);
actors are daemon threads joined with a timeout, so one wedged actor cannot take
the pose cache down with it. This preserves today's guarantee that an interrupted
run keeps its (expensive) pose resolutions.

Three defects to fix while we are in here — all present in the current code, not
introduced by this design:

* **Ctrl-C loses the CSV entirely.** The write sits *after* the `finally`, so
  `KeyboardInterrupt` propagates past it: the caches survive, the results do not.
  `Done` should flush partial rows on abort.
* **`save_pose_cache` is a bare `write_text`** and runs on every shutdown
  including Ctrl-C. A kill mid-write corrupts the most expensive artifact we
  have — re-resolving it means 24 candidate renders per model plus ~$0.30 of
  Gemini calls. Write to temp + `os.replace`.
* **`np.save` to the embedding cache is non-atomic.** A truncated `.npy` still
  passes the `.exists()` check and comes back as a cache hit next run. Same fix.

## Spikes

In the order that de-risks fastest.

### Spike 1: baseline instrumentation

Per-stage timing through the current `process()` on a real run, attributing the
idle 70% across render, embed and arbiter-tail. Loading is already known-
overlapped except on heavy meshes, so it is not a suspect. This is the only thing
that gives "worth it performance wise" a denominator.

### Spike 2: the device split

Coresidency is mostly answered — the two stages are on different GPUs and do not
contend. What is left is whether the current split is the right one, at the real
run config (`--render-size 2048 --views 8 --elevations 20,-20`):

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
* **Re-measure `_upload`.** The 15 s figure for a 4M-triangle mesh does not
  survive a first look: a 159k-triangle sphere uploads in 0.01 s and renders in
  8 ms/view at 512 px, which extrapolates to ~0.25 s rather than 15 s. One
  synthetic mesh is not enough to overturn a recorded measurement, but the gap is
  wide enough that the real number has to be established before anything is
  designed around it.
* **Peak RSS** holding N meshes at 4 workers, and the hold-vs-reload tradeoff
  across the pose round trip now that re-showing is 34 ms.

This is the most expensive spike of the four and the one with the largest
upside.

### Spike 4: GIL behaviour

Confirm Open3D's reader, Filament's render call and torch all release the GIL.
The reader is documented to; if **Filament does not**, threads buy no overlap and
the actors need to be processes — which changes message passing from queues to
something with serialization. Cheapest of the four and the only one that can
invalidate the architecture, so run it early.

## Fallback

If wall-clock comes back flat, the structural win is still worth having, in a
much smaller form: split the stages into modules under `src/` with explicit
message types and a **sequential** driver, no threads. Same boundaries, same
testability, none of the deadlock or shutdown complexity — then thread only the
stages Spike 1 says pay for it. This does not foreclose the full version.

## Migration notes

* `run_classify.sh` and the other runners invoke `python classify_stls.py`
  directly and need updating for the `src/` layout.
* `pose.py` deliberately never imports `classify_stls` — no rendering or model
  code in it. Keep that rule when it moves.
* Existing flags to map: `--prefetch` becomes `loader_worker_count` /
  `loader_host_cache` / `loader_device_cache`; `--arbiter-workers` becomes the
  `Arbiter's` window;
  `--no-defer-arbiter` loses its meaning entirely, since non-blocking deferral is
  structural here.
* `--up-axis z|y` is the one path that can skip the pose cache lookup, and the
  Cache Checker should keep that shortcut.
