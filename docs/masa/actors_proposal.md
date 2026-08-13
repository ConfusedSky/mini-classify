# Actor Proposal

## Proposal

Right now there is a lot of things that can be done at the same time that we
currently aren't doing. I propose that we build out a series of 9 actors that
each handle a stage of the process. This also should help simplify things
because right now there is a huge mess of intertwined processes and
conditionals, without many **hard** boundaries to break things up.

The GPU sits at ~30% utilization and the CPU lower still, so there is real idle
time to reclaim. Note that `nvidia-smi` utilization is percent-of-time-a-kernel-
is-resident, not occupancy — 30% means the card is idle 70% of the time, which
is the gap this is meant to close. Mesh loading is *not* part of that gap:
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
  * **Device tier** — up to [loader_device_cache=2GiB] of meshes already
    uploaded to the GPU, handed to the `Renderer` ready to draw.

The device tier is the interesting half. `_upload` is ~15 s on a 4M-triangle
mesh against ~0.15 s per view — it is the single largest per-model cost, and
today it sits squarely on the Renderer's critical path. Uploading ahead is how
that comes off it.

Loading is 33% of a median model and 55% of a p99 one (0.83 s / 15.4 s), it is
disk+CPU while everything after it is GPU, and Open3D's reader releases the GIL.
`MeshPrefetcher` already overlaps it — but with a *single* thread at depth 2. On
a median model that hides the load completely (0.83 s against ~1.7 s of
downstream work); on a p99 model the 15.4 s load exceeds the ~12.6 s behind it,
so a run of heavy meshes outruns one loader. That tail is what
`loader_worker_count` is for, not a claim that disk is an unmeasured bottleneck.

Three things are unresolved here and gate on
[Spike 3](#spike-3-mesh-ownership-and-the-two-tier-loader):

1. **The device tier presupposes we own the upload path.** It is not expressible
   against Open3D at all: the upload is `renderer.scene.add_geometry` inside
   `_upload`, owned by the Filament scene on the Renderer's thread, and
   `clear_geometry()` evicts the previous mesh. There is no "keep these N
   resident and pick one" against that API. So it is host-tier-only with Open3D,
   or both tiers with a renderer we control — the two-tier design and the
   roll-our-own question are one decision, not two.
2. **Hold vs reload across the pose round trip.** "Hold until the Renderer
   releases" pins a mesh across pose → arbiter → embed-render. With a network
   arbiter at ~24 s and ~20% of files escalating, that pins multi-GB meshes for
   half a minute each. The current code reloads instead, deliberately: "a
   4M-triangle mesh costs more memory than the reload costs time." A device tier
   changes that arithmetic — the file comes back and its mesh is still on the
   card — but only if the geometry is reusable *as-is*. Today it is not: the
   embed render mutates vertices via `mesh.rotate(rotation_to_z_up(...))` before
   `render_views`. Our own renderer would apply the rotation as a model matrix at
   draw time instead. That trick is already proven here —
   `render_up_candidate_grid` does one upload and carries the camera with `R.T`
   rather than rotating the mesh six times, verified pixel-identical.
3. **Whether 4GiB of host tier is even reachable.** A 4M-triangle mesh is
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
rather than lazily on main. Filament is not something we drive from two threads.

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

## GPU budget

With ollama out, three consumers become two, both resident:

| consumer                | footprint     |
|-------------------------|---------------|
| Embedder                | ~2.2 GiB      |
| Renderer + device tier  | up to ~2 GiB+ |
| Arbiter                 | network only  |
| **budget**              | **8 GiB**     |

The Renderer and the Loader's device tier are **one allocation, not two**: the
Renderer draws from resident geometry the Loader uploaded rather than uploading
its own. Counting them separately would put us at 2.2 + 2 + 2 = 6.2 GiB and
spend most of the headroom for nothing.

That still leaves room, but note the Renderer figure is not enforced by anything
today — it is whatever the largest mesh happens to be, and it scales with
`--render-size`. [Spike 2](#spike-2-gpu-coresidency-and-overlap) measures the
distribution at the real run config rather than trusting one number.

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

### Spike 2: GPU coresidency and overlap

Filament and SigLIP resident together at the real run config
(`--render-size 2048 --views 8 --elevations 20,-20`). Measure peak VRAM across
the mesh size distribution, and whether concurrent render+embed actually beats
taking turns or the driver just serializes them.

### Spike 3: mesh ownership and the two-tier loader

Host tier only with Open3D uploading on the Renderer's thread, versus host +
device tier on a renderer we control. Measure:

* Bytes per mesh across the collection's size distribution, to confirm what
  4GiB of host tier and 2GiB of device tier actually hold.
* Peak RSS holding N meshes at 4 workers, and the hold-vs-reload tradeoff across
  the pose round trip.
* Whether upload can overlap Filament work at all under Open3D — if it cannot,
  the device tier is the only way to get `_upload` off the critical path, and
  the roll-our-own decision is made for us.
* Whether the rotation can move to a model matrix at draw time, so a mesh
  resident from the pose render is reusable for the embed render.

`_upload` is ~15 s on a 4M-triangle mesh against ~0.15 s per view, so this
dominates whatever it touches. It is the most expensive spike of the four and
the one with the largest upside.

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
