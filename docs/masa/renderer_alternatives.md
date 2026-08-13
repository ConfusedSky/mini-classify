# Renderer Alternatives — Open3D, raylib, ModernGL

Research note, 2026-08-12. Measured on this machine (RTX 4060 Laptop + AMD
Phoenix1 iGPU, Open3D 0.19.0 CUDA build) against `test-stls/bunny.stl`
subdivided to 4,444,864 triangles.

**Read [Topology](#topology-matters-more-than-triangle-count) before trusting any
upload number here.** Upload cost tracks vertex count, not triangle count, and a
real STL has ~4× the vertices of the subdivided mesh most of these figures use.

Two questions started this: can we stage mesh data on the GPU *before*
`add_geometry`, and can we clear a scene without evicting the mesh already
uploaded? Both are answered below. Chasing them turned up a third thing that
matters more than either, so that is here too.

This note is input to the "swap Filament for something we control" thread in
[actors_proposal.md](actors_proposal.md) — it does not restate the device split
documented under [Devices](actors_proposal.md#devices), which is still accurate.

## Q1: staging data on the GPU before `add_geometry` — not possible

`add_geometry` **is** the upload. There is no stage/reserve/commit split to hook.

* `rendering.Renderer` exposes only `add_texture` / `update_texture` /
  `remove_texture`. Textures can be pre-uploaded; geometry cannot.
* The tensor path actively hurts. Passing a `t.geometry.TriangleMesh` already
  resident on `CUDA:0` logs
  `GPU resident triangle meshes are not currently supported for visualization.
  Copying data to CPU.` and measured **1211 ms against 258 ms** for the plain
  legacy mesh — 4.7× slower, because a device→host copy is added on top.
  Filament owns its buffers in an allocator with no interop with Open3D's CUDA
  tensors.
* The only in-place update is `Scene.update_geometry(name, point_cloud, flags)`
  — **point clouds only**, there is no mesh overload.
* `add_geometry` is already partly async: the call costs 258 ms, then the first
  render costs 119 ms against 26 ms steady-state, so ~93 ms of upload is
  deferred to the first draw.

## Q2: clearing without evicting — yes, `show_geometry`

`show_geometry(name, False)` hides without freeing. `clear_geometry()` and
`remove_geometry()` both destroy the buffers.

| operation | wall clock |
|---|---|
| `show_geometry(name, True)` + render | **34 ms** |
| `remove_geometry` + `add_geometry` + render | **369 ms** |

Hidden geometry stays registered: `has_geometry()` stays `True`,
`geometry_is_visible()` goes `False`. So a residency cache is available today —
add each mesh under its own name, hide rather than clear, keep an LRU of names.
Roughly 11× on any mesh we revisit.

**Caveat on what "resident" buys us.** On the iGPU this is host memory, so the
saving is Filament's CPU-side buffer construction, not a PCIe transfer. Adding
eight 4.4M-tri meshes moved `nvidia-smi` by 0 MiB while process RSS reached
873 MiB. Real VRAM residency only appears once rendering moves to the 4060.

## The third finding: Filament is the reason we are stuck on the iGPU

[Devices](actors_proposal.md#devices) is right that NVIDIA headless wants
`eglQueryDevicesEXT` + `EGL_PLATFORM_DEVICE_EXT` and Filament's GL backend does
not ask that way. Confirmed independently: `strings` on the pybind `.so` finds
no `eglQueryDevicesEXT` at all, and forcing the NVIDIA ICD gives
`eglInitialize failed`.

What is new is that the conclusion — "Filament's Vulkan backend or a renderer of
our own" — has a cheap third option. **ModernGL asks in exactly the way NVIDIA
wants, and it works here:**

```python
ctx = moderngl.create_context(standalone=True, backend="egl", device_index=0)
# GL_RENDERER: NVIDIA GeForce RTX 4060 Laptop GPU/PCIe/SSE2
```

`device_index=0` is the 4060, `1` is the AMD iGPU, `2` fails, `3` is llvmpipe
software. Fully headless, no X server, no PRIME variables. Pick the index
explicitly rather than defaulting — index 3 would silently render on the CPU.

## Measurements

Same mesh, same 512×512 target, all three on this machine:

| | Open3D / Filament | ModernGL (4060) | ModernGL (iGPU) |
|---|---|---|---|
| device | AMD iGPU only | RTX 4060 | AMD iGPU |
| headless | yes | yes | yes |
| frame | ~26 ms | **1.30 ms** | 4.02 ms |
| upload 4.4M tris | 258 ms | 194 ms | 104 ms |
| VRAM per copy | n/a (host) | +102 MiB, exact | n/a (host) |
| explicit evict | no — hide only | `buffer.release()` | `buffer.release()` |
| contexts per process | **1 — a 2nd core-dumps** | several, verified | several, verified |
| GIL held per view @2048 | ~21–26 ms | — | **~4–9 ms** |

**The frame numbers are not apples-to-apples.** The ModernGL figure is a
hand-written Lambert shader; Filament's `defaultLit` with IBL does more work per
pixel. Treat 20× as an upper bound on the shading side and the device move as
the part that is solid. Upload and residency numbers *are* comparable.

The GIL row carries the same caveat and is measured at 2048 rather than 512 —
see [The GIL](#the-gil-multi-context-is-not-parallel-rendering).

## Topology matters more than triangle count

`read_triangle_mesh` does **not** weld STL vertices. An STL is a triangle soup —
every triangle carries its own three vertices — and Open3D loads it that way:
`bunny.stl` comes in at exactly 3.00 verts per triangle. Subdividing welds them
(ratio drops to 0.70), so a subdivided benchmark mesh is *not* representative of
what the pipeline actually uploads.

Filament's buffer construction is O(vertices), so this dominates. Same geometry,
4,444,864 triangles either way:

| topology | verts | `add_geometry` | first render | total |
|---|---|---|---|---|
| welded (subdivided) | 3.1M | 268 ms | 118 ms | 386 ms |
| soup (**what an STL is**) | 13.3M | 1024 ms | 320 ms | **1344 ms** |

So a real 4M-triangle STL costs ~1.3 s to upload, not the ~0.26 s quoted
elsewhere in this note — **3.5× more**. Every Open3D upload figure above was
measured on welded geometry and should be read as a floor. The relative
comparisons (hide vs re-add, tensor vs legacy, Open3D vs ModernGL) are
unaffected, since they all use the same mesh.

Two consequences:

* The residency cache is **worth more** than the Q2 numbers suggest, because the
  upload it avoids is the larger of the two figures.
* Welding at load looked like it might buy most of the upload win without
  changing renderer. It does not — see below.

## Welding does not pay, and the upload was never the problem

Measured by `eval/load_path.py` on a real collection mesh (`75mm_PitFiend_
Complete.stl`, 800,236 tris, 2.4M verts at ratio 3.00 — real STLs confirm the
soup finding above). Page cache warmed, parsers interleaved, since these files
live on external storage.

| stage | current | alternative | |
|---|---|---|---|
| parse | `read_triangle_mesh` **3883 ms** | numpy binary-STL **120 ms** | ~15–30× |
| weld | — | `remove_duplicated_vertices` 928 ms | saves 230 ms of upload |
| upload | 275 ms | 45 ms welded | |

**Welding loses.** It costs 928 ms to save 230 ms — a net loss of ~700 ms, before
counting that it changes the render. `merge_close_vertices` is worse at 3548 ms.
Both collapse 2.4M verts to 400k (17%), so the *upload* saving is real and large;
Open3D's welding is simply slower than the thing it saves. A numpy weld via
`np.unique` was slower still (2803 ms), so this is not an Open3D-specific fix.

**The parse is the real cost.** `read_triangle_mesh` is ~3.9 s where a numpy
`frombuffer` over the fixed 50-byte binary-STL record is ~120 ms. Upload is
**275 ms of a 4158 ms path — 6.6%**. Everything above this section optimises the
small half.

| whole path, load + upload | total | |
|---|---|---|
| o3d soup (current) | 4158 ms | |
| **numpy soup** | **419 ms** | **9.9× faster** |
| o3d welded | 4856 ms | 0.9×, a regression |

## But the pixels move

The renderer is not bit-exact, so a control matters: the same mesh rendered
twice differs by 0.004% of pixels above 2/255. Against that floor:

| | mean | max | pixels >2/255 |
|---|---|---|---|
| control (noise floor) | 0.448 | 5.0 | **0.004%** |
| numpy soup | 0.665 | 27.0 | **4.477%** |
| o3d welded | 0.732 | 44.0 | **4.957%** |

Both are ~1000× the noise floor, so both are real changes, not jitter. The diff
map is diffuse low-magnitude shading noise spread over the model surface rather
than anything structural, and the numpy render is visually correct — but "looks
the same" is not the bar. Embeddings move, so this needs `eval/tile_and_vlm.py`
re-run against the labels before it ships.

Two notes on the numpy parser. It is binary-STL only, so it needs an ASCII
fallback (`read_triangle_mesh` handles both). And it produces 2,400,708 verts
against Open3D's 2,400,600 — Open3D merges 108, which is likely part of why the
pixels differ at all.

**This is the cheapest large win available and it does not touch the renderer.**
A ~10× cut to the pre-pixel path, for a parser and an eval re-run, against a
renderer port that needs the same eval re-run and much more work.

**It reaches the wall clock, and by more than the 6.6% suggests.** The 4158 ms
path is parse + upload, and those sit differently: `MeshPrefetcher` overlaps the
parse on a background thread, while `_upload` runs inline. So the question is how
often the prefetcher actually keeps up — which is what `mesh-wait`, the main
thread blocked on a mesh, measures.

A full-collection instrumented pass answers it (602 models, 528 needing work,
`--render-size 384 --views 8 --elevations 20,-20`, 2121 s wall):

| stage | % wall | ms/model |
|---|---|---|
| pose-embed | 29.3% | 1176 |
| pose-render | 18.7% | 750 |
| **mesh-wait** | **18.6%** | **746** |
| cache-save (SigLIP forward) | 16.8% | 673 |
| view-render | 13.7% | 551 |
| embed (preprocessing) | 1.8% | 73 |

`mesh-wait` is 394 s of 2121 s. One prefetch thread at depth 2 is nowhere near
keeping up with the real collection — the parse is *not* hidden, it is the third
largest line item. An 8-model sample of 32 mm models put this at 3.2% and was
wrong by ~6×; collection-wide is the number to trust.

At the 17–40× measured on real meshes, `mesh-wait` should mostly collapse — the
end-to-end check on 8 models took it from 177 ms to 25 ms per model. If that
ratio holds, ~340 s comes off a 2121 s run, roughly **16% of wall**, for a parser
and an eval re-run.

Two things follow beyond the direct saving:

* **It is the cheaper half of what `loader_worker_count` was for.** More loader
  threads and a faster parse target the same 18.6%; the parser removes the
  problem rather than parallelising around it, and needs no actor model to do it
  ([actors_proposal.md](actors_proposal.md)).
* **CPU freed for what is inline.** py-spy put `load_mesh` at 33% of all samples,
  competing for cores with SigLIP's preprocessing, which is on the critical path.

Note also what the same run says about the renderer thread: `pose-render` plus
`view-render` is 32.4% of wall, so the upload-and-draw path this document is
about is still the largest single consumer once the parse is fixed.

## The row that matters most

`contexts per process`. Creating a second `OffscreenRenderer` aborts the
interpreter — see the entry in `LEARNINGS.md` and commit `6683399`. That is the
hard ceiling on parallelising the render stage, and the reason async work
overlaps loading and the arbiter with rendering rather than rendering with
itself. ModernGL created a second standalone context in the same process,
rendered through it, and exited clean.

**This is available without moving devices.** ModernGL on the iGPU
(`device_index=1`) keeps the free cross-device split that
[Devices](actors_proposal.md#devices) argues for, and still gets multi-context
plus explicit residency. So "swap the renderer" and "move to the 4060" are
independent decisions, and the first is the one carrying the structural win.
Spike 2 can now measure three configurations rather than two.

**Read that row as "several contexts", not "parallel rendering."** Contexts are
only half the constraint; the GIL is the other half, and it does not go away —
see below.

## The GIL: multi-context is not parallel rendering

A second context removes the *crash*. It does not make the render stage
concurrent with Python, because ModernGL holds the GIL too — just far less of
it. Same spinner method used on `render_to_image` (a background thread counting
in pure Python; if a call releases the GIL the counter keeps climbing):

Comparing the per-view sequence each actually pays — Open3D's
`classify_stls.py:139`, ModernGL's draw + `finish` + `read`:

| per view @2048 | wall | GIL held | GIL-held time |
|---|---|---|---|
| Open3D | 40–44 ms | 51–58% | **~21–26 ms** |
| ModernGL (iGPU) | 7.4–12.9 ms | 51–66% | **~4–9 ms** |

Call it **3–5× less GIL-held time per view** — but not released. Two ModernGL
contexts on two threads still serialize for the held portion, so a threaded
Renderer is not free and needs its own measurement before the actor design leans
on it.

What this changes is the *magnitude*, and with it the decision. Over ~40 renders
that is roughly 0.25 s of GIL per model against Open3D's ~0.9 s — the difference
between a Renderer that starves every other actor for the whole run and a
serialization point that can be budgeted around. So "the Renderer must be its own
process", which three separate findings were converging on, becomes optional
rather than forced. Optional matters, because a process boundary means shipping
renders across it: 12.6 MB per view at 2048.

It also compounds with render size. At 512 the same sequence was ~1.8 ms, so
render size, GIL pressure and SigLIP preprocessing cost all move together.

**Caveats, and they are not small.** The ranges above are the honest output of
five runs, not tidy means — this measurement is noisy on this machine, and both
backends varied by ~1.7× run to run (Open3D's `render_to_image` alone came out
at 36.6 ms in one run and 61.2 ms in another). The direction is robust; the
multiplier is not. Re-measure before anything depends on it.

It is also not a controlled comparison — 81,920-tri icosphere with a Lambert
shader against a 159k-tri sphere with `defaultLit`+IBL, the same asymmetry the
frame-time caveat above flags.

Only the *combined* per-view row is trustworthy. Calls measured in isolation and
repeated back-to-back disagree with it: `fbo.read` alone came out slower at 512
than at 2048, and Open3D's `render_to_image` alone measured *longer* than the
full line that contains it. A readback with no draw behind it does not hit the
same path.

## raylib

**Gains.** `UploadMesh(&mesh, dynamic)` is an explicit upload returning VAO/VBO
ids we own; `DrawMesh(mesh, material, transform)` draws any resident mesh;
`UnloadMesh` evicts on our schedule. There is no scene graph, so Q2 dissolves —
"clearing" is not calling `DrawMesh`. It can also reach the 4060, via GLX PRIME
offload (`__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia`,
verified with `glxinfo`).

**Missing.**

* **No STL loader** — OBJ, glTF, IQM, M3D, VOX only. Minor; we need trimesh
  anyway to replace `sample_points_uniformly`.
* **Cannot run headless.** raylib requires `InitWindow` before any GL call and
  GLFW needs X11/Wayland. `FLAG_WINDOW_HIDDEN` hides the window but keeps the
  display dependency, so CI or a container needs Xvfb. This is a regression
  against what we have.
* **No lighting.** `rlights.h` is a Blinn-Phong example, not a `defaultLit`
  equivalent.
* **Still one context per process** — `rlgl` is global single-context state, so
  the parallelism ceiling survives the move. This is the deciding point.
* The Python binding is a third-party cffi wrapper.

Net: raylib fixes residency and dGPU access but costs headless operation and
keeps the ceiling. ModernGL is strictly better for our case — same GLSL work,
headless, multi-context.

## Other options

* **pyrender** — purpose-built for offscreen mesh → image, trimesh-native, EGL
  with device selection, and it ships a real light model (`DirectionalLight`,
  `SpotLight`) so it needs far less GLSL than ModernGL or raylib. Best
  effort-to-result ratio and the closest match to current output; downside is
  sparse maintenance.
* **VTK** — native STL reader, offscreen EGL with device index, built-in
  lighting. Batteries included, heavier API.
* **nvdiffrast** — the only option that removes the CPU bounce. Today the chain
  is `render_to_image()` → numpy → PIL → SigLIP preprocess → back to GPU.
  nvdiffrast rasterises into torch CUDA tensors that feed SigLIP directly, and
  we already have torch 2.2.2+cu121. Largest rewrite — all shading is ours —
  but it is the honest answer to "maximise GPU usage" end to end.

## What a switch actually costs

Not the code. The Open3D surface is small — `read_triangle_mesh`,
`get_rotation_matrix_from_xyz`, `get_rotation_matrix_from_axis_angle`,
`sample_points_uniformly`, `utility.random.seed`, `compute_vertex_normals`,
`get_axis_aligned_bounding_box`, `OffscreenRenderer`, `MaterialRecord`.
trimesh covers the geometry half.

The cost is that **any renderer swap changes the pixels.** The lighting rig is
tuned — `SUN_INTENSITY` 90000, `FILL_INTENSITY` 10000, and the crushed-black
analysis recorded in the `classify_stls.py` comments. Different shading means
different SigLIP embeddings, which invalidates `pose-cache.json` and every tuned
threshold, and needs an eval re-run to show classification did not regress.
That re-validation, not the port, is the thing to budget for.

Two things now sharpen that. `eval/parser_gate.py` is the pattern to copy — it
runs the production pose path twice changing one thing, and it reports
`MARGIN_THRESHOLD` crossings separately, because a change that perturbs margins
silently adds or removes arbiter escalations without moving any accuracy number.
The parser did exactly that on two models the ensemble gets wrong.

And the re-run can only validate *pose*. There is no category ground truth in
this repo, so "classification did not regress" is currently unmeasurable — the
output most likely to move under different shading is the one nothing can score.
See `OPEN_QUESTIONS.md`; that gap is a prerequisite for a renderer swap, not a
parallel nice-to-have.

## Recommendation

**Swap the STL parser first.** It is ~10× on the pre-pixel path, it is a
contained change, and it does not touch the renderer. Both it and any renderer
port need the same eval re-run, so do the cheap one first and spend the
validation budget once.

Then prototype ModernGL behind the existing `make_renderer` / `_upload` /
`_shoot` seam, on the **iGPU** first so the device split is held constant and
the only variable is the renderer. That isolates the multi-context win, which is
the one that unblocks the actor design. Decide the 4060 move separately, on
Spike 2 numbers.

Do **not** weld at load. It is a net regression on time and it changes the
render.

Note the ordering this implies for the actor design: if parse drops from 3.9 s
to 0.1 s, `MeshPrefetcher` has far less to hide, and the balance between the
loader and renderer stages shifts. Worth re-measuring the overlap after the
parser lands rather than designing the actor graph around today's split.

## Reproducing

The harnesses are in `eval/` and print every number quoted above.

```bash
# Q1, Q2 and the device finding
.venv/bin/python eval/renderer_open3d.py

# device selection, residency, second context
uv venv /tmp/mgl && uv pip install --python /tmp/mgl/bin/python moderngl trimesh
/tmp/mgl/bin/python eval/renderer_moderngl.py 0     # 0 = 4060, 1 = iGPU

# GIL, both backends — run each a few times, the spread is real
.venv/bin/python  eval/renderer_gil.py open3d 2048
/tmp/mgl/bin/python eval/renderer_gil.py moderngl 1 2048

# parse / weld / upload, and the pixel diff against the noise floor
.venv/bin/python eval/load_path.py <mesh.stl>
```

moderngl and trimesh are deliberately kept out of the project deps — nothing in
the pipeline imports them yet. Both scripts default to `test-stls/bunny.stl`
subdivided 3× (~4.4M tris); `test-stls/` is gitignored, so pass a mesh path if
you do not have the fixtures.

Two supporting checks that need no harness:

```bash
# Filament cannot ask for an EGL device: no such symbol in the binary
strings .venv/lib/python3.12/site-packages/open3d/cuda/pybind*.so | grep eglQueryDevices

# GLX PRIME offload does reach the 4060 — this is the raylib path
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia glxinfo -B | grep renderer
```

Numbers above are single-run on an otherwise idle machine. Re-measure before
anything depends on the exact values.

One trap worth knowing when reading `renderer_open3d.py` output: the script's
own `t.geometry -> CUDA` step opens a CUDA context, so this pid *does* appear in
`nvidia-smi` holding ~350 MiB. That is the context, not the renderer. The tell
is that it does not grow — every mesh added afterwards shows `+0` on the card
while RSS climbs.
