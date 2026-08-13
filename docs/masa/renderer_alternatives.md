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

**The frame numbers are not apples-to-apples.** The ModernGL figure is a
hand-written Lambert shader; Filament's `defaultLit` with IBL does more work per
pixel. Treat 20× as an upper bound on the shading side and the device move as
the part that is solid. Upload and residency numbers *are* comparable.

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

Both harnesses are in `eval/` and print every number quoted above.

```bash
# Q1, Q2 and the device finding
.venv/bin/python eval/renderer_open3d.py

# device selection, residency, second context
uv venv /tmp/mgl && uv pip install --python /tmp/mgl/bin/python moderngl trimesh
/tmp/mgl/bin/python eval/renderer_moderngl.py 0     # 0 = 4060, 1 = iGPU
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
