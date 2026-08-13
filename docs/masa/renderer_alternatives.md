# Renderer Alternatives — Open3D, raylib, ModernGL

Research note, 2026-08-12. Measured on this machine (RTX 4060 Laptop + AMD
Phoenix1 iGPU, Open3D 0.19.0 CUDA build) against `test-stls/bunny.stl`
subdivided to 4,444,864 triangles / ~2.2M verts.

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

Prototype ModernGL behind the existing `make_renderer` / `_upload` / `_shoot`
seam, on the **iGPU** first so the device split is held constant and the only
variable is the renderer. That isolates the multi-context win, which is the one
that unblocks the actor design. Decide the 4060 move separately, on Spike 2
numbers.

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
