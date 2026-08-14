## Where the run actually goes, instrumented (2026-08-13)

`--instrument PATH` records exclusive per-stage timing plus CPU / NVIDIA /
amdgpu utilization, sampled against whichever stage the main thread is in
(`instrument.py`). Stage timing is exact; utilization is statistical.

### Rendering does not happen on the RTX 4060

Open3D brings up EGL headless on `/dev/dri/renderD129`, which is **`amdgpu` —
the Phoenix1 integrated GPU**. The 4060 is `renderD128`/`nvidia`. Rendering is
hardware accelerated, just on the wrong card, and the ~30% seen on the 4060
during a run is SigLIP by itself.

This is not a flag flip. Forcing the NVIDIA EGL vendor library fails at
`eglInitialize` and then core-dumps; NVIDIA headless wants `eglQueryDevicesEXT`
+ `EGL_PLATFORM_DEVICE_EXT` and Filament's GL backend does not ask that way —
confirmed by `strings` finding no such symbol in the pybind `.so`.

Consequences: render and embed **never contended**, they are on different
devices. The 8 GB card holds SigLIP and nothing else. The iGPU draws from shared
system RAM, and its buffers land in **GTT, not the vram carve-out** — sizing
anything off `mem_info_vram_used` misses the geometry entirely. And it is not
obvious we want to move: iGPU-render plus dGPU-embed is free cross-device
parallelism. See `docs/masa/renderer_alternatives.md`.

### Full-collection baseline, and what the parser swap moved

602 models, 528 needing work, `--render-size 384 --views 8 --elevations 20,-20`,
2121 s wall, old parser:

| stage | % wall | ms/model |
|---|---|---|
| pose-embed | 29.3% | 1176 |
| pose-render | 18.7% | 750 |
| **mesh-wait** | **18.6%** | **746** |
| cache-save (the SigLIP forward — see below) | 16.8% | 673 |
| view-render | 13.7% | 551 |
| embed (preprocessing) | 1.8% | 73 |

**`mesh-wait` is the main thread blocked on `MeshPrefetcher`.** One loader thread
at depth 2 does not keep up with the real collection, so the parse was never
hidden — it was the third largest line item. An 8-model sample of 32 mm models
put it at 3.2% and was wrong by ~6×; small samples understate this badly because
mesh size varies more across the collection than anything else does.

After the numpy STL parser landed, on a 104-model subset: `mesh-wait` **746 ms →
10 ms**, and the profile inverts —

| | full collection, old parser | subset, new parser |
|---|---|---|
| CPU | 15% | 10% |
| NVIDIA | 44% | **64%** |
| AMD iGPU | 26% | 19% |

SigLIP on the 4060 is now **68.5%** of the run (`pose-embed` 41.7% + `cache-save`
24.4% + `embed` 2.4%) against 27.9% for rendering. The run is neither load-bound
nor render-bound any more; it is embedding-bound, and `pose-embed` alone — 24
up-candidate tiles — costs **1.6× what embedding the classification views
costs**. Read the subset's percentages as shape, not absolutes: its meshes are
about half the collection's average size (`pose-render` and `view-render` both
roughly halved, which a parser cannot cause).

### `embed` and `cache-save` are one number split by a CUDA sync

Torch launches async, so `embed_images` returns before the GPU has done
anything and the `.cpu()` inside the cache write absorbs the wait. The
utilization table shows it plainly: NVIDIA at 1–12% during `embed` and 28–65%
during `cache-save`. **Always read the two together.** `cache-save` scales with
view count (523 ms at 16 views, 123 ms at 4 — ~31 ms per image at 384 input) and
is independent of render size, which is what identifies it as the forward pass.

The corollary caught a wrong conclusion in this session: py-spy put the SigLIP
forward at ~2.5% of samples and preprocessing at 19%, which reads as "the GPU is
free". py-spy measures *Python stack* time, and an async launch returns
instantly. Preprocessing really is ~19% and really does vanish at a 384 source,
but the forward is ~520 ms/model, not free.

### `render_to_image` holds the GIL; the mesh reader does not

`py-spy record --gil` on a real run: 99% of GIL-held time is the main thread, and
`_shoot`'s line 139 alone is 62% of it. Splitting that line with a spinner thread
(`eval/renderer_gil.py`):

| @2048 | wall | GIL held |
|---|---|---|
| `render_to_image()` | 36.6–61.2 ms | **~85–92%** |
| `np.asarray()` | ~0 ms | free — it is a view, not a copy |
| `Image.fromarray()` | 14–17 ms | ~11% — releases |
| full line 139 | 40–44 ms | 51–58% |

So Filament's render call blocks every other Python thread for most of its
duration, while Open3D's mesh reader releases (the prefetch thread takes 1% of
GIL time while being 33% of all samples — which is *why* prefetching works).
Threads doing native GIL-free work overlap fine; threads doing Python-level work
do not.

ModernGL holds it too, roughly 3–5× less (~4–9 ms per view against ~21–26 ms).
**So "several contexts per process" is not "parallel rendering"** — a threaded
renderer still serializes for the held portion. Both figures are noisy: five
runs, both backends moving ~1.7× run to run, and calls measured in isolation
disagree with the combined sequence badly enough that only the per-view row is
quotable.

### Binary STL parsing: `read_triangle_mesh` was the cost

An STL is a triangle soup — 3.00 verts per triangle — and the binary format is a
fixed 50-byte record after an 84-byte header, so `np.fromfile` over a record
dtype is the entire parser. Measured **17–40× faster** on real collection meshes
(1163→66 ms, 933→23 ms), with triangle counts and bounding boxes exact against
the old reader.

Detect the format by arithmetic, not the header: plenty of binary STLs begin
with `solid`, so the only reliable test is whether the file length is exactly
`84 + 50n`. ASCII, truncated and junk files fall back to `read_triangle_mesh`.

It moves the pixels — Open3D welds a handful of soup vertices and this does not
— so it was gated against the labels (`eval/parser_gate.py`, which runs the
production pose path twice changing only the loader):

| | old parser | new parser | picks changed |
|---|---|---|---|
| 384px | 45/49 | 45/49 | 0/49 |
| 2048px | 43/49 | 44/49 | 1/49 |

Geometry picks identical at both sizes; margins shift by a mean of 0.024. The
one flip is `Container_complete` at a margin of **0.001**, which is the ensemble
undecided rather than improved — do not read the +1 as a gain.

**The cost that no accuracy column shows is the escalation gate.** A 0.024 margin
shift walks models across `MARGIN_THRESHOLD` (0.45). `Concrete Chunk (2)` stops
escalating at *both* render sizes and `Bedienkonsole` at 384, and the ensemble
has both **wrong** — two lost chances for the arbiter to rescue them. Any change
that perturbs margins should report threshold crossings separately, not just
accuracy.

### Rendering pose tiles below `SHEET_THUMB` costs the whole 512 px gain

`make_contact_sheet` scales tiles with `Image.thumbnail`, which shrinks but never
enlarges. At `--render-size 384` the tiles sit in the top-left of their 512 px
cells with white gutters, and the subject covers **56% of the sheet's pixels**.
`classify_stls.py` warns at startup; nothing had measured it.

gemini-3.5-flash, 49 labelled models, one Vertex call per model per
configuration (`eval/gemini_sheet_fill.py`):

| | orig+holdout (n=44) | all (n=49) |
|---|---|---|
| padded-384 (what `--render-size 384` does today) | **41/44** | 44/49 |
| filled-384 (same tiles upscaled into the cell) | 42/44 | 45/49 |
| native-512 | **43/44** | 46/49 |

Both endpoints land *exactly* on the numbers already recorded above for this
model — 41/44 at `thumb=256`, 43/44 at `thumb=512`. So rendering pose tiles at
384 does not cost a little resolution: it costs the entire documented 256→512
improvement, the one that justified adopting the 512 px sheet. Filling the cells
recovers half of it without re-rendering anything.

Two of 44 is p=0.5 by sign test on its own; what makes it credible is that both
ends reproduce independent measurements rather than landing near them. The
answers are unstable regardless — 6/49 change between padded and filled, 3/49
between padded and native, and on the frozen holdout padded actually scored
21/21 against filled's 20/21.

**This was never covered by the existing sweeps.** `tile_and_vlm.py` sweeps tile
resolution 384..2048 but hands the VLM only its 2048 tiles (`tiles_big`), and
`backbone_sweep.py` crosses render size with SigLIP towers — both measure the
*ensemble*. `common.build_sheets`' docstring says "only the sheet size matters to
a VLM; render_px ... made no difference", which applies the ensemble result to a
question it was not asked. That docstring is wrong and should be corrected.

