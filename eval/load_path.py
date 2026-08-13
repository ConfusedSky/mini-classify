"""Where the time before the first pixel actually goes: parse, weld, upload.

Started as "does welding STL vertices at load pay for itself?" — an STL is a
triangle soup at 3.00 verts per triangle, Filament's buffer build is O(verts),
so welding should cut upload sharply (docs/masa/renderer_alternatives.md).

It does, and it still loses: Open3D's weld costs more than the upload it saves,
and it changes the render, because a soup vertex belongs to one triangle and
shades flat while a welded vertex averages its neighbours and shades smooth.

The measurement that matters turned out to be next to it. `read_triangle_mesh`
takes ~15x longer than parsing the same binary STL with numpy, and it dwarfs the
upload it feeds. Optimising the upload was optimising the smaller half.

Usage: python eval/load_path.py [mesh.stl]
Writes renders and diff maps to eval/out/loadpath/.
"""
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from common import OUT as EVAL_OUT  # puts REPO on sys.path for classify_stls
from classify_stls import make_renderer, rotation_to_z_up, render_views, _upload
import pose

STL = sys.argv[1] if len(sys.argv) > 1 else "test-stls/bunny.stl"
SIZE = 512
OUT = Path(EVAL_OUT) / "loadpath"
OUT.mkdir(parents=True, exist_ok=True)
REC = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])


def timed(label, fn):
    s = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - s
    print(f"{dt * 1000:9.1f} ms  {label}")
    return out, dt


def numpy_read(path):
    """Binary STL is a fixed 50-byte record per triangle after an 84-byte head."""
    with open(path, "rb") as fh:
        n = int(np.frombuffer(fh.read(84)[80:84], "<u4")[0])
        rec = np.frombuffer(fh.read(n * 50), dtype=REC, count=n)
    v = rec["v"].reshape(-1, 3).astype(np.float64)
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v),
        o3d.utility.Vector3iVector(np.arange(n * 3, dtype=np.int32).reshape(-1, 3)))
    m.compute_vertex_normals()
    return m


print(f"mesh: {STL}")
open(STL, "rb").read()  # warm the page cache so neither parser is charged for I/O

# 1. The parsers. Interleaved, because these files usually live on external
#    storage and a cold first read would be charged to whichever ran first.
print("\n--- parse (page cache warm, interleaved) ---")
for i in range(2):
    npm, t_np = timed(f"numpy binary-STL parse{'' if i == 0 else ' (again)'}",
                      lambda: numpy_read(STL))
    o3m, t_o3 = timed(f"o3d.read_triangle_mesh{'' if i == 0 else ' (again)'}",
                      lambda: o3d.io.read_triangle_mesh(STL))
o3m.compute_vertex_normals()
print(f"           o3d  : verts={len(o3m.vertices):,} tris={len(o3m.triangles):,} "
      f"ratio={len(o3m.vertices) / len(o3m.triangles):.2f}")
print(f"           numpy: verts={len(npm.vertices):,} tris={len(npm.triangles):,}")
print(f"           numpy is {t_o3 / t_np:.1f}x faster")

# 2. Welding, and what it buys on the upload it feeds.
print("\n--- weld strategies ---")
welded, t_weld = timed("remove_duplicated_vertices (exact)",
                       lambda: o3d.geometry.TriangleMesh(o3m).remove_duplicated_vertices())
welded.compute_vertex_normals()
print(f"           -> verts={len(welded.vertices):,} "
      f"({len(welded.vertices) / len(o3m.vertices) * 100:.0f}% of soup)")
_, t_close = timed("merge_close_vertices(1e-6) (spatial)",
                   lambda: o3d.geometry.TriangleMesh(o3m).merge_close_vertices(1e-6))

# 3. Upload + render for each, through the real code path.
print("\n--- upload + render, via classify_stls ---")
rot = rotation_to_z_up(np.array(pose.detect_up_axis(o3m)[0]))
r = make_renderer(SIZE)
cases = [("o3d soup (current)", o3m, t_o3), ("numpy soup", npm, t_np),
         ("o3d welded", welded, t_o3 + t_weld)]
images, totals = {}, {}
for name, m, load_s in cases:
    mm = o3d.geometry.TriangleMesh(m)
    mm.rotate(rot, center=(0, 0, 0))
    s = time.perf_counter()
    _upload(r, mm)
    up_s = time.perf_counter() - s
    img = render_views(r, mm, [(0.0, np.deg2rad(20.0))])[0]
    img.save(OUT / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.png")
    images[name] = np.asarray(img.convert("RGB"), dtype=np.float32)
    totals[name] = load_s + up_s
    print(f"  {name:20s} load={load_s * 1000:8.1f} + upload={up_s * 1000:7.1f} "
          f"= {totals[name] * 1000:8.1f} ms")
base = totals["o3d soup (current)"]
for n, v in totals.items():
    if n != "o3d soup (current)":
        print(f"  {n:20s} {base / v:.1f}x faster than current ({(base - v) * 1000:+.0f} ms)")

# 4. Do the pixels move? Needs a noise floor: this renderer is not bit-exact.
print("\n--- pixel difference (control first: same mesh rendered twice) ---")
mm = o3d.geometry.TriangleMesh(o3m)
mm.rotate(rot, center=(0, 0, 0))
ctrl = np.asarray(render_views(r, mm, [(0.0, np.deg2rad(20.0))])[0].convert("RGB"),
                  dtype=np.float32)
ref = images["o3d soup (current)"]


def report(label, img):
    d = np.abs(img - ref)
    print(f"  {label:22s} mean={d.mean():6.3f}/255  max={d.max():5.1f}  "
          f"pixels >2/255: {(d.max(axis=2) > 2).mean() * 100:6.3f}%")
    return d


report("CONTROL (noise floor)", ctrl)
for name, img in images.items():
    if name != "o3d soup (current)":
        d = report(name, img)
        Image.fromarray(np.clip(d.max(axis=2) * 4, 0, 255).astype(np.uint8)).save(
            OUT / f"diff_{name.replace(' ', '_')}.png")

print(f"\nrenders + diff maps in {OUT}/")
print("Anything well above the control row is a real change to the pixels, so it "
      "moves SigLIP embeddings and needs an eval re-run before it ships.")
