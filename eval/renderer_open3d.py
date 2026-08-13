"""What Open3D's Filament renderer will and will not let us do with GPU memory.

Answers three things, in order:
  1. can mesh data be staged on the GPU before add_geometry?  (no)
  2. can a scene be cleared without evicting the upload?       (yes, show_geometry)
  3. which GPU is any of this actually running on?             (the iGPU)

Usage: python eval/renderer_open3d.py [mesh.stl] [subdivisions]
Defaults to ../test-stls/bunny.stl subdivided 3x (~4.4M tris). test-stls/ is
gitignored, so pass a path if you do not have the fixtures.

Numbers from this script are written up in docs/masa/renderer_alternatives.md.
"""
import os
import subprocess
import sys
import time

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering

MESH = sys.argv[1] if len(sys.argv) > 1 else "test-stls/bunny.stl"
SUBDIV = int(sys.argv[2]) if len(sys.argv) > 2 else 3
SIZE = 512


def timed(label, fn):
    t = time.perf_counter()
    out = fn()
    print(f"{(time.perf_counter() - t) * 1000:9.1f} ms  {label}")
    return out


def vram():
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return int(q.stdout.strip().splitlines()[0]) if q.returncode == 0 else -1


def material():
    m = rendering.MaterialRecord()
    m.shader = "defaultLit"
    m.base_color = [0.7, 0.7, 0.7, 1.0]
    return m


print("--- load, as the pipeline does it ---")
mesh = timed("read_triangle_mesh", lambda: o3d.io.read_triangle_mesh(MESH))
if SUBDIV:
    mesh = timed(f"subdivide_midpoint({SUBDIV})",
                 lambda: mesh.subdivide_midpoint(SUBDIV))
timed("compute_vertex_normals", mesh.compute_vertex_normals)
print(f"           tris={len(mesh.triangles):,} verts={len(mesh.vertices):,}")

tmesh = timed("legacy -> t.geometry (CPU)",
              lambda: o3d.t.geometry.TriangleMesh.from_legacy(mesh))
tcuda = timed("t.geometry -> CUDA",
              lambda: tmesh.to(o3d.core.Device("CUDA:0")))

r = rendering.OffscreenRenderer(SIZE, SIZE)
r.scene.set_background([1.0, 1.0, 1.0, 1.0])
r.scene.scene.enable_sun_light(True)
b = mesh.get_axis_aligned_bounding_box()
center, radius = b.get_center(), np.linalg.norm(b.get_extent()) * 1.4


def shoot():
    r.setup_camera(45.0, center, center + np.array([radius, 0, 0]), [0, 0, 1])
    return r.render_to_image()


# 1. Staging before add_geometry. There is no stage/commit split to hook: the
#    call *is* the upload, and a CUDA-resident tensor mesh is copied back to the
#    host first (Filament logs the warning itself), so it is strictly slower.
print("\n--- Q1: add_geometry variants ---")
timed("add legacy (downsampled copy ON, default)",
      lambda: r.scene.add_geometry("a", mesh, material()))
timed("  first render after add", shoot)
timed("  second render (steady state)", shoot)
timed("add legacy (downsampled copy OFF)",
      lambda: r.scene.add_geometry("b", mesh, material(), False))
timed("add t.geometry CPU", lambda: r.scene.add_geometry("c", tmesh, material(), False))
timed("add t.geometry CUDA  <- watch for the copy-to-CPU warning",
      lambda: r.scene.add_geometry("d", tcuda, material(), False))

# 1b. Topology, which dominates everything above. read_triangle_mesh does not
#     weld STL vertices -- an STL is a triangle soup at exactly 3.00 verts per
#     triangle -- but subdividing welds them, so the mesh built above is NOT
#     representative of what the pipeline uploads. Filament's buffer build is
#     O(verts), so the same geometry as soup costs several times more.
print("\n--- Q1b: welded vs soup, same geometry ---")
shoot()  # drain Q1's deferred uploads, else they land in the first timing below
for n in "abcd":  # and stop the Q1 meshes being redrawn in it
    r.scene.show_geometry(n, False)
shoot()
v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
soup = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(v[f].reshape(-1, 3)),
    o3d.utility.Vector3iVector(np.arange(len(f) * 3).reshape(-1, 3)))
soup.compute_vertex_normals()
raw = o3d.io.read_triangle_mesh(MESH)
print(f"  as loaded from disk: {len(raw.vertices) / len(raw.triangles):.2f} "
      f"verts/tri (3.00 = soup, this is what we really upload)")
for label, g in (("welded", mesh), ("soup  ", soup)):
    t = time.perf_counter()
    r.scene.add_geometry(f"topo_{label.strip()}", g, material(), False)
    add = time.perf_counter() - t
    t = time.perf_counter()
    shoot()
    first = time.perf_counter() - t
    r.scene.show_geometry(f"topo_{label.strip()}", False)
    print(f"  {label} verts={len(g.vertices):>10,}  add={add * 1000:8.1f} ms  "
          f"first-render={first * 1000:7.1f} ms  total={(add + first) * 1000:8.1f} ms")

# 2. Clearing without evicting. show_geometry keeps the buffers; clear_geometry
#    and remove_geometry destroy them.
print("\n--- Q2: hide vs remove/re-add ---")
for n in "abcd":
    r.scene.show_geometry(n, False)
shoot()
t = time.perf_counter()
r.scene.show_geometry("a", True)
shoot()
print(f"{(time.perf_counter() - t) * 1000:9.1f} ms  show_geometry(True) + render")
r.scene.show_geometry("a", False)

t = time.perf_counter()
r.scene.remove_geometry("b")
r.scene.add_geometry("b", mesh, material(), False)
shoot()
print(f"{(time.perf_counter() - t) * 1000:9.1f} ms  remove + add + render")
print("hidden geometry still registered:",
      {n: (r.scene.has_geometry(n), r.scene.geometry_is_visible(n)) for n in "acd"})

# 3. Where does it run, and does 'resident' cost VRAM? On the iGPU it does not:
#    the memory is host RAM, so nvidia-smi stays flat while RSS climbs.
print("\n--- Q3: device and residency ---")
base = vram()
for i in range(6):
    r.scene.add_geometry(f"m{i}", mesh, material(), False)
    r.scene.show_geometry(f"m{i}", False)
    shoot()
    print(f"  {i + 1} extra resident: nvidia-smi {vram()} MiB (+{vram() - base})")

rss = int(open(f"/proc/{os.getpid()}/status").read()
          .split("VmRSS:")[1].split()[0]) // 1024
print(f"  process RSS: {rss} MiB  <- the meshes live here, in host memory")
# Our pid does show up on the 4060, but that is the CUDA context created by the
# t.geometry->CUDA test above, not the renderer. The tell is that it is a fixed
# context-sized allocation and does not grow as meshes are added: every '+0'
# above is a 107 MB mesh that landed in RAM rather than VRAM.
print(f"  this pid: {os.getpid()} (CUDA ctx from Q1, not the renderer)")
print(subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                      "--format=csv"], capture_output=True, text=True).stdout.strip())

t = time.perf_counter()
for _ in range(10):
    shoot()
print(f"\n{(time.perf_counter() - t) / 10 * 1000:.1f} ms per {SIZE}x{SIZE} frame "
      f"({len(mesh.triangles) * 7:,} tris resident)")
