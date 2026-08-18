"""Mesh loading for the render child: `LoadedMesh` + `get()` — the Loader
half of the child's Loader/Renderer seam (docs/actor-refactor/
data_structures.md §Inside the child, interfaces.md §Render child).

`get()` raises on malformed input; the child loop (`src/render_child.py`) is
the boundary that converts exceptions to `Failure`. Extracted from
`classify_stls.py` (`read_binary_stl`, `load_mesh`) so the child never
imports the CLI module — child side imports open3d/PIL/numpy only, never
torch (interfaces.md import-rule row 1).

`read_binary_stl`: the file outvotes a lying header (commit bd6be81, for
Materialise Magics files). The remaining guards — ASCII detection, the
finite/magnitude bound, the whole-record remainder — are deliberate; do not
loosen them (CLAUDE.md hard constraints).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


@dataclass
class LoadedMesh:
    file: Path
    mesh: o3d.geometry.TriangleMesh
    nbytes: int                    # feeds ResidentMesh accounting


# Binary STL: an 80-byte header, a uint32 triangle count, then this fixed
# 50-byte record per triangle. The fixed stride is the whole trick.
STL_RECORD = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])


def read_binary_stl(path):
    """A binary STL read straight into arrays, or None if it is not one.

    read_triangle_mesh dominates everything before the first pixel: ~3.9 s on an
    800k-triangle collection mesh against ~120 ms here, where the upload it
    feeds is 275 ms. Optimising the renderer was optimising the small half
    (eval/load_path.py, docs/actor-refactor/renderer_alternatives.md).

    The header cannot be trusted to say which format this is — plenty of binary
    STLs start with "solid" — so the test is arithmetic: it is binary only if
    the file is exactly the length a triangle count implies. Anything else
    (ASCII, truncated, junk) returns None and takes the Open3D path.

    Which count, though, is not always the one in the header. Materialise
    Magics writes a `COLOR=... MATERIAL=...` header and a triangle count that
    can be wrong by anything from 8 triangles to 1.5 million, while the data
    itself fills the file exactly. Open3D refuses those outright ("Failed to
    determine STL storage representation") though the meshes are sound, so when
    the header disagrees and the remaining bytes are a whole number of records,
    the file wins and says so out loud.

    That is a real loosening: a file truncated at an exact 50-byte boundary is
    now read short rather than refused. Two things keep it narrow — an ASCII
    STL is detected and rejected before the arithmetic can coincide, and a
    derived read must parse to finite coordinates.

    The result is a triangle soup, three unshared vertices per triangle, which
    is what an STL *is*. Open3D's reader welds a handful (108 of 2.4M on a real
    mesh); we do not, and that difference shows up in the render."""
    size = path.stat().st_size
    if size < 84:
        return None
    with open(path, "rb") as fh:
        head = fh.read(84)
        n = int(np.frombuffer(head[80:84], "<u4")[0])
        derived = False
        if size != 84 + 50 * n:
            if (size - 84) % 50:
                return None
            # an ASCII STL's bytes 80:84 are text, so its implied count is
            # nonsense — and one file in fifty would pass the arithmetic by
            # coincidence. Read the real marker instead of gambling on it.
            if b"facet" in head + fh.read(448):
                return None
            fh.seek(84)
            n, derived = (size - 84) // 50, True
        if n == 0:
            return None
        # fromfile continues from the header rather than materialising the
        # whole record block as bytes first — 200 MB on a 4M-triangle mesh
        rec = np.fromfile(fh, dtype=STL_RECORD, count=n)
    if len(rec) != n:                       # short read despite the size check
        return None
    if derived:
        # Coordinates are millimetres. Finite alone is too weak a test — junk
        # decodes to huge-but-finite floats as readily as to NaN (0x7f7f7f7f is
        # 3.4e38) — so bound the magnitude too: a thousand kilometres is not a
        # miniature, and no real mesh comes close to the limit.
        v = rec["v"]
        if not (np.isfinite(v).all() and np.abs(v).max() < 1e9):
            return None                     # not triangles, whatever it is
        print(f"  {path.name}: header claims "
              f"{int(np.frombuffer(head[80:84], '<u4')[0]):,} triangles, file holds "
              f"{n:,} — trusting the file")
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(rec["v"].reshape(-1, 3).astype(np.float64)),
        o3d.utility.Vector3iVector(np.arange(3 * n, dtype=np.int32).reshape(-1, 3)))


def load_mesh(mesh_path):
    """The raw mesh, normals computed. Raises on malformed input (`ValueError`
    for the no-triangles case; whatever the parser raises otherwise)."""
    mesh_path = Path(mesh_path)
    mesh = None
    if mesh_path.suffix.lower() == ".stl":
        mesh = read_binary_stl(mesh_path)
    if mesh is None:
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles():
        raise ValueError("no triangles")
    mesh.compute_vertex_normals()
    return mesh


def mesh_nbytes(mesh):
    """Host-side bytes of Open3D's representation: float64 points, vertex AND
    triangle normals (compute_vertex_normals populates both — A-R1-1: leaving
    triangle_normals out undercounted residency ~13%), int32 triangles."""
    return int(np.asarray(mesh.vertices).nbytes
               + np.asarray(mesh.vertex_normals).nbytes
               + np.asarray(mesh.triangle_normals).nbytes
               + np.asarray(mesh.triangles).nbytes)


def get(file) -> LoadedMesh:
    """Load one mesh for the render child. Raises on malformed input — the
    child loop converts to `Failure(file, index, str(e))`."""
    file = Path(file)
    mesh = load_mesh(file)
    return LoadedMesh(file=file, mesh=mesh, nbytes=mesh_nbytes(mesh))
