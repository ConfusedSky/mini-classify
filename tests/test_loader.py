"""src/loader.py: LoadedMesh + get() — the child's Loader half. get() raises
on malformed input (the child loop converts to Failure); nbytes feeds the
renderer's residency accounting."""
import struct
from pathlib import Path

import numpy as np
import pytest

from src import loader

REPO = Path(__file__).resolve().parent.parent

TRI = struct.pack("<12fH", 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0)   # 50 bytes


def binary(path, n_tris, header=b"exported by something", count=None):
    """A binary STL of n_tris triangles, with `count` in the header field."""
    path.write_bytes(header.ljust(80, b"\0")
                     + struct.pack("<I", n_tris if count is None else count)
                     + TRI * n_tris)
    return path


def test_get_returns_a_loaded_mesh(tmp_path):
    lm = loader.get(binary(tmp_path / "a.stl", 4))
    assert isinstance(lm, loader.LoadedMesh)
    assert lm.file == tmp_path / "a.stl"
    assert len(lm.mesh.triangles) == 4
    assert lm.mesh.has_vertex_normals()      # computed, ready to render


def test_nbytes_matches_open3ds_representation(tmp_path):
    lm = loader.get(binary(tmp_path / "a.stl", 4))
    # triangle soup: 12 verts * 3 float64 + 12 vertex normals * 3 float64
    # + 4 triangle normals * 3 float64 + 4 triangles * 3 int32
    assert lm.nbytes == 12 * 24 + 12 * 24 + 4 * 24 + 4 * 12
    assert lm.nbytes == (np.asarray(lm.mesh.vertices).nbytes
                         + np.asarray(lm.mesh.vertex_normals).nbytes
                         + np.asarray(lm.mesh.triangle_normals).nbytes
                         + np.asarray(lm.mesh.triangles).nbytes)


def test_get_raises_on_junk(tmp_path):
    """Malformed input raises out of get(); the child loop is the boundary
    that converts to Failure — get itself never swallows."""
    junk = tmp_path / "junk.stl"
    junk.write_bytes(b"x" * 300)             # not a whole record count
    with pytest.raises(Exception):
        loader.get(junk)


def test_get_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        loader.get(tmp_path / "gone.stl")


def test_get_raises_on_empty_mesh(tmp_path):
    """A parseable container with no triangles is malformed for this
    pipeline: ValueError('no triangles'), today's load_mesh contract."""
    empty = binary(tmp_path / "e.stl", 0)
    with pytest.raises(ValueError, match="no triangles"):
        loader.get(empty)


def test_lying_header_still_trusts_the_file(tmp_path):
    """The Materialise Magics case (commit bd6be81): header count wrong, data
    fills the file exactly — the file wins. Pinned here because the loader is
    now the child's copy of that behaviour."""
    lm = loader.get(binary(tmp_path / "m.stl", 6, count=1_500_000))
    assert len(lm.mesh.triangles) == 6


def test_real_collection_stls_load(tmp_path):
    for name in ("bunny.stl", "torus.stl", "blocky_building.stl"):
        lm = loader.get(REPO / "test-stls" / name)
        assert len(lm.mesh.triangles) > 0
        assert lm.nbytes > 0
