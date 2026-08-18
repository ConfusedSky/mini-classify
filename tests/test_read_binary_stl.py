import struct

import numpy as np

from src.loader import read_binary_stl

TRI = struct.pack("<12fH", 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0)   # 50 bytes


def binary(path, n_tris, header=b"exported by something", count=None):
    """A binary STL of n_tris triangles, with `count` in the header field."""
    path.write_bytes(header.ljust(80, b"\0")
                     + struct.pack("<I", n_tris if count is None else count)
                     + TRI * n_tris)
    return path


def ascii_stl(path, n_facets=3, pad=b""):
    body = b"".join(
        b"facet normal 0 0 1\n outer loop\n"
        b"  vertex 0 0 0\n  vertex 1 0 0\n  vertex 0 1 0\n"
        b" endloop\nendfacet\n" for _ in range(n_facets))
    path.write_bytes(b"solid thing\n" + body + b"endsolid thing\n" + pad)
    return path


# --- the behaviour that must not change -------------------------------------

def test_a_well_formed_binary_stl_reads(tmp_path):
    m = read_binary_stl(binary(tmp_path / "a.stl", 4))
    assert m is not None and len(m.triangles) == 4


def test_an_ascii_stl_is_left_to_open3d(tmp_path):
    assert read_binary_stl(ascii_stl(tmp_path / "a.stl")) is None


def test_junk_and_truncation_are_still_refused(tmp_path):
    (tmp_path / "junk.stl").write_bytes(b"x" * 300)          # not a whole record
    assert read_binary_stl(tmp_path / "junk.stl") is None
    short = binary(tmp_path / "short.stl", 4)
    short.write_bytes(short.read_bytes()[:-13])              # mid-record cut
    assert read_binary_stl(short) is None
    (tmp_path / "tiny.stl").write_bytes(b"\0" * 40)
    assert read_binary_stl(tmp_path / "tiny.stl") is None
    assert read_binary_stl(binary(tmp_path / "empty.stl", 0)) is None


# --- the Magics case this exists for ----------------------------------------

def test_a_header_count_that_disagrees_loses_to_the_file(tmp_path, capsys):
    # Materialise Magics: real data, wrong count. Open3D refuses these outright
    # though the meshes are sound.
    f = binary(tmp_path / "magics.stl", 70, header=b"COLOR=\x7f\x7f\x7f MATERIAL=",
               count=62)
    m = read_binary_stl(f)
    assert m is not None and len(m.triangles) == 70
    out = capsys.readouterr().out
    assert "header claims 62" in out and "file holds 70" in out


def test_the_strict_path_stays_silent(tmp_path, capsys):
    read_binary_stl(binary(tmp_path / "ok.stl", 4))
    assert capsys.readouterr().out == ""


def test_an_ascii_file_whose_length_coincides_is_still_refused(tmp_path):
    # one ASCII STL in fifty has (size - 84) divisible by 50 by luck; parsing
    # its text as float32 records is the failure this guards
    for pad in range(50):
        f = ascii_stl(tmp_path / f"a{pad}.stl", pad=b" " * pad)
        if (f.stat().st_size - 84) % 50 == 0:
            assert read_binary_stl(f) is None
            return
    raise AssertionError("no coinciding length found to test")


def test_a_derived_read_of_nan_coordinates_is_refused(tmp_path):
    f = tmp_path / "nan.stl"
    f.write_bytes(b"\0" * 80 + struct.pack("<I", 999) + b"\xff" * (50 * 6))
    assert read_binary_stl(f) is None


def test_a_derived_read_of_absurd_coordinates_is_refused(tmp_path):
    # junk decodes to huge-but-finite floats as readily as to NaN: 0x7f7f7f7f
    # is 3.4e38, which passes a finiteness test and is not a miniature
    f = tmp_path / "huge.stl"
    f.write_bytes(b"\0" * 80 + struct.pack("<I", 999) + b"\x7f" * (50 * 6))
    assert read_binary_stl(f) is None


def test_a_large_but_real_mesh_is_not_caught_by_the_bound(tmp_path):
    # the limit must be far above anything a real model reaches
    big = struct.pack("<12fH", 0, 0, 1, 0, 0, 0, 5000.0, 0, 0, 0, 5000.0, 0, 0)
    f = tmp_path / "big.stl"
    f.write_bytes(b"\0" * 80 + struct.pack("<I", 99) + big * 8)
    m = read_binary_stl(f)
    assert m is not None and len(m.triangles) == 8


def test_a_boundary_truncation_is_read_short_and_says_so(tmp_path, capsys):
    # the documented cost of trusting the file: a cut at an exact record
    # boundary is no longer refused. It is at least never silent.
    f = binary(tmp_path / "cut.stl", 10)
    f.write_bytes(f.read_bytes()[:84 + 50 * 7])
    m = read_binary_stl(f)
    assert m is not None and len(m.triangles) == 7
    assert "header claims 10" in capsys.readouterr().out
