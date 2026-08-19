"""src/collection.py: the scope resolver, the pose block, and the hit shape.

Built on a real cache in `tmp_path` rather than mocks — the module's whole job
is agreeing with what a classify run wrote, so the fixture writes the same
files `done.py` writes (`.npy` under `embeds/`, a `pose-cache.json`) under keys
computed by the production functions. A mock would agree with itself.

What is pinned here is the *contract the API rests on*: which rows a path
selects, that the three rejections are three different errors, that a real
directory with nothing embedded is an answer rather than an error, and that a
loaded Collection touches the filesystem for nothing but a 404's stat.

numpy only; no torch, no GPU, no HTTP.
"""
import argparse
import json

import numpy as np
import pytest

from src import pose
from src.cachedir import cache_key, embeds_dir
from src.collection import (COVERS, Collection, NoSuchPath, OutsideCollection,
                            Scope, VirtualPath)

DIM = 8


def make_args(tmp_path, **over):
    d = dict(input=str(tmp_path / "stl"), cache_dir=str(tmp_path / "cache"),
             views=2, elevations=[20.0], render_size=384,
             model="google/siglip-so400m-patch14-384", compile=False,
             up_axis="auto", rescan=True)
    d.update(over)
    return argparse.Namespace(**d)


def build(tmp_path, layout, *, embed=None, ups=None, front=None, **over):
    """A collection on disk plus the cache a classify run would have left.

    `layout` is a list of root-relative .stl paths. `embed` names which of them
    get a cached embedding (default: all). `ups` overrides the pose up vector
    per path; `front` sets a front_view index for the run's view config."""
    args = make_args(tmp_path, **over)
    root = tmp_path / "stl"
    embed = layout if embed is None else embed
    ups, front = ups or {}, front or {}

    files = {}
    for rel in layout:
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"solid x\nendsolid x\n")     # never parsed; only stat'd
        files[rel] = f

    entries, n_views = {}, args.views * len(args.elevations)
    for rel, f in files.items():
        up = ups.get(rel, [0.0, 0.0, 1.0])
        e = {"up": list(up), "confidence": 0.9, "source": "geometry",
             "margin": 0.5, "v": pose.POSE_CACHE_VERSION}
        if rel in front:
            e["front_view"] = {f"{args.views}v-e20": front[rel]}
        entries[pose.file_identity(f, root)] = e
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    # flat {identity: entry}, exactly what `pose.save_pose_cache` writes
    (cache / "pose-cache.json").write_text(json.dumps(entries))

    ed = embeds_dir(args.cache_dir)
    ed.mkdir(parents=True, exist_ok=True)
    poses = pose.load_pose_cache(args.cache_dir)
    for rel in embed:
        f = files[rel]
        token = pose.embed_cache_token(poses.get(pose.file_identity(f, root)),
                                       args.up_axis)
        vec = np.full((n_views, DIM), 0.1, dtype=np.float32)
        np.save(ed / f"{cache_key(f, args, token, root)}.npy", vec)
    return args, root, files


# --- scope: which rows a path selects ---------------------------------------

def test_the_whole_collection_is_every_row(tmp_path):
    args, *_ = build(tmp_path, ["a/one.stl", "b/two.stl", "b/c/three.stl"])
    c = Collection.load(args)
    s = c.resolve(None)
    assert s.path is None and len(s.rows) == 3
    assert s.status == "indexed" and s.covers == COVERS


def test_a_directory_selects_only_what_is_under_it(tmp_path):
    args, root, _ = build(tmp_path, ["a/one.stl", "b/two.stl", "b/c/three.stl"])
    c = Collection.load(args)
    got = {c.files[i].name for i in c.resolve("b").rows}
    assert got == {"two.stl", "three.stl"}          # recursive, and only b


def test_a_scope_accepts_absolute_and_root_relative_alike(tmp_path):
    args, root, _ = build(tmp_path, ["a/one.stl", "b/two.stl"])
    c = Collection.load(args)
    assert c.resolve("a").rows.tolist() == c.resolve(str(root / "a")).rows.tolist()


def test_a_scope_may_name_one_file(tmp_path):
    args, root, files = build(tmp_path, ["a/one.stl", "a/two.stl"])
    c = Collection.load(args)
    s = c.resolve("a/one.stl")
    assert len(s.rows) == 1 and c.files[s.rows[0]].name == "one.stl"


def test_a_sibling_prefix_is_not_a_parent(tmp_path):
    """`kits` must not match `kits-old` — the bug a string prefix would have."""
    args, *_ = build(tmp_path, ["kits/one.stl", "kits-old/two.stl"])
    c = Collection.load(args)
    assert len(c.resolve("kits").rows) == 1


# --- scope: the three rejections are three answers --------------------------

def test_a_zip_virtual_path_is_rejected_as_unaddressable(tmp_path):
    args, *_ = build(tmp_path, ["a/one.stl"])
    c = Collection.load(args)
    with pytest.raises(VirtualPath):
        c.resolve("a/pack.zip!/inner.stl")


def test_a_path_outside_the_collection_is_its_own_error(tmp_path):
    args, *_ = build(tmp_path, ["a/one.stl"])
    (tmp_path / "elsewhere").mkdir()
    c = Collection.load(args)
    with pytest.raises(OutsideCollection):
        c.resolve(str(tmp_path / "elsewhere"))


def test_a_path_that_does_not_exist_is_not_found(tmp_path):
    args, *_ = build(tmp_path, ["a/one.stl"])
    c = Collection.load(args)
    with pytest.raises(NoSuchPath):
        c.resolve("a/nowhere")


def test_a_real_directory_with_nothing_embedded_is_an_answer_not_an_error(tmp_path):
    """The distinction the whole scope block exists for: 'nothing matched' and
    'nothing here is classified' must not look the same to the UI."""
    args, root, _ = build(tmp_path, ["a/one.stl", "b/two.stl"], embed=["a/one.stl"])
    c = Collection.load(args)
    s = c.resolve("b")
    assert s.rows.size == 0
    assert s.status == "unindexed" and s.n_scanned == 1 and s.n_indexed == 0


def test_partial_is_scanned_but_not_embedded(tmp_path):
    args, *_ = build(tmp_path, ["a/one.stl", "a/two.stl"], embed=["a/one.stl"])
    c = Collection.load(args)
    s = c.resolve("a")
    assert (s.n_indexed, s.n_scanned, s.status) == (1, 2, "partial")


def test_covers_is_published_so_a_3mf_folder_is_not_read_as_covered(tmp_path):
    """classify walks .stl only; model-browser lists .3mf and .obj too."""
    args, root, _ = build(tmp_path, ["a/one.stl"])
    (root / "b").mkdir()
    (root / "b" / "thing.3mf").write_bytes(b"x")
    c = Collection.load(args)
    s = c.resolve("b")
    assert s.status == "unindexed" and s.n_scanned == 0
    assert s.as_dict()["covers"] == ["stl"]         # the disambiguator


# --- pose ------------------------------------------------------------------

def test_pose_carries_up_azimuth_zero_and_the_front_camera(tmp_path):
    args, *_ = build(tmp_path, ["a/one.stl"], front={"a/one.stl": 1})
    c = Collection.load(args)
    p = c.pose_of(0)
    assert p["up"] == [0.0, 0.0, 1.0]
    assert p["source"] == "geometry" and p["confidence"] == pytest.approx(0.9)
    # view 1 of a 2-view ring at elevation 20
    assert p["front"] == {"view": 1, "azimuth_deg": 180.0, "elevation_deg": 20.0}


def test_azimuth_zero_is_the_direction_azimuth_zero_is_measured_from(tmp_path):
    """It must equal `rotation_to_z_up(up).T @ [1,0,0]` for every up, which is
    what lets a viewer derive its own offset instead of inheriting ours
    (surface.md §pose). +Z is the identity case and proves nothing alone."""
    layout = ["y/one.stl", "x/two.stl", "nz/three.stl"]
    ups = {"y/one.stl": [0.0, 1.0, 0.0], "x/two.stl": [1.0, 0.0, 0.0],
           "nz/three.stl": [0.0, 0.0, -1.0]}
    args, *_ = build(tmp_path, layout, ups=ups)
    c = Collection.load(args)
    for i in range(len(c.files)):
        up = np.array(c.pose_of(i)["up"])
        want = pose.rotation_to_z_up(up).T @ np.array([1.0, 0.0, 0.0])
        assert np.allclose(c.pose_of(i)["azimuth_zero"], want)
        assert not np.isnan(want).any()


def test_front_is_null_when_the_view_config_has_no_entry(tmp_path):
    """An index cached at another view config is not this run's; the viewer
    falls back to view 0 rather than being handed a wrong camera."""
    args, *_ = build(tmp_path, ["a/one.stl"])          # no front_view written
    c = Collection.load(args)
    assert c.pose_of(0)["front"] is None


def test_front_out_of_range_for_this_config_is_dropped(tmp_path):
    """A front_view of 5 under a 2-view run indexes nothing; better null than
    an IndexError in a request handler."""
    args, *_ = build(tmp_path, ["a/one.stl"], front={"a/one.stl": 5})
    c = Collection.load(args)
    assert c.pose_of(0)["front"] is None


# --- hit --------------------------------------------------------------------

def test_hit_has_the_documented_shape(tmp_path):
    args, root, files = build(tmp_path, ["Kits/Baal/No Supports/x.stl"])
    c = Collection.load(args)
    h = c.hit(0, 0.21, 3.1)
    assert set(h) == {"id", "path", "rel_path", "name", "score", "z", "pose"}
    assert h["rel_path"] == "Kits/Baal/No Supports/x.stl"
    assert h["name"] == "Kits/Baal/x"               # filler dir and .stl dropped
    assert h["path"] == str(files["Kits/Baal/No Supports/x.stl"])
    assert h["id"].startswith("x_") and len(h["id"]) == len("x_") + 6


def test_hit_does_not_validate_the_path(tmp_path):
    """Two caches of one tree drift by design; a moved file is a normal stale
    hit for the caller to drop, not an error and not a stat per hit."""
    args, root, files = build(tmp_path, ["a/one.stl"])
    c = Collection.load(args)
    files["a/one.stl"].unlink()
    h = c.hit(0, 0.1, 1.0)                          # must not raise
    assert h["rel_path"] == "a/one.stl"


# --- the load-once property -------------------------------------------------

def test_a_query_touches_the_filesystem_only_for_the_404_stat(tmp_path, monkeypatch):
    """The reason `n_scanned` reads a cached walk instead of walking: this
    collection lives on spinning exfat where a cold walk is ~32 s, and two
    processes walking one platter contend. If a scope ever calls `iterdir`,
    `walk`, or `stat` per file, that budget is gone."""
    args, *_ = build(tmp_path, ["a/one.stl", "b/two.stl"])
    c = Collection.load(args)

    import os
    from pathlib import Path as P
    for name in ("iterdir", "glob", "rglob"):
        monkeypatch.setattr(P, name, lambda *a, **k: pytest.fail(f"scope called {name}"))
    monkeypatch.setattr(os, "walk", lambda *a, **k: pytest.fail("scope walked"))

    s = c.resolve("a")                  # resolve + exists are the allowed stats
    assert len(s.rows) == 1
    h = c.hit(int(s.rows[0]), 0.5, 2.0)  # and a hit must touch nothing at all
    assert h["pose"] is not None


def test_reload_returns_a_new_instance_and_leaves_the_old_one_usable(tmp_path):
    """The server rebinds a name; an in-flight request keeps the instance it
    started with, so a half-loaded matrix is never visible."""
    args, root, _ = build(tmp_path, ["a/one.stl"])
    first = Collection.load(args)
    assert len(first.files) == 1

    later = build(tmp_path, ["a/one.stl", "a/two.stl"])[0]
    second = first.reload(rescan=True)
    assert second is not first
    assert len(second.files) == 2
    assert len(first.files) == 1 and first.matrix.shape[0] == 1   # untouched


def test_importing_collection_costs_no_torch_or_open3d(tmp_path):
    """implementation.md's row for this module: numpy, pose, cachedir,
    embed_store, query — never a model and never a renderer."""
    import subprocess
    import sys
    from pathlib import Path as P
    repo = P(__file__).resolve().parent.parent
    code = ("import sys; from src import collection; "
            "bad=[k for k in sys.modules if k in ('torch','open3d') "
            "or k.startswith(('torch.','open3d.'))]; "
            "print(','.join(bad)); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=repo,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"forbidden imports: {r.stdout}\n{r.stderr}"
