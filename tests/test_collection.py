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
from pathlib import Path

import numpy as np
import pytest

from src import cachedir
from src import pose
from src.cachedir import (CACHE_VERSION, cache_key, embeds_dir, find_stls,
                          stamp_cache_version, view_config)
from src.collection import (COVERS, CacheUnusable, Collection, NoSuchPath,
                            OutsideCollection, Scope, ScopeError, VirtualPath,
                            VolumeUnavailable)

DIM = 8


def _replace(args, **over):
    return argparse.Namespace(**{**vars(args), **over})


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
            # through the production function, so a fixture with non-default
            # elevations keys its entry the way a real run would
            e["front_view"] = {view_config(args): front[rel]}
        entries[pose.file_identity(f, root)] = e
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    # flat {identity: entry}, exactly what `pose.save_pose_cache` writes
    (cache / "pose-cache.json").write_text(json.dumps(entries))
    # every real cache carries one, and it is what anchors the keys: without
    # it `cache_root` falls back to the input, so a run scoped to a subdirectory
    # would silently re-anchor and the root/input distinction would vanish
    (cache / "run-params.json").write_text(json.dumps(
        {"input": args.input, "collection_root": str(root),
         "views": args.views, "elevations": args.elevations,
         "render_size": args.render_size, "model": args.model,
         "compile": args.compile, "up_axis": args.up_axis}))

    # a real run stamps the key scheme; without it `require_cache_version`
    # correctly reads this fixture as an unmigrated cache (found when that
    # guard was added, 2026-08-19)
    stamp_cache_version(args.cache_dir)

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


def reexport(tmp_path, args, root, f):
    """Re-export one model: new bytes, a newer mtime, a second cache entry.

    The state `build` cannot produce, because it writes each rel exactly once.
    Nothing prunes pose-cache.json, so a re-exported file leaves its old
    identity *and* old `.npy` on disk beside the new pair — 3540 pose entries
    against 3396 loaded models on embed-cache2 (docs/cache-rebuild.md). Both
    halves go through the production writers, like the rest of the fixture.

    Returns the new identity. The new embedding is deliberately not `build`'s
    0.1, so a loader that picks the stale entry is visible in the matrix rather
    than only in the identity list."""
    import os
    f.write_bytes(b"solid x\nendsolid x\nre-exported, and longer\n")
    st = f.stat()
    # a *later whole second*: `identity.mtime_key` truncates to seconds, so a
    # sub-second bump would key identically and never make a second entry
    os.utime(f, (st.st_atime, st.st_mtime + 100))

    cache = Path(args.cache_dir) / "pose-cache.json"
    entries = json.loads(cache.read_text())
    ident = pose.file_identity(f, root)
    entries[ident] = {"up": [0.0, 0.0, 1.0], "confidence": 0.9,
                      "source": "geometry", "margin": 0.5,
                      "v": pose.POSE_CACHE_VERSION}
    cache.write_text(json.dumps(entries))

    token = pose.embed_cache_token(entries[ident], args.up_axis)
    vec = np.full((args.views * len(args.elevations), DIM), 0.2, dtype=np.float32)
    np.save(embeds_dir(args.cache_dir) / f"{cache_key(f, args, token, root)}.npy",
            vec)
    return ident


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


@pytest.mark.parametrize("bad", [
    "a\x00b",                       # ValueError from the null byte
    "x" * 5000,                     # OSError ENAMETOOLONG
    5,                              # TypeError
    b"bytes",                       # TypeError
])
def test_a_malformed_path_is_a_scope_error_not_a_500(tmp_path, bad):
    """`path` is a request-body string, so anything that escapes as itself is
    a 500 from a crafted input. An earlier version caught only OSError — and
    its comment named symlink loops, which `Path.resolve` raises as
    RuntimeError, so it was dead code describing the one case it could not
    catch (review, 2026-08-19)."""
    args, *_ = build(tmp_path, ["a/one.stl"])
    c = Collection.load(args)
    with pytest.raises(ScopeError):
        c.resolve(bad)


def test_a_symlink_loop_is_a_scope_error(tmp_path):
    """The case that comment named. `Path.resolve()` swallows the OSError
    internally and raises RuntimeError, which sailed past."""
    args, root, _ = build(tmp_path, ["a/one.stl"])
    loop = root / "loop"
    loop.symlink_to(root / "loop2")
    (root / "loop2").symlink_to(loop)
    c = Collection.load(args)
    with pytest.raises(ScopeError):
        c.resolve("loop")


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


def test_a_scope_is_matched_by_realpath_not_by_string(tmp_path):
    """The commit that introduced this claimed realpath comparison and nothing
    pinned it (review, 2026-08-19). A symlink is the cheap stand-in for the
    real case: the library remounts as .../STLLibrary1 for the same tree, and
    a string prefix stops matching."""
    args, root, _ = build(tmp_path, ["a/one.stl", "b/two.stl"])
    link = tmp_path / "by-another-name"
    link.symlink_to(root / "a")
    c = Collection.load(args)
    assert c.resolve(str(link)).rows.tolist() == c.resolve("a").rows.tolist()


def test_a_symlinked_input_still_produces_root_relative_paths(tmp_path):
    """Walking a symlink to the collection used to yield absolute `_rel`
    tuples: every subdirectory scope returned zero rows while the collection
    reported them indexed, and `rel_path` came out with a doubled leading
    slash — the documented join key, silently unusable (review, 2026-08-19)."""
    args, root, _ = build(tmp_path, ["Kits/Baal/x.stl"])
    link = tmp_path / "link"
    link.symlink_to(root)
    c = Collection.load(_replace(args, input=str(link)))
    assert c.hit(0, 0.1, 1.0)["rel_path"] == "Kits/Baal/x.stl"
    assert len(c.resolve("Kits").rows) == 1          # not silently unindexed


def test_a_run_scoped_to_a_subdirectory_keeps_root_relative_paths(tmp_path):
    """Cache keys stay anchored at the collection root even when the run walks
    one kit, so `rel_path` must carry the intervening directories."""
    args, root, _ = build(tmp_path, ["Kits/Baal/x.stl", "Kits/Other/y.stl"])
    c = Collection.load(_replace(args, input=str(root / "Kits" / "Baal")))
    assert c.hit(0, 0.1, 1.0)["rel_path"] == "Kits/Baal/x.stl"
    assert len(c.resolve("Kits").rows) == 1          # only what was walked


def test_a_symlink_walked_run_is_readable_through_the_resolved_input(tmp_path):
    """The walk cache is keyed on `inp.resolve()` but stores paths *as walked*,
    so one cache serves every spelling of the root while holding exactly one of
    them. A classify run through a symlink leaves link-spelled entries there,
    and a bare `serve_api.py --cache-dir ...` then loads them with `input`
    taken from run-params.json — which records it **resolved**. Both lexical
    relations in `_parts` fail on that pair, `_rel` fell back to absolute
    tuples, and the damage is total and silent: `row_of` missed for every
    model, so `POST /poses` answered null for all of them with `/status` green,
    and every scope matched nothing. The 2026-08-19 absolute-`_rel` bug
    arriving by a second route (cross-session review of f074334)."""
    args, root, _ = build(tmp_path, ["Kits/Baal/x.stl", "Kits/Other/y.stl"])
    link = tmp_path / "link"
    link.symlink_to(root)
    Collection.load(_replace(args, input=str(link), rescan=True))      # the run
    c = Collection.load(_replace(args, input=str(root), rescan=False))  # the server
    assert all(f.is_relative_to(link) for f in c.files)     # the fixture's premise
    for i, f in enumerate(c.files):
        # the realpath is the only spelling model-browser holds for a model
        assert c.row_of(str(Path(f).resolve())) == i, f
        assert c.row_of(c.hit(i, 0.1, 1.0)["rel_path"]) == i
    assert {c.hit(i, 0.1, 1.0)["rel_path"] for i in range(len(c.files))} == \
        {"Kits/Baal/x.stl", "Kits/Other/y.stl"}
    s = c.resolve("Kits")
    assert len(s.rows) == 2 and s.n_scanned == 2 and s.status == "indexed"
    assert len(c.resolve(str(root / "Kits" / "Baal")).rows) == 1


def test_a_real_path_walked_run_is_readable_through_a_symlinked_input(tmp_path):
    """The other direction of the same matrix. It already worked — the walk
    stores resolved spellings, so `_parts` falls through to
    `f.relative_to(self.root)` — and the repair must leave it that way, which
    is the half a fix keyed on the input alone would have broken."""
    args, root, _ = build(tmp_path, ["Kits/Baal/x.stl", "Kits/Other/y.stl"])
    link = tmp_path / "link"
    link.symlink_to(root)
    Collection.load(_replace(args, input=str(root), rescan=True))      # the run
    c = Collection.load(_replace(args, input=str(link), rescan=False))  # the server
    assert all(f.is_relative_to(root) for f in c.files)     # the fixture's premise
    for i, f in enumerate(c.files):
        assert c.row_of(str(f)) == i, f             # the realpath, as sent
        assert c.row_of(c.hit(i, 0.1, 1.0)["rel_path"]) == i
    assert {c.hit(i, 0.1, 1.0)["rel_path"] for i in range(len(c.files))} == \
        {"Kits/Baal/x.stl", "Kits/Other/y.stl"}
    s = c.resolve("Kits")
    assert len(s.rows) == 2 and s.n_scanned == 2 and s.status == "indexed"
    # `resolve` realpaths, so the link spelling scopes the same rows — the
    # difference from `row_of`, which is lexical and answers None for a third
    # alias by design
    assert c.resolve(str(link / "Kits")).rows.tolist() == s.rows.tolist()


def test_a_scoped_symlink_walked_run_keeps_root_relative_paths(tmp_path):
    """`_prefix` and the realpath repair must not both apply. `_prefix` is the
    *input's* offset from the root and belongs only to the input-relative
    branch; `_real(f).relative_to(self._real_root)` is already root-relative,
    so prepending `_prefix` there would spell `Kits/Baal/Kits/Baal/x.stl` — a
    scope that matches nothing and a join key no consumer can use, which is the
    same failure by the opposite mistake."""
    args, root, _ = build(tmp_path, ["Kits/Baal/x.stl", "Kits/Other/y.stl"])
    link = tmp_path / "link"
    link.symlink_to(root)
    Collection.load(_replace(args, input=str(link / "Kits" / "Baal"), rescan=True))
    c = Collection.load(_replace(args, input=str(root / "Kits" / "Baal"),
                                rescan=False))
    assert len(c.files) == 1 and c.files[0].is_relative_to(link)
    assert c.hit(0, 0.1, 1.0)["rel_path"] == "Kits/Baal/x.stl"
    assert c.row_of(str(root / "Kits" / "Baal" / "x.stl")) == 0
    assert len(c.resolve("Kits").rows) == 1          # only what was walked


def test_a_file_walked_out_of_the_collection_keeps_its_absolute_parts(tmp_path):
    """The terminal fallback the realpath repair must not swallow. A walk that
    followed a symlink out of the tree holds a file with no root-relative name
    at all — realpath cannot heal that one, and 2026-08-19's reason for
    returning something usable rather than raising still stands for it. What it
    must not do is land under a scope inside the root."""
    args, root, _ = build(tmp_path, ["Kits/Baal/x.stl"])
    outside = tmp_path / "elsewhere" / "z.stl"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"solid x\nendsolid x\n")
    Collection.load(args)                            # writes the walk cache
    walk = next((tmp_path / "cache").glob("walk-*.json"))
    saved = json.loads(walk.read_text())
    saved["files"].append(str(outside))          # as a walk out of the tree left it
    walk.write_text(json.dumps(saved))

    c = Collection.load(_replace(args, rescan=False))            # must not raise
    assert c.resolve(None).n_scanned == 2
    assert c.resolve("Kits").n_scanned == 1          # under no scope in the root


# --- in_rel_order: the order a listing is cut in ----------------------------

def _recorded_in_reverse(args):
    """Reverse the order the walk cache records, and load it back.

    Row order is not something the API gets to inherit. `load_file_list`
    replays this file verbatim, so it is whatever the file holds; and
    `find_stls`'s own order is `sorted()` over `Path`s, which compared whole
    strings before Python 3.12 and components since — the two disagree on
    exactly `Kits/A-B` vs `Kits/A`, where `-` sorts under `/`."""
    Collection.load(args)                            # writes the walk cache
    walk = next(Path(args.cache_dir).glob("walk-*.json"))
    saved = json.loads(walk.read_text())
    saved["files"].reverse()
    walk.write_text(json.dumps(saved))
    return Collection.load(_replace(args, rescan=False))


def test_rows_come_back_in_root_relative_order_not_walk_order(tmp_path):
    """`POST /under` cuts a *prefix* of this order at its limit, so which
    models survive the cut depends on it being the component-wise order a
    consumer's tree shows rather than the order the rows are stored in."""
    args, root, _ = build(tmp_path, ["Kits/A-B/x.stl", "Kits/A/y.stl"])
    c = _recorded_in_reverse(args)
    assert [f.name for f in c.files] == ["x.stl", "y.stl"]     # the premise
    rows = c.in_rel_order(c.resolve("Kits").rows)
    assert [c.files[i].name for i in rows] == ["y.stl", "x.stl"]
    assert [type(i) for i in rows] == [int, int]      # ordinary ints, not intp


def test_in_rel_order_keeps_only_the_rows_it_was_given(tmp_path):
    """It sorts a scope, it does not widen one."""
    args, *_ = build(tmp_path, ["Kits/A/y.stl", "other/z.stl"])
    c = Collection.load(args)
    rows = c.in_rel_order(c.resolve("Kits").rows)
    assert [c.files[i].name for i in rows] == ["y.stl"]


# --- listing: one row of POST /under ----------------------------------------

def test_a_listing_entry_is_the_path_and_pose_a_hit_carries(tmp_path):
    """A consumer lists a folder and then hands one of those strings back to
    `/poses` or `/similar`, so the listing's spelling has to be the one every
    other route means by that model — `hit`'s own, from the same
    `Collection.files` entry."""
    args, root, files = build(tmp_path, ["Kits/Baal/x.stl"],
                              ups={"Kits/Baal/x.stl": [0.0, 1.0, 0.0]},
                              front={"Kits/Baal/x.stl": 1})
    c = Collection.load(args)
    e, h = c.listing(0), c.hit(0, 0.1, 1.0)
    assert set(e) == {"path", "pose"}
    assert e["path"] == h["path"] == str(files["Kits/Baal/x.stl"])
    assert e["pose"] == h["pose"] and e["pose"]["up"] == [0.0, 1.0, 0.0]
    assert c.row_of(e["path"]) == 0             # the string the batch joins on


def test_a_listing_entry_carries_a_null_pose_rather_than_dropping_the_model(tmp_path):
    """An indexed model whose pose never resolved is still in the folder."""
    args, root, files = build(tmp_path, ["a/one.stl"])
    c = Collection.load(args)
    c.poses.clear()
    assert c.listing(0) == {"path": str(files["a/one.stl"]), "pose": None}



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


def test_front_is_null_when_no_front_view_was_ever_cached(tmp_path):
    args, *_ = build(tmp_path, ["a/one.stl"])          # no front_view written
    c = Collection.load(args)
    assert c.pose_of(0)["front"] is None


def test_front_cached_under_another_view_config_is_not_used(tmp_path):
    """The state the null actually guards, and the one the previous test could
    not distinguish (review, 2026-08-19): an entry exists, but it indexes some
    other run's view list. An index cached at 8 views is out of range at 4 and
    silently wrong at the same count with different elevations."""
    args, root, files = build(tmp_path, ["a/one.stl"])
    ident = pose.file_identity(files["a/one.stl"], root)
    cache = tmp_path / "cache" / "pose-cache.json"
    entries = json.loads(cache.read_text())
    entries[ident]["front_view"] = {"8v-e20,-20": 3}    # another config entirely
    cache.write_text(json.dumps(entries))
    c = Collection.load(args)
    assert c.view_cfg == "2v-e20-ev2"   # -evN is the framing version, which
                                        # is itself part of "another config"
    assert c.pose_of(0)["front"] is None


def test_front_out_of_range_for_this_config_is_dropped(tmp_path):
    """A front_view of 5 under a 2-view run indexes nothing; better null than
    an IndexError in a request handler."""
    args, *_ = build(tmp_path, ["a/one.stl"], front={"a/one.stl": 5})
    c = Collection.load(args)
    assert c.pose_of(0)["front"] is None


# --- row_of: the batch lookup behind POST /poses ----------------------------

def test_row_of_finds_a_model_by_the_path_a_hit_carries(tmp_path):
    """The guarantee that makes the join work: a consumer holding a hit's
    `path` can ask about that model again without normalising anything."""
    args, root, files = build(tmp_path, ["Kits/Baal/x.stl", "b/two.stl"])
    c = Collection.load(args)
    h = c.hit(0, 0.1, 1.0)
    assert c.row_of(h["path"]) == 0
    assert c.row_of(h["rel_path"]) == 0          # root-relative too


def test_row_of_finds_a_symlink_walked_model_by_its_own_spelling(tmp_path):
    """Why the lookup is keyed twice. A run walked through a symlink to the
    library produces files spelled under the link, so that is the absolute
    `path` every hit carries — and it is lexically under neither the recorded
    root nor the resolved one. Stripping a root would answer None for the very
    path this side handed the caller; the root-relative spelling has to keep
    working alongside it. Same fixture as
    `test_a_symlinked_input_still_produces_root_relative_paths`."""
    args, root, _ = build(tmp_path, ["Kits/Baal/x.stl"])
    link = tmp_path / "link"
    link.symlink_to(root)
    c = Collection.load(_replace(args, input=str(link)))
    h = c.hit(0, 0.1, 1.0)
    assert h["path"].startswith(str(link))       # the fixture's premise
    assert c.row_of(h["path"]) == 0
    assert c.row_of(h["rel_path"]) == 0
    assert c.row_of(str(root / "Kits" / "Baal" / "x.stl")) == 0


def test_row_of_normalises_the_spellings_of_one_path(tmp_path):
    args, root, _ = build(tmp_path, ["a/one.stl"])
    c = Collection.load(args)
    for spelling in ("a/one.stl", "./a/one.stl", "a//one.stl", "a/b/../one.stl",
                     str(root / "a" / "one.stl"), str(root) + "//a/./one.stl"):
        assert c.row_of(spelling) == 0, spelling


@pytest.mark.parametrize("path", [
    "a/nowhere.stl",                    # never walked
    "a",                                # a directory, not a model
    "/etc/passwd",                      # outside the collection
    "a/pack.zip!/inner.stl",            # a zip virtual path
    "",                                 # nothing at all
    "a/\x00one.stl",                    # a null from the wire
])
def test_row_of_answers_none_where_resolve_would_raise(tmp_path, path):
    """`resolve`'s three ScopeErrors are the wrong shape for a batch: one
    unaddressable member must not cost the rest of the request its answer, so
    everything this index does not hold is None rather than an exception."""
    args, *_ = build(tmp_path, ["a/one.stl"])
    c = Collection.load(args)
    assert c.row_of(path) is None


def test_a_full_batch_of_lookups_touches_the_filesystem_not_at_all(tmp_path):
    """The bound on `POST /poses` is 1024 paths, and the whole design rests on
    that costing no I/O: a realpath per path would be ~1024 * depth `lstat`s
    on a volume that may be an HDD, which is exactly the storage-proportional
    request cost this module refuses (module docstring). Asserted as a budget
    rather than as the absence of named calls — that is what let `hit`'s nine
    lstats per result through once."""
    args, root, _ = build(tmp_path, ["a/one.stl", "b/two.stl"])
    c = Collection.load(args)
    batch = ([str(root / "a" / "one.stl"), "b/two.stl", "b/missing.stl"] * 342)[:1024]
    calls = count_syscalls(lambda: [c.row_of(p) for p in batch])
    assert sum(calls.values()) == 0, dict(calls)


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

def count_syscalls(fn):
    """Real syscalls `fn()` makes, by name. Counts `os`-level calls rather than
    patching `Path` methods, because pathlib reaches straight through: an
    earlier version of this test blocked `iterdir`/`glob`/`rglob`/`os.walk`
    and missed `hit` doing nine `lstat`s per result (review, 2026-08-19).

    Restores in a `finally` rather than leaving it to fixture teardown — the
    spies stay installed otherwise and the counter keeps absorbing whatever
    the rest of the test does, including building the next fixture."""
    import collections
    import os
    seen = collections.Counter()
    names = ("stat", "lstat", "scandir", "listdir", "readlink", "open")
    real = {n: getattr(os, n) for n in names}
    for name in names:
        def spy(*a, _n=name, _f=real[name], **k):
            seen[_n] += 1
            return _f(*a, **k)
        setattr(os, name, spy)
    try:
        fn()
    finally:
        for name, f in real.items():
            setattr(os, name, f)
    return seen


def test_a_hit_touches_the_filesystem_not_at_all(tmp_path):
    """A `top=10` query must not pay 90 syscalls on a slow USB volume.

    `hit` used to call `identity.render_key`, which resolves the path and so
    lstats every component — 89 syscalls for ten results on the real
    collection. The key is precomputed at load now, and this asserts the
    budget rather than the absence of a few named walk functions, which is
    what let that through."""
    args, *_ = build(tmp_path, ["a/one.stl", "b/two.stl"])
    c = Collection.load(args)
    calls = count_syscalls(lambda: [c.hit(i, 0.5, 2.0) for i in range(2)])
    assert sum(calls.values()) == 0, dict(calls)


def test_listing_a_whole_directory_touches_the_filesystem_not_at_all(tmp_path):
    """`POST /under` answers a folder out of the store, which is the entire
    reason it exists: the consumer's own walk is what ran out of budget. A
    per-model stat here would put the storage back in the request cost, and
    on a directory the size of a kit that is the failure this route replaces
    arriving from the other side."""
    args, *_ = build(tmp_path, ["Kits/A-B/x.stl", "Kits/A/y.stl", "b/z.stl"])
    c = Collection.load(args)
    rows = c.resolve("Kits").rows
    calls = count_syscalls(lambda: [c.listing(i) for i in c.in_rel_order(rows)])
    assert sum(calls.values()) == 0, dict(calls)


def test_pose_of_touches_the_filesystem_not_at_all(tmp_path):
    """It must not re-read pose-cache.json per call — the poses are in memory."""
    args, *_ = build(tmp_path, ["a/one.stl", "b/two.stl"])
    c = Collection.load(args)
    calls = count_syscalls(lambda: [c.pose_of(i) for i in range(2)])
    assert sum(calls.values()) == 0, dict(calls)


def test_a_scope_costs_a_bounded_handful_of_syscalls(tmp_path):
    """The budget is small and constant — a `resolve()` of the scope path plus
    one `exists()` — and explicitly *not* proportional to the collection, so
    adding a per-file `stat` anywhere fails here rather than in production.

    Note what this is *not* justified by: an earlier version cited a ~32 s cold
    walk, borrowed from another repo's measurement of a spinning drive. This
    library is ext4 on an SSD and walks in 0.07 s. The rule earns its keep by
    making request cost independent of the storage — the library may yet live
    on an HDD, and this budget holds there unchanged — and on `n_scanned` being
    a stable claim about the index rather than about the tree right now."""
    # both collections at the same directory depth: `resolve` lstats one
    # component per level, so depth is a legitimate cost and file count is not
    args, *_ = build(tmp_path / "small", ["a/one.stl", "a/two.stl", "b/four.stl"])
    c = Collection.load(args)
    calls = count_syscalls(lambda: c.resolve("a"))
    assert sum(calls.values()) < 16, dict(calls)
    assert calls["scandir"] == calls["listdir"] == 0, dict(calls)

    args2, *_ = build(tmp_path / "big", [f"a/m{i}.stl" for i in range(40)] + ["b/x.stl"])
    c2 = Collection.load(args2)
    big = count_syscalls(lambda: c2.resolve("a"))
    assert sum(big.values()) == sum(calls.values()), (dict(calls), dict(big))


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


def _mangle(tmp_path, files, root, ident_of, change):
    cache = tmp_path / "cache" / "pose-cache.json"
    entries = json.loads(cache.read_text())
    ident = pose.file_identity(files[ident_of], root)
    if change == "drop-up":
        entries[ident].pop("up")
    else:
        entries[ident].update(change)
    cache.write_text(json.dumps(entries))
    return ident


@pytest.mark.parametrize("change", [
    {"up": None},                       # not iterable
    {"up": [0.0, 0.0]},                 # wrong length
    {"up": ["x", "y", "z"]},            # not numbers
    {"up": [0.0, 0.0, 0.0]},            # no rotation takes nothing to +Z
    {"up": [float("nan"), 0.0, 1.0]},   # json.loads accepts a bare NaN
    "drop-up",                          # missing key -> KeyError
])
def test_an_unusable_up_does_not_break_the_load(tmp_path, change):
    """pose-cache.json is hand-editable and `load_pose_cache` validates only
    `v`. One bad entry used to kill the whole *load*, not just a hit:
    `embed_store` asks `pose.embed_cache_token` for every file's token and
    that read `entry["up"]` unguarded, so a single null crashed the process
    (review, 2026-08-19).

    An entry with no usable up carries no pose, so its embedding keys under
    the "unresolved" token and simply is not in the index — the model is
    dropped, counted in `missing`, and everything else loads."""
    args, root, files = build(tmp_path, ["a/one.stl", "a/two.stl"])
    _mangle(tmp_path, files, root, "a/one.stl", change)

    c = Collection.load(args)                            # must not raise
    assert len(c.files) == 1 and c.missing == 1
    assert c.files[0].name == "two.stl"
    assert c.pose_of(0) is not None                      # the survivor is fine


@pytest.mark.parametrize("change", [
    {"up": None}, {"up": [0.0, 0.0]}, "drop-up",
    # zero and NaN pass a float/length check but blow up in rotation_to_z_up,
    # which raised out of `pose_of` and 500ed the whole /query response — one
    # hand-edited entry against the very promise this test names (review,
    # 2026-08-20)
    {"up": [0.0, 0.0, 0.0]}, {"up": [float("nan"), 0.0, 1.0]},
])
def test_pose_of_returns_null_for_a_malformed_entry(tmp_path, change):
    """The other half: an entry that is malformed *after* the row was indexed
    — the cache edited under a running server — must yield a null pose rather
    than failing the query response that contains it."""
    args, root, files = build(tmp_path, ["a/one.stl"])
    c = Collection.load(args)
    entry = c.poses[c._ident[0]]
    if change == "drop-up":
        entry.pop("up")
    else:
        entry.update(change)
    assert c.pose_of(0) is None
    assert c.hit(0, 0.1, 1.0)["pose"] is None            # the hit still forms


def test_a_null_confidence_keeps_the_pose_and_defaults_the_number(tmp_path):
    """A missing confidence is not a missing pose: the up vector is what the
    viewer needs, and discarding the orientation over an absent score would
    lose more than it protects.

    Mutated *after* the load, because since adversarial review pass 4
    (2026-08-31) `pose._readable` drops a non-numeric `confidence` at the
    door — `Pose.from_cache`'s `float(d.get("confidence", 0.0))` raises on
    it — so no such entry survives `Collection.load` to be asked here (the
    drop is pinned in `tests/test_pose.py`). The belt this asserts is what
    stays: never raising is a property of `pose_of` itself, not of whichever
    loader filled `self.poses`."""
    args, root, files = build(tmp_path, ["a/one.stl"])
    c = Collection.load(args)
    c.poses[c._ident[0]]["confidence"] = None
    p = c.pose_of(0)
    assert p is not None and p["confidence"] == 0.0
    assert p["up"] == [0.0, 0.0, 1.0]


# --- the cache itself -------------------------------------------------------

def test_a_corrupt_run_params_is_cache_unusable_not_a_crash(tmp_path):
    """`cache_root` json.loads run-params.json, and it ran *before* the
    CacheUnusable boundary — so a torn manifest escaped `POST /reload` as a
    bare JSONDecodeError 500 while `/status` kept reporting the stale success
    (review, 2026-08-20)."""
    args, *_ = build(tmp_path, ["a/one.stl"])
    (Path(args.cache_dir) / "run-params.json").write_text("{ torn")
    with pytest.raises(CacheUnusable):
        Collection.load(args)


@pytest.mark.parametrize("content", ["[]", "null", '"poses"'])
def test_a_non_object_pose_cache_is_cache_unusable(tmp_path, content):
    """A list- or null-shaped pose-cache.json raised AttributeError from
    `raw.items()`, which no except clause named (review, 2026-08-20). The
    file is hand-editable; its shape is a state of the cache, not a bug."""
    args, *_ = build(tmp_path, ["a/one.stl"])
    (Path(args.cache_dir) / "pose-cache.json").write_text(content)
    with pytest.raises(CacheUnusable):
        Collection.load(args)


def test_a_null_pose_entry_is_dropped_like_a_stale_one(tmp_path):
    """One bad *entry* costs one model, never the load: a null value crashed
    `v.get("v")` before it could be filtered (review, 2026-08-20)."""
    args, root, files = build(tmp_path, ["a/one.stl", "a/two.stl"])
    cache = Path(args.cache_dir) / "pose-cache.json"
    entries = json.loads(cache.read_text())
    entries[pose.file_identity(files["a/one.stl"], root)] = None
    cache.write_text(json.dumps(entries))

    c = Collection.load(args)                            # must not raise
    assert len(c.files) == 1 and c.missing == 1
    assert c.files[0].name == "two.stl"


def test_an_empty_cache_raises_instead_of_exiting_the_process(tmp_path):
    """`embed_store` raises SystemExit, which is a BaseException and walks
    straight through the `except Exception` a web framework wraps handlers in
    — and `POST /reload` runs this inside one (review, 2026-08-19)."""
    import shutil
    args, *_ = build(tmp_path, ["a/one.stl"])
    c = Collection.load(args)
    shutil.rmtree(embeds_dir(args.cache_dir))
    with pytest.raises(CacheUnusable):
        c.reload()
    try:                                    # and it is an ordinary Exception
        c.reload()
    except Exception as e:
        assert isinstance(e, CacheUnusable)


@pytest.mark.parametrize("content", ["", "{not json", '{"files": '])
def test_a_torn_walk_cache_is_unusable_not_a_crash(tmp_path, content):
    """`POST /reload {"rescan": true}` writes this file from a request handler
    while `classify_stls.py` may be writing it too. A torn file is not merely a
    stale list — it is a JSONDecodeError on every later read, which reached the
    handler as a bare 500 with none of the 503 envelope (review, 2026-08-19)."""
    args, *_ = build(tmp_path, ["a/one.stl"])
    Collection.load(args)
    walk = next(Path(args.cache_dir).glob("walk-*.json"))
    walk.write_text(content)
    with pytest.raises(CacheUnusable) as e:
        Collection.load(_replace(args, rescan=False))
    assert "unreadable cache" in e.value.message


@pytest.mark.parametrize("content", ["", "{trunc", '{"other": 1}', "not json"])
def test_a_corrupt_stamp_reads_as_unstamped_rather_than_raising(tmp_path, content):
    """`cache_version` is called by `/status`, the route that exists to explain
    a server which cannot start — so an exception there made the diagnostic
    route the first to fail on a broken cache (review, 2026-08-19). Reading
    corrupt as 0 keeps it total.

    What 0 then *means* changed with the 2026-08-31 rebuild (§4): it named the
    pre-stamp key scheme and was refused; now it means only "not stamped yet",
    so a torn stamp is rewritten and the load goes through. That is the right
    answer for a file whose only job is to record the one live scheme — the
    keys underneath it are that scheme whatever the stamp says."""
    from src.cachedir import CACHE_VERSION, cache_version
    args, *_ = build(tmp_path, ["a/one.stl"])
    (Path(args.cache_dir) / "cache-meta.json").write_text(content)
    assert cache_version(args.cache_dir) == 0
    Collection.load(args)                           # no longer refused
    assert cache_version(args.cache_dir) == CACHE_VERSION       # re-stamped


def test_the_walk_cache_is_written_atomically(tmp_path):
    """temp + os.replace, the treatment `Done.flush` gives the pose cache: a
    reader never sees a partial file, so the case above cannot be *caused* by
    this project writing it."""
    args, *_ = build(tmp_path, ["a/one.stl", "b/two.stl"])
    cache = Path(args.cache_dir)
    for f in cache.glob("walk-*.json"):
        f.unlink()
    Collection.load(_replace(args, rescan=True))            # writes it fresh
    assert list(cache.glob("walk-*.json"))
    assert not list(cache.glob("*.tmp")), "temp file left behind"


def test_an_old_key_scheme_is_named_with_the_right_fix(tmp_path, monkeypatch):
    """The guard every other cache consumer calls
    (`cachedir.require_cache_version`) and this module did not. Without it a cache from
    an older key scheme misses on every lookup and reports "run
    classify_stls.py first", when the actionable line is migrate_cache_keys —
    the exact wrong-advice shape VolumeUnavailable exists to prevent.

    The mismatch has to be staged by moving `CACHE_VERSION` rather than by
    writing `CACHE_VERSION - 1`: since the 2026-08-31 rebuild (§4) that
    literal is 0, which now means "not stamped yet" and is stamped rather than
    refused. Bumping the constant is also the real shape of the next
    occurrence — new code, a cache stamped by the old."""
    args, *_ = build(tmp_path, ["a/one.stl"])
    Collection.load(args)                                # stamps it current
    monkeypatch.setattr(cachedir, "CACHE_VERSION", CACHE_VERSION + 1)
    with pytest.raises(CacheUnusable) as e:
        Collection.load(args)
    assert "cache_version" in e.value.message
    assert e.value.hint and "migrate_cache_keys" in e.value.hint
    assert "classify_stls" not in (e.value.hint or "")


def test_covers_agrees_with_what_the_classifier_actually_walks(tmp_path):
    """COVERS names `find_stls`'s rule as its source of truth with no
    mechanical link (review, 2026-08-19). It matches case-insensitively, so
    the published extension is the family, not the literal spelling."""
    root = tmp_path / "walk"
    (root / "d").mkdir(parents=True)
    for name in ("a.stl", "b.STL", "c.Stl", "d.3mf", "e.obj", "f.txt"):
        (root / "d" / name).write_bytes(b"x")
    found = {f.name for f in find_stls(root)}
    assert found == {"a.stl", "b.STL", "c.Stl"}          # case-insensitive
    assert COVERS == ["stl"]
    assert all(n.lower().endswith("." + COVERS[0]) for n in found)


# --- the volume ------------------------------------------------------------

def test_an_absent_volume_raises_rather_than_reading_as_an_empty_cache(tmp_path, capsys):
    """The library lives on removable media. Unmounted, `load_file_list` drops
    every entry and `load_embedding_matrix` says "no cached embeddings found —
    run classify_stls.py first", which is wrong twice: the embeddings are
    intact and local, and re-running would not help. Fail early, and say so."""
    import shutil
    args, root, _ = build(tmp_path, ["a/one.stl"])
    Collection.load(args)                       # loads while it is there
    shutil.rmtree(root)                         # "unplug the drive"

    with pytest.raises(VolumeUnavailable) as e:
        Collection.load(args)
    # `required` is true by construction: only the mode that requires the
    # volume can raise this, which is what makes `present: false` actionable
    assert e.value.as_dict() == {"present": False, "root": str(root),
                                 "missing": str(root), "required": True}
    out = capsys.readouterr().out
    assert "not available" in out and str(root) in out
    assert "intact and local" in out            # the console explanation
    assert "classify_stls" not in out           # never the misleading advice


def test_a_present_volume_is_reported_as_present(tmp_path):
    args, root, _ = build(tmp_path, ["a/one.stl"])
    assert Collection.load(args).volume == {"present": True, "root": str(root),
                                            "missing": None, "required": True}


def test_a_missing_input_under_a_mounted_volume_says_something_different(tmp_path, capsys):
    """Mounted drive, deleted scope directory — a different problem with a
    different fix, and the message must not blame the volume."""
    import shutil
    args, root, _ = build(tmp_path, ["a/one.stl"], input=str(tmp_path / "stl" / "a"))
    shutil.rmtree(root / "a")
    with pytest.raises(VolumeUnavailable) as e:
        Collection.load(args)
    assert e.value.missing == root / "a" and e.value.root == root
    out = capsys.readouterr().out
    assert "input path is not available" in out and "is mounted" in out


def test_the_volume_check_precedes_the_walk(tmp_path):
    """Order matters: the walk's own failure is a silent zero-file result that
    reads as an empty cache, so the check has to come first."""
    import shutil
    args, root, _ = build(tmp_path, ["a/one.stl"])
    shutil.rmtree(root)
    with pytest.raises(VolumeUnavailable):      # not SystemExit from embed_store
        Collection.load(args)


# --- serving without the volume (--no-volume) -------------------------------

@pytest.mark.parametrize("layout", [
    ["a/one.stl", "b/two.stl", "b/c/three.stl"],
    # the pair `find_stls`' sort orders differently from a plain string sort:
    # '-' is below '/', so "Kits/A-B" precedes "Kits/A" as a string and follows
    # it as parts. A real collection shape (test_rows_come_back_in_rel_order),
    # and the one that decides whether the two loaders agree at all.
    ["Kits/A-B/x.stl", "Kits/A/y.stl", "Kits/AB/z.stl"],
    # a filename holding the identity's own separator: `rel|mtime|size` is
    # split from the right precisely because rel may contain "|" and the two
    # numeric fields never do. A left split reads this file's name as a
    # directory and the row then names a path nothing walked.
    ["a/we|rd.stl", "a/plain.stl"],
])
def test_a_manifest_load_is_the_same_index_as_a_walked_one(tmp_path, layout):
    """The claim the mode rests on: pose-cache.json plus run-params.json
    rebuild every embedding key with no filesystem access (identity.py's
    `cache_key_from_identity`, review §P3.1), so the index built from the
    records is the index built from the volume — same rows, same order, same
    keys, byte for byte."""
    args, root, _ = build(tmp_path, layout)
    walked = Collection.load(args)
    manifest = Collection.load(_replace(args, no_volume=True))

    assert np.array_equal(walked.matrix, manifest.matrix)
    assert manifest.files == walked.files
    assert manifest._keys == walked._keys
    assert manifest._ident == walked._ident
    assert manifest.hit(0, 0.5, 2.0) == walked.hit(0, 0.5, 2.0)


def test_a_re_exported_file_is_one_row_at_its_newest_identity(tmp_path):
    """Nothing prunes pose-cache.json, so iterating it sees identities a walk
    cannot: a re-exported file has two, and the old `.npy` is still there.

    Two rows for one file is not a cosmetic duplicate. `Collection`'s
    `_row_by_rel`/`_row_by_path` collision rule holds only because colliding
    rows share a pose entry, an embedding key and a render key — two exports of
    one model share none of those — so `row_of` (and `POST /poses`) would
    answer from whichever row won insertion, and the model would appear twice
    in one result list under two different scores. The walk-driven loader keys
    the current stat and gets one row; this one has to reach the same answer by
    picking the newest identity."""
    args, root, files = build(tmp_path, ["a/one.stl", "b/two.stl"])
    fresh = reexport(tmp_path, args, root, files["a/one.stl"])

    walked = Collection.load(args)
    manifest = Collection.load(_replace(args, no_volume=True))
    assert len(manifest.files) == 2                  # not three
    assert manifest.files == walked.files
    assert np.array_equal(manifest.matrix, walked.matrix)
    assert manifest._ident == walked._ident and fresh in manifest._ident
    assert manifest.missing == 1                     # superseded, not absent
    # the row really is the new export: `build` writes 0.1, `reexport` 0.2
    row = manifest.files.index(files["a/one.stl"])
    assert manifest.matrix[row].min() == pytest.approx(0.2)


def test_a_manifest_load_survives_the_volume_going_away(tmp_path):
    """The whole point, stated against the refusal it is the opt-in for: the
    same load that raises `VolumeUnavailable` with the drive unplugged answers
    with the same rows under the flag. The `.npy` files and the pose cache are
    intact and local; only the STL library is gone."""
    import shutil
    args, root, _ = build(tmp_path, ["a/one.stl", "b/two.stl"])
    before = Collection.load(_replace(args, no_volume=True))
    shutil.rmtree(root)                         # "unplug the drive"

    after = Collection.load(_replace(args, no_volume=True))
    assert after.files == before.files
    assert np.array_equal(after.matrix, before.matrix)
    assert after.volume == {"present": None, "root": str(root),
                            "missing": None, "required": False}
    with pytest.raises(VolumeUnavailable):      # and the default still refuses
        Collection.load(args)


def test_a_pose_entry_with_no_embedding_is_dropped_and_counted(tmp_path):
    """`missing` keeps its meaning across the two loaders: known to the cache,
    not in the index. Here it is the pose-only model — resolved, never
    embedded, or embedded under other args — where the walked loader's is the
    walked-but-unembedded file."""
    args, root, files = build(tmp_path, ["a/one.stl", "a/two.stl"])
    full = Collection.load(_replace(args, no_volume=True))
    assert len(full.files) == 2 and full.missing == 0

    poses = pose.load_pose_cache(args.cache_dir)
    ident = pose.file_identity(files["a/one.stl"], root)
    token = pose.embed_cache_token(poses[ident], args.up_axis)
    key = cache_key(files["a/one.stl"], args, token, root)
    (embeds_dir(args.cache_dir) / f"{key}.npy").unlink()

    c = Collection.load(_replace(args, no_volume=True))
    assert [f.name for f in c.files] == ["two.stl"] and c.missing == 1


def test_a_cache_with_no_poses_names_the_reason_it_cannot_be_served(tmp_path):
    """A cache built entirely under a forced `--up-axis` records no poses —
    the flag *is* the up vector, so nothing is written — and manifest mode has
    then nothing to enumerate. The advice must say that rather than the
    embedding loader's "run classify_stls.py first", which is the wrong-advice
    shape `VolumeUnavailable` was invented to stop."""
    args, *_ = build(tmp_path, ["a/one.stl"])
    (Path(args.cache_dir) / "pose-cache.json").write_text("{}")

    with pytest.raises(CacheUnusable) as e:
        Collection.load(_replace(args, no_volume=True))
    assert "pose entries" in e.value.message and "--no-volume" in e.value.message
    assert "up-axis" in e.value.message
    assert e.value.hint and "--no-volume" in e.value.hint


def test_a_manifest_scope_answers_from_the_index_not_from_a_stat(tmp_path):
    """With no disk to ask, a scope exists iff it prefixes something indexed.
    `OutsideCollection` is unaffected — it never touched the disk — and the
    root spelled out is not a 404, since a `None` path is the only spelling
    that short-circuits."""
    import shutil
    args, root, _ = build(tmp_path, ["a/one.stl", "b/two.stl"])
    c = Collection.load(_replace(args, no_volume=True))
    shutil.rmtree(root)

    assert c.resolve("a").status == "indexed"
    assert len(c.resolve("a").rows) == 1
    assert len(c.resolve(str(root)).rows) == 2          # the root, spelled out
    assert len(c.resolve(None).rows) == 2
    with pytest.raises(NoSuchPath):                     # under root, no rows
        c.resolve("nowhere")
    with pytest.raises(OutsideCollection):
        c.resolve(str(tmp_path / "elsewhere"))


def test_a_manifest_scope_404s_where_a_walked_one_says_unindexed(tmp_path):
    """The documented cost of the mode (surface.md §scope): "a real directory
    with nothing classified" is a 200 saying `unindexed` only because a stat
    can tell it from "no such path". Without the volume the two are one
    answer, and the collapse is to the 404."""
    args, root, _ = build(tmp_path, ["a/one.stl", "b/two.stl"], embed=["a/one.stl"])
    walked = Collection.load(args)
    assert walked.resolve("b").status == "unindexed"    # the answer, not an error

    manifest = Collection.load(_replace(args, no_volume=True))
    with pytest.raises(NoSuchPath):
        manifest.resolve("b")


def test_importing_collection_costs_no_torch_or_open3d(tmp_path):
    """interfaces.md's row for this module: numpy, `pose`, `cachedir`,
    `embed_store`, `identity` — never a model and never a renderer. (It does
    not import `query`; scoring is the caller's.)"""
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
