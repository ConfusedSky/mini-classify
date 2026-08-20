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
    assert c.view_cfg == "2v-e20"
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


@pytest.mark.parametrize("change", [{"up": None}, {"up": [0.0, 0.0]}, "drop-up"])
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
    lose more than it protects."""
    args, root, files = build(tmp_path, ["a/one.stl"])
    _mangle(tmp_path, files, root, "a/one.stl", {"confidence": None})
    p = Collection.load(args).pose_of(0)
    assert p is not None and p["confidence"] == 0.0
    assert p["up"] == [0.0, 0.0, 1.0]


# --- the cache itself -------------------------------------------------------

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
    corrupt as 0 is the safe direction: `require_cache_version` then refuses a
    populated cache rather than vouching for keys it cannot read."""
    from src.cachedir import cache_version
    args, *_ = build(tmp_path, ["a/one.stl"])
    (Path(args.cache_dir) / "cache-meta.json").write_text(content)
    assert cache_version(args.cache_dir) == 0
    with pytest.raises(CacheUnusable):              # and the guard still bites
        Collection.load(args)


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


def test_an_old_key_scheme_is_named_with_the_right_fix(tmp_path):
    """The guard every other cache consumer calls (classify_stls.py:255,
    test_categories.py:104) and this module did not. Without it a cache from
    an older key scheme misses on every lookup and reports "run
    classify_stls.py first", when the actionable line is migrate_cache_keys —
    the exact wrong-advice shape VolumeUnavailable exists to prevent."""
    args, *_ = build(tmp_path, ["a/one.stl"])
    Collection.load(args)                                # stamps it current
    (Path(args.cache_dir) / "cache-meta.json").write_text(
        json.dumps({"cache_version": CACHE_VERSION - 1}))
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
    assert e.value.as_dict() == {"present": False, "root": str(root),
                                 "missing": str(root)}
    out = capsys.readouterr().out
    assert "not available" in out and str(root) in out
    assert "intact and local" in out            # the console explanation
    assert "classify_stls" not in out           # never the misleading advice


def test_a_present_volume_is_reported_as_present(tmp_path):
    args, root, _ = build(tmp_path, ["a/one.stl"])
    assert Collection.load(args).volume == {"present": True, "root": str(root),
                                            "missing": None}


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
