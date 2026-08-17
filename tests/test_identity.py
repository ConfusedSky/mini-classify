import argparse
import os

from src import identity
from src import pose
from classify_stls import cache_key, render_key


def args(**kw):
    base = dict(render_size=384, views=8, elevations=[20.0, -20.0],
                model="google/siglip2-so400m-patch14-384")
    base.update(kw)
    return argparse.Namespace(**base)


def collection(base, name="Kit I/model.stl", content=b"solid", mtime=1_700_000_000):
    """One model at a fixed relative path, with a pinned mtime.

    The mtime is pinned because it is part of the identity: two copies of a
    library only share keys if the copy preserved mtimes."""
    f = base / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(content)
    os.utime(f, (mtime, mtime))
    return f


# --- the point of the module: a moved library keeps its keys -----------------

def test_every_key_survives_a_move_between_roots(tmp_path):
    a, b = tmp_path / "driveA" / "STL", tmp_path / "driveB" / "STL"
    fa, fb = collection(a), collection(b)
    args_ = args()
    assert pose.file_identity(fa, a) == pose.file_identity(fb, b)
    assert render_key(fa, a) == render_key(fb, b)
    assert cache_key(fa, args_, "auto", a) == cache_key(fb, args_, "auto", b)


def test_an_absolute_key_would_not_have_survived_it(tmp_path):
    # guards the regression: if the keys go back to absolute paths, the
    # assertions above start passing for the wrong reason
    a, b = tmp_path / "driveA" / "STL", tmp_path / "driveB" / "STL"
    fa, fb = collection(a), collection(b)
    assert str(fa.resolve()) != str(fb.resolve())


def test_a_move_that_drops_mtimes_still_invalidates(tmp_path):
    # `cp` without -a rewrites mtime, and an edited file really is a different
    # model — so this must miss, and the docs have to say "preserve mtimes"
    a, b = tmp_path / "driveA" / "STL", tmp_path / "driveB" / "STL"
    fa = collection(a)
    fb = collection(b, mtime=1_700_009_999)
    assert pose.file_identity(fa, a) != pose.file_identity(fb, b)
    # renders key on the path alone, so those do survive a careless copy
    assert render_key(fa, a) == render_key(fb, b)


def test_moving_a_file_within_the_collection_changes_its_key(tmp_path):
    # relocating the *root* is free; reorganising inside it is not, and should
    # not be — the model genuinely has a new identity in the collection
    root = tmp_path / "STL"
    one = collection(root, "Kit I/model.stl")
    two = collection(root, "Kit II/model.stl")
    assert render_key(one, root) != render_key(two, root)
    assert pose.file_identity(one, root) != pose.file_identity(two, root)


# --- rel_path / collection_root ---------------------------------------------

def test_rel_path_is_root_relative_and_posix(tmp_path):
    f = collection(tmp_path)
    assert identity.rel_path(f, tmp_path) == "Kit I/model.stl"


def test_rel_path_normalises_the_spelling_of_the_path(tmp_path):
    # writer and readers must agree even when one was handed an unnormalised path
    f = collection(tmp_path)
    odd = tmp_path / "Kit I" / ".." / "Kit I" / "model.stl"
    assert identity.rel_path(odd, tmp_path) == identity.rel_path(f, tmp_path)


def test_a_file_outside_the_root_falls_back_to_absolute(tmp_path):
    # reachable through a symlink out of the tree: still a stable key, just not
    # relocatable. Returning something usable beats killing the run.
    outside = collection(tmp_path / "elsewhere")
    root = tmp_path / "STL"
    root.mkdir()
    assert identity.rel_path(outside, root) == outside.resolve().as_posix()


def test_collection_root_of_a_directory_is_itself(tmp_path):
    (tmp_path / "STL").mkdir()
    assert identity.collection_root(tmp_path / "STL") == (tmp_path / "STL").resolve()


def test_a_root_that_does_not_exist_yet_is_still_a_root(tmp_path):
    # an unmounted drive, or a typo: taking the parent here would anchor the
    # whole cache one level up from what was asked for, silently
    missing = tmp_path / "not" / "mounted" / "STL"
    assert identity.collection_root(missing) == missing


def test_collection_root_of_a_single_file_is_its_parent(tmp_path):
    # `classify_stls.py model.stl` has no collection; the parent stands in so
    # the key stays stable while the file sits beside its neighbours
    f = collection(tmp_path, "loose.stl")
    assert identity.collection_root(f) == tmp_path.resolve()


# --- the properties the old keys had, which must not regress ------------------

def test_render_key_still_separates_models_sharing_a_filename(tmp_path):
    root = tmp_path / "STL"
    a = collection(root, "Kit I/Baal_Flaming_Sword_L.stl")
    b = collection(root, "Kit II/Baal_Flaming_Sword_L.stl")
    assert render_key(a, root) != render_key(b, root)
    assert render_key(a, root).startswith("Baal_Flaming_Sword_L_")


def test_cache_key_still_separates_render_configs(tmp_path):
    root = tmp_path / "STL"
    f = collection(root)
    assert cache_key(f, args(), "auto", root) != cache_key(f, args(views=4), "auto", root)
    assert cache_key(f, args(), "auto", root) != cache_key(f, args(), "vlm:0,0,1", root)


# --- mtime truncation: the cross-filesystem half of portability -------------

def test_mtime_survives_an_exfat_round_trip(tmp_path):
    # exFAT stores 10ms granularity, ext4 nanoseconds. A file written on ext4
    # and copied back to exFAT truncates, and full st_mtime_ns would lose it.
    f = collection(tmp_path)
    ext4_ns = 1_700_000_000_642_647_652     # arbitrary ns, as ext4 records
    exfat_ns = ext4_ns // 10_000_000 * 10_000_000   # what exFAT can store
    os.utime(f, ns=(ext4_ns, ext4_ns))
    on_ext4 = pose.file_identity(f, tmp_path)
    os.utime(f, ns=(exfat_ns, exfat_ns))
    assert pose.file_identity(f, tmp_path) == on_ext4


def test_a_real_edit_still_moves_the_key(tmp_path):
    # truncation must not cost the point of having mtime in the identity
    f = collection(tmp_path)
    before = pose.file_identity(f, tmp_path)
    os.utime(f, (1_700_003_600, 1_700_003_600))
    assert pose.file_identity(f, tmp_path) != before


# --- anchor edge cases ------------------------------------------------------

def test_a_loose_file_never_moves_the_anchor(tmp_path):
    # classify_stls.py /tmp/download.stl against the library cache must not
    # re-anchor the whole cache at /tmp
    lib = tmp_path / "STL"
    lib.mkdir()
    loose = tmp_path / "elsewhere" / "download.stl"
    loose.parent.mkdir()
    loose.touch()
    assert identity.resolve_root(loose, lib.resolve()) == (lib.resolve(), None)


def test_the_library_growing_upward_is_re_keyable_not_a_mismatch(tmp_path):
    # first run was one kit, now the run covers the whole library
    lib = tmp_path / "STL"
    kit = lib / "Loot Studios"
    kit.mkdir(parents=True)
    root, note = identity.resolve_root(lib, kit.resolve())
    assert (root, note) == (lib.resolve(), "superdir")


def test_growing_upward_only_needs_a_prefix_on_the_old_keys(tmp_path):
    # what makes "superdir" recoverable: the old key is the new key minus the
    # directories between the two roots
    lib = tmp_path / "STL"
    kit = lib / "Loot Studios"
    f = collection(kit, "Kit/model.stl")
    assert identity.rel_path(f, lib.resolve()) == \
        "Loot Studios/" + identity.rel_path(f, kit.resolve())
