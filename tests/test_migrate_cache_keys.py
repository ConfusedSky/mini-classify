import argparse
import json
import os
import shutil

import numpy as np
import pytest
from PIL import Image

import identity
import pose
from classify_stls import EMBEDS_SUBDIR, RENDERS_SUBDIR, cache_key, render_key
from migrate_cache_keys import (old_cache_key, old_identity, old_render_key,
                                plan_embeds, plan_poses, plan_renders, move_all)

CFG = "384px-8v-e20,-20"


def args(**kw):
    base = dict(render_size=384, views=8, elevations=[20.0, -20.0],
                model="siglip", up_axis="auto")
    base.update(kw)
    return argparse.Namespace(**base)


def model(root, name="Kit/model.stl", mtime=1_700_000_000):
    f = root / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"solid")
    # a nanosecond mtime, as ext4 records and the old keys captured
    os.utime(f, ns=(mtime * 10**9 + 642_647_652,) * 2)
    return f


def old_cache(tmp_path, files, root, a, *, absolute=True, source="ensemble"):
    """A cache in the pre-migration shape: absolute keys, flat .npy, renders
    in a directory of their own."""
    cache, renders = tmp_path / "cache", tmp_path / "my_renders"
    cache.mkdir(exist_ok=True)
    entries = {}
    for f in files:
        entry = {"up": [0.0, 0.0, 1.0], "confidence": 0.1, "source": source,
                 "margin": 0.9, "v": pose.POSE_CACHE_VERSION, "front_view": 3}
        entries[old_identity(f, root, root, absolute)] = entry
        token = pose.embed_cache_token(entry, a.up_axis)
        np.save(cache / f"{old_cache_key(f, a, token, root, root, absolute)}.npy",
                np.zeros((16, 4), dtype=np.float32))
        d = renders / CFG
        d.mkdir(parents=True, exist_ok=True)
        for tail in ("view0", "view1", "pose"):
            Image.new("RGB", (8, 8)).save(
                d / f"{old_render_key(f, root, root, absolute)}_{tail}.jpg")
    (cache / "pose-cache.json").write_text(json.dumps(entries))
    return cache, renders


# --- poses ------------------------------------------------------------------

def test_poses_are_re_keyed_onto_the_relative_identity(tmp_path):
    root = tmp_path / "STL"
    f = model(root)
    cache, _ = old_cache(tmp_path, [f], root, args())
    rekeyed, old, dropped = plan_poses([f], cache, root, root, True)
    assert list(rekeyed) == [pose.file_identity(f, root)]
    assert not dropped
    assert rekeyed[pose.file_identity(f, root)]["front_view"] == 3


def test_entries_matching_no_file_are_dropped(tmp_path):
    # load_pose_cache filters on version alone, so these would ride along forever
    root = tmp_path / "STL"
    f, gone = model(root), model(root, "Kit/deleted.stl")
    cache, _ = old_cache(tmp_path, [f, gone], root, args())
    gone.unlink()
    rekeyed, _, dropped = plan_poses([f], cache, root, root, True)
    assert len(rekeyed) == 1 and len(dropped) == 1


def test_a_re_run_re_keys_nothing_and_drops_nothing(tmp_path):
    root = tmp_path / "STL"
    f = model(root)
    cache, _ = old_cache(tmp_path, [f], root, args())
    rekeyed, _, _ = plan_poses([f], cache, root, root, True)
    (cache / "pose-cache.json").write_text(json.dumps(rekeyed))
    again, _, dropped = plan_poses([f], cache, root, root, True)
    assert again == rekeyed and not dropped


# --- the library moved ------------------------------------------------------

def test_a_moved_library_keeps_every_entry(tmp_path):
    old_root, new_root = tmp_path / "driveA", tmp_path / "driveB"
    f_old, f_new = model(old_root), model(new_root)
    cache, renders = old_cache(tmp_path, [f_old], old_root, args())
    rekeyed, _, dropped = plan_poses([f_new], cache, old_root, new_root, True)
    assert len(rekeyed) == 1 and not dropped


def test_the_library_growing_upward_prepends_the_prefix(tmp_path):
    # cache built on one kit, run now covers the library around it
    lib = tmp_path / "STL"
    kit = lib / "Loot Studios"
    f = model(kit, "Kit/model.stl")
    cache, _ = old_cache(tmp_path, [f], kit, args(), absolute=False)
    rekeyed, _, dropped = plan_poses([f], cache, kit, lib, False)
    assert list(rekeyed) == [pose.file_identity(f, lib)]
    assert not dropped


# --- embeds and renders -----------------------------------------------------

def test_embeds_move_into_the_subdirectory_under_their_new_key(tmp_path):
    root = tmp_path / "STL"
    f = model(root)
    a = args()
    cache, _ = old_cache(tmp_path, [f], root, a)
    rekeyed, _, _ = plan_poses([f], cache, root, root, True)
    moves, already, missing, orphans = plan_embeds([f], cache, a, rekeyed, root, root, True)
    assert (already, missing, orphans) == (0, 0, []) and len(moves) == 1
    move_all(moves)
    token = pose.embed_cache_token(rekeyed[pose.file_identity(f, root)], a.up_axis)
    assert (cache / EMBEDS_SUBDIR / f"{cache_key(f, a, token, root)}.npy").exists()


def test_renders_move_under_the_cache_keeping_their_config(tmp_path):
    root = tmp_path / "STL"
    f = model(root)
    a = args()
    cache, renders = old_cache(tmp_path, [f], root, a)
    moves, already, orphans = plan_renders([f], renders, cache, a, root, root, True)
    assert len(moves) == 3 and not orphans and already == 0
    move_all(moves)
    out = cache / RENDERS_SUBDIR / CFG
    assert sorted(p.name for p in out.iterdir()) == \
        [f"{render_key(f, root)}_{t}.jpg" for t in ("pose", "view0", "view1")]


def test_renders_nothing_claims_are_left_where_they_are(tmp_path):
    # the interim hash-of-absolute names are what migrate_renders.py called
    # orphans; here anything unmatched is reported, never deleted
    root = tmp_path / "STL"
    f = model(root)
    a = args()
    cache, renders = old_cache(tmp_path, [f], root, a)
    stray = renders / CFG / "someone_elses_view0.jpg"
    Image.new("RGB", (8, 8)).save(stray)
    _, _, orphans = plan_renders([f], renders, cache, a, root, root, True)
    assert orphans == [stray] and stray.exists()


def test_a_second_render_pass_moves_nothing(tmp_path):
    root = tmp_path / "STL"
    f = model(root)
    a = args()
    cache, renders = old_cache(tmp_path, [f], root, a)
    moves, _, _ = plan_renders([f], renders, cache, a, root, root, True)
    move_all(moves)
    again, already, _ = plan_renders([f], renders, cache, a, root, root, True)
    assert not again and already == 0   # sources are gone; nothing left to claim


def test_no_old_renders_directory_is_not_an_error(tmp_path):
    root = tmp_path / "STL"
    f = model(root)
    cache, _ = old_cache(tmp_path, [f], root, args())
    assert plan_renders([f], None, cache, args(), root, root, True) == ([], 0, [])


def test_embeddings_nothing_claims_are_reported_not_removed(tmp_path):
    # a model deleted since the cache was written leaves its .npy behind; a
    # half-mounted collection looks identical, so this decides nothing
    root = tmp_path / "STL"
    f, gone = model(root), model(root, "Kit/deleted.stl")
    a = args()
    cache, _ = old_cache(tmp_path, [f, gone], root, a)
    gone.unlink()
    rekeyed, _, _ = plan_poses([f], cache, root, root, True)
    _, _, _, orphans = plan_embeds([f], cache, a, rekeyed, root, root, True)
    assert len(orphans) == 1 and orphans[0].exists()


def test_a_newer_resolution_is_not_rolled_back_by_an_old_entry(tmp_path):
    # both formats present: only possible if a new-code run happened before the
    # migration, and then the new-format entry is the later of the two
    root = tmp_path / "STL"
    f = model(root)
    cache, _ = old_cache(tmp_path, [f], root, args(), source="heuristic")
    entries = json.loads((cache / "pose-cache.json").read_text())
    entries[pose.file_identity(f, root)] = {
        "up": [0.0, 1.0, 0.0], "confidence": 0.9, "source": "vlm", "margin": 0.1,
        "v": pose.POSE_CACHE_VERSION, "front_view": 5}
    (cache / "pose-cache.json").write_text(json.dumps(entries))
    rekeyed, _, dropped = plan_poses([f], cache, root, root, True)
    assert rekeyed[pose.file_identity(f, root)]["source"] == "vlm"
    assert not dropped          # the superseded old entry is claimed, not dropped


def test_a_part_applied_rerun_does_not_call_the_leftover_unclaimed(tmp_path):
    root = tmp_path / "STL"
    f = model(root)
    a = args()
    cache, _ = old_cache(tmp_path, [f], root, a)
    rekeyed, _, _ = plan_poses([f], cache, root, root, True)
    moves, _, _, _ = plan_embeds([f], cache, a, rekeyed, root, root, True)
    src = moves[0][0]
    shutil.copy(src, moves[0][1].parent.mkdir(parents=True, exist_ok=True) or moves[0][1])
    _, already, _, orphans = plan_embeds([f], cache, a, rekeyed, root, root, True)
    assert already == 1 and orphans == []   # the source is spoken for
