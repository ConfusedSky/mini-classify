import argparse
import json
import os
import shutil

import numpy as np
import pytest
from PIL import Image

from src import identity
from src import pose
from classify_stls import (CACHE_VERSION, EMBEDS_SUBDIR, RENDERS_SUBDIR,
                           cache_key, cache_key_from_identity, cache_version,
                           render_key, require_cache_version)
from migrate_cache_keys import (old_cache_key, old_embed_cache_token,
                                old_identity, old_render_key, plan_embeds,
                                plan_poses, plan_renders, plan_token_moves,
                                move_all)

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
        # a root-era cache necessarily used the old, eliding token
        token = old_embed_cache_token(entry, a.up_axis)
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
    # a render this collection does not account for — an older key scheme, or
    # a model since deleted. Reported, never deleted: a half-mounted collection
    # looks exactly like one that shrank.
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


# --- the token migration (cache_version 0 -> 1) -----------------------------

def token_era_cache(tmp_path, files, root, a, *, source="ensemble"):
    """A cache in the post-root, pre-token shape: relative keys, embeds/ in
    place, up-token still the old elision."""
    cache = tmp_path / "cache"
    (cache / EMBEDS_SUBDIR).mkdir(parents=True, exist_ok=True)
    entries = {}
    for f in files:
        entry = {"up": [0.0, 0.0, 1.0], "confidence": 0.1, "source": source,
                 "margin": 0.9, "v": pose.POSE_CACHE_VERSION}
        entries[pose.file_identity(f, root)] = entry
        token = old_embed_cache_token(entry, a.up_axis)
        np.save(cache / EMBEDS_SUBDIR / f"{cache_key(f, a, token, root)}.npy",
                np.zeros((16, 4), dtype=np.float32))
    (cache / "pose-cache.json").write_text(json.dumps(entries))
    return cache


def test_token_migration_rekeys_within_embeds(tmp_path):
    root = tmp_path / "STL"
    f = model(root)
    a = args()
    cache = token_era_cache(tmp_path, [f], root, a)
    poses = json.loads((cache / "pose-cache.json").read_text())
    moves, already, missing, superseded, unclaimed = \
        plan_token_moves([f], cache, a, poses, root)
    assert (already, missing, superseded, unclaimed) == (0, 0, [], [])
    assert len(moves) == 1
    move_all(moves)
    entry = poses[pose.file_identity(f, root)]
    token = pose.embed_cache_token(entry, a.up_axis)
    assert (cache / EMBEDS_SUBDIR / f"{cache_key(f, a, token, root)}.npy").exists()
    # a re-run has nothing left to move
    moves, already, _, _, _ = plan_token_moves([f], cache, a, poses, root)
    assert moves == [] and already == 1


def test_elided_token_also_moves(tmp_path):
    # the elision case: a geometry-agreed pose keyed as the literal "auto"
    root = tmp_path / "STL"
    f = model(root)
    a = args()
    cache = token_era_cache(tmp_path, [f], root, a, source="heuristic")
    old = cache / EMBEDS_SUBDIR / f"{cache_key(f, a, 'auto', root)}.npy"
    assert old.exists()          # the fixture really used the elided token
    poses = json.loads((cache / "pose-cache.json").read_text())
    moves, *_ = plan_token_moves([f], cache, a, poses, root)
    assert moves and moves[0][0] == old
    assert moves[0][1].name == f"{cache_key(f, a, '0,0,1', root)}.npy"


# --- cache-meta -------------------------------------------------------------

def test_unstamped_populated_cache_is_refused(tmp_path):
    # a moved key scheme does not error on its own — every lookup just misses
    # and the run silently re-embeds the collection; the stamp turns that into
    # one line naming the migration
    (tmp_path / "pose-cache.json").write_text("{}")
    with pytest.raises(SystemExit, match="migrate_cache_keys"):
        require_cache_version(tmp_path)


def test_empty_cache_is_stamped_current(tmp_path):
    d = tmp_path / "fresh"
    require_cache_version(d)
    assert cache_version(d) == CACHE_VERSION
    require_cache_version(d)     # and idempotent thereafter


def test_token_migration_reaches_entries_outside_the_walk(tmp_path):
    # S1: driven from the pose cache with no stat() — a half-mounted
    # collection cannot leave embeddings behind
    root = tmp_path / "STL"
    f = model(root)
    a = args()
    cache = token_era_cache(tmp_path, [f], root, a)
    poses = json.loads((cache / "pose-cache.json").read_text())
    f.unlink()                      # the model is gone from the walk entirely
    moves, _, _, _, unclaimed = plan_token_moves([], cache, a, poses, root)
    assert len(moves) == 1 and unclaimed == []


def test_superseded_source_is_reported_not_clobbered(tmp_path):
    # S4/T1: a destination occupied by a newer write names the superseded
    # source instead of silently filing it under "already" — the cross-axis
    # forced/geometry duplicate, by contrast, surfaces in `unclaimed`
    root = tmp_path / "STL"
    f = model(root)
    a = args()
    cache = token_era_cache(tmp_path, [f], root, a, source="heuristic")
    ident = pose.file_identity(f, root)
    poses = json.loads((cache / "pose-cache.json").read_text())
    new_key = cache_key_from_identity(
        ident, a, pose.embed_cache_token(poses[ident], a.up_axis))
    np.save(cache / EMBEDS_SUBDIR / f"{new_key}.npy",
            np.ones((16, 4), dtype=np.float32))
    moves, already, _, superseded, _ = plan_token_moves([f], cache, a, poses, root)
    assert moves == [] and already == 0
    assert len(superseded) == 1 and superseded[0].exists()


def test_pre_layout_cache_is_refused_not_stamped(tmp_path):
    # S2: root-level .npy with no pose-cache.json or embeds/ is still a
    # populated cache — stamping it current would shut the guard forever
    np.save(tmp_path / "deadbeef.npy", np.zeros(3))
    with pytest.raises(SystemExit, match="migrate_cache_keys"):
        require_cache_version(tmp_path)
    assert cache_version(tmp_path) == 0      # and it was not stamped


def test_cache_key_first_fields_are_file_identity(tmp_path):
    # the coupling §P3.1 verified and plan_token_moves now leans on:
    # file_identity is byte-identical to cache_key's first three fields
    root = tmp_path / "STL"
    f = model(root)
    a = args()
    assert cache_key(f, a, "0,0,1", root) == \
        cache_key_from_identity(pose.file_identity(f, root), a, "0,0,1")
