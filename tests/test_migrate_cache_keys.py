"""migrate_cache_keys.py, reduced to the one migration a current cache can
still need: the collection root moved (docs/cache-rebuild.md §5).

Deleted with the 2026-08-31 rebuild, along with the key formats they exercised:
the absolute-path/nanosecond-mtime root migration, the flat pre-`embeds/`
layout, and the up-token migration (`cache_version` 0 -> 1) with its
`old_embed_cache_token` elision. Their tests went with them; what is left is
built through the live writers, which is also what makes
`test_an_older_recipes_embedding_is_never_promoted_into_a_current_name` mean
something.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pytest
from PIL import Image

from src import pose
from src.cachedir import (CACHE_VERSION, EMBEDS_SUBDIR, RENDERS_SUBDIR,
                          cache_key, cache_version, render_subdir,
                          require_cache_version, save_run_params)
from src.identity import cache_key_from_identity, render_key
from migrate_cache_keys import (main, move_all, plan_embeds, plan_poses,
                                plan_renders)


def args(**kw):
    base = dict(render_size=384, views=8, elevations=[20.0, -20.0],
                model="siglip", up_axis="auto")
    base.update(kw)
    return argparse.Namespace(**base)


CFG = render_subdir(args())


def model(root, name="Kit/model.stl", mtime=1_700_000_000):
    f = root / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"solid")
    os.utime(f, ns=(mtime * 10**9 + 642_647_652,) * 2)
    return f


def anchored_cache(tmp_path, files, root, a, *, source="siglip"):
    """A cache in the current shape, keyed relative to `root`.

    Written through the production key functions rather than by hand, so the
    migration is read against what a real run leaves behind (CLAUDE.md: build
    cache fixtures through the production writers)."""
    cache = tmp_path / "cache"
    (cache / EMBEDS_SUBDIR).mkdir(parents=True, exist_ok=True)
    entries = {}
    for f in files:
        entry = {"up": [0.0, 0.0, 1.0], "confidence": 0.1, "source": source,
                 "margin": 0.9, "v": pose.POSE_CACHE_VERSION,
                 "front_view": {CFG: 3}}
        entries[pose.file_identity(f, root)] = entry
        token = pose.embed_cache_token(entry, a.up_axis)
        np.save(cache / EMBEDS_SUBDIR / f"{cache_key(f, a, token, root)}.npy",
                np.zeros((16, 4), dtype=np.float32))
        d = cache / RENDERS_SUBDIR / CFG
        d.mkdir(parents=True, exist_ok=True)
        for tail in ("view0", "view1", "pose"):
            Image.new("RGB", (8, 8)).save(d / f"{render_key(f, root)}_{tail}.jpg")
    (cache / "pose-cache.json").write_text(json.dumps(entries))
    return cache, cache / RENDERS_SUBDIR


def run_params(cache, root, a=None):
    """run-params.json as a classify run leaves it.

    Through `save_run_params` rather than a hand-written dict, so `main` reads
    the manifest a real run records — including the `collection_root` key the
    anchoring turns on (CLAUDE.md: production writers)."""
    a = a or args()
    save_run_params(argparse.Namespace(
        cache_dir=str(cache), input=str(root), collection_root=str(root),
        **vars(a)))


def invoke(cache, inp, *extra):
    """`main()` as the command line reaches it — argv, not a call signature,
    because the root derivation these tests pin sits behind `apply_run_params`
    and the manifest it merges."""
    return ["migrate_cache_keys.py", str(inp), "--cache-dir", str(cache),
            *extra]


def grown_library(tmp_path):
    """The root move that actually happens: a cache built on one kit, run now
    against the library that grew around it. The old keys are relative, just to
    a narrower root, so migrating is prepending the intervening directories."""
    lib = tmp_path / "STL"
    kit = lib / "Loot Studios"
    return lib, kit, model(kit, "Kit/model.stl")


# --- poses ------------------------------------------------------------------

def test_poses_are_re_keyed_onto_the_wider_root(tmp_path):
    lib, kit, f = grown_library(tmp_path)
    cache, _ = anchored_cache(tmp_path, [f], kit, args())
    rekeyed, old, dropped, _ = plan_poses([f], cache, kit, lib)
    assert list(rekeyed) == [pose.file_identity(f, lib)]
    assert list(old) == [pose.file_identity(f, kit)]     # what it was keyed as
    assert not dropped
    assert rekeyed[pose.file_identity(f, lib)]["front_view"] == {CFG: 3}


def test_entries_matching_no_file_are_dropped(tmp_path):
    # load_pose_cache filters on version and shape, never on whether the file
    # still exists, so these would ride along forever
    lib, kit, f = grown_library(tmp_path)
    gone = model(kit, "Kit/deleted.stl")
    cache, _ = anchored_cache(tmp_path, [f, gone], kit, args())
    gone.unlink()
    rekeyed, _, dropped, _ = plan_poses([f], cache, kit, lib)
    assert len(rekeyed) == 1 and len(dropped) == 1


def test_a_re_run_re_keys_nothing_and_drops_nothing(tmp_path):
    lib, kit, f = grown_library(tmp_path)
    cache, _ = anchored_cache(tmp_path, [f], kit, args())
    rekeyed, _, _, _ = plan_poses([f], cache, kit, lib)
    (cache / "pose-cache.json").write_text(json.dumps(rekeyed))
    again, _, dropped, _ = plan_poses([f], cache, kit, lib)
    assert again == rekeyed and not dropped


def test_a_move_between_drives_needs_no_re_keying_at_all(tmp_path):
    # the case the relative keys already solved: same layout, different mount.
    # Every key is unchanged, and the migration must not lose the entry anyway.
    a_root, b_root = tmp_path / "driveA", tmp_path / "driveB"
    f_old, f_new = model(a_root), model(b_root)
    cache, _ = anchored_cache(tmp_path, [f_old], a_root, args())
    rekeyed, _, dropped, _ = plan_poses([f_new], cache, a_root, b_root)
    assert list(rekeyed) == [pose.file_identity(f_new, b_root)]
    assert pose.file_identity(f_new, b_root) == pose.file_identity(f_old, a_root)
    assert not dropped


def test_a_newer_resolution_is_not_rolled_back_by_an_old_entry(tmp_path):
    # both roots present: only possible if a new-code run happened before the
    # migration, and then the new-root entry is the later of the two
    lib, kit, f = grown_library(tmp_path)
    cache, _ = anchored_cache(tmp_path, [f], kit, args(), source="geometry")
    entries = json.loads((cache / "pose-cache.json").read_text())
    entries[pose.file_identity(f, lib)] = {
        "up": [0.0, 1.0, 0.0], "confidence": 0.9, "source": "vlm", "margin": 0.1,
        "v": pose.POSE_CACHE_VERSION, "front_view": {CFG: 5}}
    (cache / "pose-cache.json").write_text(json.dumps(entries))
    rekeyed, _, dropped, _ = plan_poses([f], cache, kit, lib)
    assert rekeyed[pose.file_identity(f, lib)]["source"] == "vlm"
    assert not dropped          # the superseded old entry is claimed, not dropped


def test_entries_the_loader_will_drop_are_counted_not_repaired(tmp_path):
    """The migration re-keys; it does not decide what is readable.

    An entry the *next* run's `pose.load_pose_cache` will drop — an older `v`,
    or one of the shapes `pose._readable` refuses — is still claimed by a
    walked file, so it is re-keyed and copied unchanged like any other. That
    is deliberate: re-keying is a rename, and a migration that quietly deleted
    entries would be deciding something nobody asked it to. What changes is
    that it is now *counted*, so a run that re-keys N poses and re-resolves
    some of them anyway says so before the re-render bill arrives rather than
    after (adversarial review pass 3, 2026-08-31).

    `_readable` is imported, not mirrored: two copies of that rule would drift
    and this line would start lying about which entries survive."""
    lib, kit, f = grown_library(tmp_path)
    doomed = model(kit, "Kit/doomed.stl")
    cache, _ = anchored_cache(tmp_path, [f, doomed], kit, args())

    entries = json.loads((cache / "pose-cache.json").read_text())
    key = pose.file_identity(doomed, kit)
    entries[key].pop("margin")            # present-and-null is the rule
    (cache / "pose-cache.json").write_text(json.dumps(entries))
    before = dict(entries[key])

    rekeyed, _, dropped, unreadable = plan_poses([f, doomed], cache, kit, lib)
    new_key = pose.file_identity(doomed, lib)
    assert not dropped and len(rekeyed) == 2
    assert list(unreadable) == [new_key]
    assert rekeyed[new_key] == before                     # copied unchanged
    # and the count really is the loader's answer, not a second opinion
    pose.save_pose_cache(cache, rekeyed)
    assert set(pose.load_pose_cache(cache)) == set(rekeyed) - set(unreadable)


# --- embeds -----------------------------------------------------------------

def test_embeds_are_re_keyed_under_the_new_root(tmp_path):
    lib, kit, f = grown_library(tmp_path)
    a = args()
    cache, _ = anchored_cache(tmp_path, [f], kit, a)
    rekeyed, _, _, _ = plan_poses([f], cache, kit, lib)
    moves, already, missing, unclaimed = plan_embeds([f], cache, a, rekeyed, kit, lib)
    assert (already, missing, unclaimed) == (0, 0, []) and len(moves) == 1
    move_all(moves)
    token = pose.embed_cache_token(rekeyed[pose.file_identity(f, lib)], a.up_axis)
    assert (cache / EMBEDS_SUBDIR / f"{cache_key(f, a, token, lib)}.npy").exists()


def test_embeddings_nothing_claims_are_reported_not_removed(tmp_path):
    # a model deleted since the cache was written leaves its .npy behind; a
    # half-mounted collection looks identical, so this decides nothing
    lib, kit, f = grown_library(tmp_path)
    gone = model(kit, "Kit/deleted.stl")
    a = args()
    cache, _ = anchored_cache(tmp_path, [f, gone], kit, a)
    gone.unlink()
    rekeyed, _, _, _ = plan_poses([f], cache, kit, lib)
    _, _, _, unclaimed = plan_embeds([f], cache, a, rekeyed, kit, lib)
    assert len(unclaimed) == 1 and unclaimed[0].exists()


def test_an_older_recipes_embedding_is_never_promoted_into_a_current_name(tmp_path):
    """The hazard the framing change's review named, and the property that
    closes it (migrate_cache_keys module docstring).

    A migration writes destinations with the *live* `cache_key`, so anything it
    picks up is promoted into the current recipe's namespace. The deleted
    `old_cache_key` computed sources in the pre-`ev` format —

        sha1(rel|mtime|size|views|render_size|up_token|model|pv[|e:...])

    with no `|evN` and no `|rN` — so it would have found an embedding of a
    framing nothing renders any more and filed it under a current name, where
    every later run reads it as current pixels. Computing *both* sides with the
    live `cache_key_from_identity` is what makes that impossible: a name
    carrying a version token this code no longer writes matches neither side.

    This is that property as a test rather than as a warning, because a warning
    cannot fail."""
    lib, kit, f = grown_library(tmp_path)
    a = args()
    cache, _ = anchored_cache(tmp_path, [f], kit, a)
    rekeyed, _, _, _ = plan_poses([f], cache, kit, lib)
    entry = rekeyed[pose.file_identity(f, lib)]
    token = pose.embed_cache_token(entry, a.up_axis)

    # an ev1-era .npy for the same model under the same (old) root
    ident = pose.file_identity(f, kit)
    elev = "|e:" + ",".join(f"{e:g}" for e in a.elevations)
    raw = f"{ident}|{a.views}|{a.render_size}|{token}|{a.model}|pv{elev}"
    stale = cache / EMBEDS_SUBDIR / f"{hashlib.sha1(raw.encode()).hexdigest()}.npy"
    np.save(stale, np.ones((16, 4), dtype=np.float32))
    before = stale.read_bytes()
    assert stale.name != f"{cache_key(f, a, token, kit)}.npy"   # really a different name

    moves, already, missing, unclaimed = plan_embeds([f], cache, a, rekeyed, kit, lib)

    # the live-key embedding moves; the ev1-era one is named as unclaimed
    assert len(moves) == 1 and moves[0][0] != stale
    assert unclaimed == [stale]
    move_all(moves)
    assert stale.exists() and stale.read_bytes() == before
    assert (cache / EMBEDS_SUBDIR / f"{cache_key(f, a, token, lib)}.npy").exists()


def test_a_part_applied_rerun_does_not_call_the_leftover_unclaimed(tmp_path):
    lib, kit, f = grown_library(tmp_path)
    a = args()
    cache, _ = anchored_cache(tmp_path, [f], kit, a)
    rekeyed, _, _, _ = plan_poses([f], cache, kit, lib)
    moves, _, _, _ = plan_embeds([f], cache, a, rekeyed, kit, lib)
    src, dst = moves[0]
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    _, already, _, unclaimed = plan_embeds([f], cache, a, rekeyed, kit, lib)
    assert already == 1 and unclaimed == []   # the source is spoken for


# --- renders ----------------------------------------------------------------

def test_renders_are_re_keyed_keeping_their_config(tmp_path):
    lib, kit, f = grown_library(tmp_path)
    cache, renders = anchored_cache(tmp_path, [f], kit, args())
    moves, already, unclaimed = plan_renders([f], renders, cache, kit, lib)
    assert len(moves) == 3 and not unclaimed and already == 0
    move_all(moves)
    out = cache / RENDERS_SUBDIR / CFG
    assert sorted(p.name for p in out.iterdir()) == \
        [f"{render_key(f, lib)}_{t}.jpg" for t in ("pose", "view0", "view1")]


def test_renders_nothing_claims_are_left_where_they_are(tmp_path):
    # a render this collection does not account for — an older key scheme, or
    # a model since deleted. Reported, never deleted: a half-mounted collection
    # looks exactly like one that shrank.
    lib, kit, f = grown_library(tmp_path)
    cache, renders = anchored_cache(tmp_path, [f], kit, args())
    stray = renders / CFG / "someone_elses_view0.jpg"
    Image.new("RGB", (8, 8)).save(stray)
    _, _, unclaimed = plan_renders([f], renders, cache, kit, lib)
    assert unclaimed == [stray] and stray.exists()


def test_a_second_render_pass_moves_nothing(tmp_path):
    lib, kit, f = grown_library(tmp_path)
    cache, renders = anchored_cache(tmp_path, [f], kit, args())
    moves, _, _ = plan_renders([f], renders, cache, kit, lib)
    move_all(moves)
    again, already, _ = plan_renders([f], renders, cache, kit, lib)
    assert not again and already == 3   # every file already carries its new name


def test_no_renders_directory_is_not_an_error(tmp_path):
    lib, kit, f = grown_library(tmp_path)
    cache, _ = anchored_cache(tmp_path, [f], kit, args())
    assert plan_renders([f], None, cache, kit, lib) == ([], 0, [])


# --- cache-meta -------------------------------------------------------------

def test_an_unstamped_cache_is_stamped_not_refused(tmp_path):
    """§4: version 0 used to mean "keyed under the up-token elision", and
    `require_cache_version` refused it with a line naming this tool. Nothing
    writes that scheme any more and the 2026-08-31 rebuild regenerated the last
    cache holding it, so 0 now means only "not stamped yet".

    Both shapes that used to be refused are here: a populated cache, and the
    pre-layout root-level `.npy` that S2 added to the definition of populated
    so a genuinely unmigrated cache could not be stamped current."""
    (tmp_path / "pose-cache.json").write_text("{}")
    np.save(tmp_path / "deadbeef.npy", np.zeros(3))
    assert cache_version(tmp_path) == 0
    require_cache_version(tmp_path)
    assert cache_version(tmp_path) == CACHE_VERSION


def test_empty_cache_is_stamped_current(tmp_path):
    d = tmp_path / "fresh"
    require_cache_version(d)
    assert cache_version(d) == CACHE_VERSION
    require_cache_version(d)     # and idempotent thereafter


# --- main: the root it migrates to, and the write that lands ----------------

def test_a_subdir_scoped_run_is_refused_before_anything_is_written(tmp_path, monkeypatch):
    """`identity.resolve_root`'s anchoring, which `main` used to go around.

    A run scoped to one kit *inside* the recorded collection root has nothing
    to migrate — the recorded root still anchors those keys, which is exactly
    why `resolve_root` keeps it and answers "subdir". Deriving the new root
    from the input alone answered the kit instead: every key recomputed
    relative to it, and on --apply every pose outside the kit dropped, because
    `plan_poses` claims only what the walk sees and the walk sees one kit. The
    second model here is that population — it must still be in the file
    afterwards."""
    lib, kit, f = grown_library(tmp_path)
    outside = model(lib, "Other Kit/model.stl")
    cache, _ = anchored_cache(tmp_path, [f, outside], lib, args())
    run_params(cache, lib)
    before = (cache / "pose-cache.json").read_bytes()

    monkeypatch.setattr(sys, "argv", invoke(cache, kit, "--rescan", "--apply"))
    with pytest.raises(SystemExit) as e:
        main()
    assert str(lib) in str(e.value)                  # names the recorded root
    assert (cache / "pose-cache.json").read_bytes() == before
    assert len(json.loads(before)) == 2


def test_a_torn_pose_cache_write_leaves_the_previous_one_readable(tmp_path, monkeypatch, capsys):
    """--apply rewrites the most expensive file in the cache, and re-running
    is this tool's entire recovery story (module docstring) — so a write that
    dies half-way has to leave something to re-run against.

    It used to go through `pose.save_pose_cache`, whose bare `write_text`
    truncates in place: ENOSPC or a Ctrl-C mid-write left a torn
    pose-cache.json and the re-run had no cache to read, on the one artifact
    whose loss costs money. Only the pose write is failed here, not every
    `os.replace` — `load_file_list` publishes its walk cache through the same
    idiom first, and a blanket failure would pass without the migration ever
    reaching the write under test."""
    lib, kit, f = grown_library(tmp_path)
    cache, _ = anchored_cache(tmp_path, [f], kit, args())
    run_params(cache, kit)
    before = (cache / "pose-cache.json").read_bytes()

    real_replace = os.replace

    def full_disk(src, dst):
        if os.path.basename(dst) == "pose-cache.json":
            raise OSError(28, "No space left on device")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", full_disk)
    monkeypatch.setattr(sys, "argv", invoke(cache, lib, "--rescan", "--apply"))
    with pytest.raises(OSError):
        main()

    assert "1 re-keyed of 1" in capsys.readouterr().out   # not a vacuous plan
    assert (cache / "pose-cache.json").read_bytes() == before
    assert not list(cache.glob("*.tmp"))                  # nothing stranded


def test_cache_key_first_fields_are_file_identity(tmp_path):
    # the coupling §P3.1 verified and plan_embeds leans on: file_identity is
    # byte-identical to cache_key's first three fields, which is what lets the
    # migration name an embedding from a pose entry alone
    root = tmp_path / "STL"
    f = model(root)
    a = args()
    assert cache_key(f, a, "0,0,1", root) == \
        cache_key_from_identity(pose.file_identity(f, root), a, "0,0,1")
