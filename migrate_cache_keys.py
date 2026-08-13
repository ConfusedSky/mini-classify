"""Re-key and re-shape a cache written before the keys went relative.

Usage:
  python migrate_cache_keys.py [/path/to/stls] --cache-dir embed-cache2 [--apply]

Two things changed at once and this migrates both, because they touch the same
entries and walking the collection twice would be silly:

* Keys named absolute paths and full-nanosecond mtimes; they now name a path
  relative to the collection root and a whole-second mtime (identity.py).
* The .npy files sat in the cache root and the renders in a directory of their
  own; both now live under the cache, as embeds/ and renders/<camera config>/.

Without this every entry misses and the next run re-renders, re-embeds and
re-resolves the whole collection — hours, and real money once a pose entry is
VLM-sourced.

Order is forced: poses first, because an embedding's key contains the pose's
up-token, then embeds, then renders.

It also handles the case where a cache was built on one kit and the library
later grew upward around it. There the old keys are already relative, just to a
narrower root, so migrating is prepending the intervening directories.

Dry run by default; nothing moves without --apply. Safe to re-run — anything
already carrying its new name is left alone. Nothing outside the collection is
deleted: unmatched .npy files and renders are reported and left where they are.
The one exception is pose entries that match no file, which are dropped,
because load_pose_cache filters on version alone and would carry them forever.
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import identity
import pose
from classify_stls import (DEFAULT_ELEVATIONS, EMBEDS_SUBDIR, RENDERS_SUBDIR,
                           add_cache_args, apply_run_params, cache_key,
                           load_file_list, load_run_params, render_key)

# "<render key>_view3" / "<render key>_pose", split off the right-hand end
# because the stem itself may contain anything.
SUFFIX = re.compile(r"^(?P<key>.+)_(?P<tail>view\d+|pose)$")


# --- the key formats this migrates *from*, preserved here on purpose ---------
#
# A migration is the one place old formats have to keep working. These mirror
# pose.file_identity / classify_stls.cache_key / classify_stls.render_key as
# they were, so the two can be diffed rather than recalled.

def old_base(f, old_root, new_root, absolute):
    """The path half of an old key, as the cache would have spelled it.

    Reconstructed from where the file sits under the *current* root and then
    re-rooted at the old one, rather than read off f.resolve(): a library that
    has already moved has to be able to work out what its keys used to say.

    Only the path can be recovered that way. The mtime cannot, so migrating a
    library that has already crossed filesystems may still miss — migrate
    first, move second, and the question never arises."""
    rel = identity.rel_path(f, new_root)
    return str(Path(old_root) / rel) if absolute else identity.rel_path(f, old_root)


def old_identity(f, old_root, new_root, absolute):
    st = f.stat()
    stamp = st.st_mtime_ns if absolute else identity.mtime_key(st)
    return f"{old_base(f, old_root, new_root, absolute)}|{stamp}|{st.st_size}"


def old_cache_key(f, args, up_token, old_root, new_root, absolute):
    st = f.stat()
    elev = "" if args.elevations == DEFAULT_ELEVATIONS else \
        "|e:" + ",".join(f"{e:g}" for e in args.elevations)
    stamp = st.st_mtime_ns if absolute else identity.mtime_key(st)
    raw = (f"{old_base(f, old_root, new_root, absolute)}|{stamp}|{st.st_size}|{args.views}"
           f"|{args.render_size}|{up_token}|{args.model}|pv{elev}")
    return hashlib.sha1(raw.encode()).hexdigest()


def old_render_key(f, old_root, new_root, absolute):
    digest = hashlib.sha1(old_base(f, old_root, new_root, absolute).encode()).hexdigest()
    return f"{f.stem}_{digest[:6]}"


# --- planning ---------------------------------------------------------------

def plan_poses(files, cache_dir, old_root, new_root, absolute):
    """(rekeyed, kept, dropped) for pose-cache.json.

    `dropped` are entries no walked file claims — a model deleted or renamed
    since the cache was written. They are returned rather than discarded
    quietly so a VLM-sourced one can be called out before it goes."""
    path = Path(cache_dir) / "pose-cache.json"
    if not path.exists():
        return {}, {}, {}
    cache = json.loads(path.read_text())
    rekeyed, claimed = {}, set()
    for f in files:
        old = old_identity(f, old_root, new_root, absolute)
        new = pose.file_identity(f, new_root)
        # a file can be filed under both formats if a new-code run happened
        # before this migration. The new-format entry is then the later
        # resolution, and taking the old one would roll it back.
        entry = cache.get(new, cache.get(old))
        if entry is None:
            continue
        claimed.update(k for k in (old, new) if k in cache)
        rekeyed[new] = entry
    dropped = {k: v for k, v in cache.items() if k not in claimed}
    return rekeyed, cache, dropped


def plan_embeds(files, cache_dir, args, poses, old_root, new_root, absolute):
    """(moves, already, missing, orphans) for the .npy files.

    `orphans` are .npy left in the cache root that no walked model claims —
    embeddings for files deleted since. Reported rather than removed, for the
    same reason as the renders: this decides nothing about data it cannot
    identify, and a collection that is only half-mounted looks exactly like a
    collection that shrank."""
    cache_dir = Path(cache_dir)
    dst_dir = cache_dir / EMBEDS_SUBDIR
    moves, already, missing, claimed = [], 0, 0, set()
    for f in files:
        # the up-token comes from the entry, which migration does not change
        token = pose.embed_cache_token(poses.get(pose.file_identity(f, new_root)),
                                       args.up_axis)
        src = cache_dir / f"{old_cache_key(f, args, token, old_root, new_root, absolute)}.npy"
        dst = dst_dir / f"{cache_key(f, args, token, new_root)}.npy"
        if dst.exists():
            # a part-applied run leaves the source behind; it is spoken for,
            # not unclaimed, and mislabelling it reads as data nothing wants
            already += 1
            claimed.add(src)
        elif src.exists():
            moves.append((src, dst))
            claimed.add(src)
        else:
            missing += 1
    orphans = [p for p in sorted(cache_dir.glob("*.npy")) if p not in claimed]
    return moves, already, missing, orphans


def plan_renders(files, old_renders, cache_dir, args, old_root, new_root, absolute):
    """[(src, dst)] moves for saved renders, and the files nothing claims.

    Every camera-config directory is migrated, not just the current one: a
    render config is a directory name, and an old config's images are still
    that config's images under the new key."""
    remap = {old_render_key(f, old_root, new_root, absolute): render_key(f, new_root)
             for f in files}
    moves, already, orphans = [], 0, []
    if not old_renders or not Path(old_renders).is_dir():
        return moves, already, orphans
    for cfg in sorted(p for p in Path(old_renders).iterdir() if p.is_dir()):
        for p in sorted(cfg.iterdir()):
            if not p.is_file():
                continue
            m = SUFFIX.match(p.stem)
            new = remap.get(m.group("key")) if m else None
            if new is None:
                orphans.append(p)
                continue
            dst = Path(cache_dir) / RENDERS_SUBDIR / cfg.name / \
                f"{new}_{m.group('tail')}{p.suffix}"
            if dst.exists():
                already += 1
            else:
                moves.append((p, dst))
    return moves, already, orphans


def move_all(moves):
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def main():
    parser = argparse.ArgumentParser()
    add_cache_args(parser, "STL directory, where the library is now "
                           "(defaults to the last classify_stls.py run)")
    parser.add_argument("--old-renders", default=None,
                        help="renders directory from before the move (defaults to "
                             "the renders_dir the cache recorded)")
    parser.add_argument("--apply", action="store_true",
                        help="actually re-key and move; without it this only reports")
    args = apply_run_params(parser)
    if not args.input:
        sys.exit("no input given, and no directory recorded by classify_stls.py — "
                 "pass the STL directory explicitly")

    params = load_run_params(args.cache_dir)
    new_root = identity.collection_root(Path(args.input))
    anchored = params.get("collection_root")
    old_root = Path(anchored) if anchored else Path(params.get("input") or new_root)
    absolute = anchored is None       # no anchor recorded = keys name absolute paths
    if anchored and Path(anchored) == new_root:
        print(f"cache is already anchored at {new_root} — nothing to re-key")
        return

    print(f"cache      {args.cache_dir}")
    print(f"old keys   {'absolute paths' if absolute else f'relative to {old_root}'}")
    print(f"new keys   relative to {new_root}")

    files = load_file_list(Path(args.input), args.cache_dir, args.rescan)
    print(f"collection {len(files)} models\n")

    rekeyed, old_cache, dropped = plan_poses(
        files, args.cache_dir, old_root, new_root, absolute)
    paid = [v for v in dropped.values() if v.get("source") == "vlm"]
    print(f"poses      {len(rekeyed)} re-keyed of {len(old_cache)}, "
          f"{len(dropped)} match no file and would be dropped")
    if paid:
        print(f"           WARNING: {len(paid)} of those are VLM-sourced — a paid "
              f"arbiter call each. Check the collection is fully mounted first.")
    if dropped and not args.rescan:
        # the walk decides what "matches no file" means, so a list from before
        # the last download drops entries for models that are sitting right there
        print(f"           the file list came from cache — rerun with --rescan "
              f"before --apply, or those {len(dropped)} may just be unwalked")

    moves_e, already_e, missing_e, orphans_e = plan_embeds(
        files, args.cache_dir, args, rekeyed, old_root, new_root, absolute)
    print(f"embeds     {len(moves_e)} to move into {EMBEDS_SUBDIR}/"
          + (f", {already_e} already there" if already_e else "")
          + (f", {missing_e} models have no cached embedding" if missing_e else "")
          + (f", {len(orphans_e)} claimed by no model (left alone)" if orphans_e else ""))

    old_renders = args.old_renders or params.get("renders_dir")
    moves_r, already_r, orphans_r = plan_renders(
        files, old_renders, args.cache_dir, args, old_root, new_root, absolute)
    print(f"renders    {len(moves_r)} to move into {RENDERS_SUBDIR}/ from {old_renders}"
          + (f", {already_r} already there" if already_r else "")
          + (f", {len(orphans_r)} claimed by no model (left alone)" if orphans_r else ""))
    for src, dst in moves_r[:3]:
        print(f"             {src.name}  ->  {RENDERS_SUBDIR}/{dst.parent.name}/{dst.name}")

    if not args.apply:
        print("\ndry run — nothing changed; pass --apply to do it")
        return

    # poses first: an embedding's key contains the pose's up-token
    pose.save_pose_cache(args.cache_dir, rekeyed)
    move_all(moves_e)
    move_all(moves_r)
    params["collection_root"] = str(new_root)
    params.pop("renders_dir", None)      # the location is derived from the cache now
    (Path(args.cache_dir) / "run-params.json").write_text(json.dumps(params, indent=2))

    print(f"\nre-keyed {len(rekeyed)} poses, moved {len(moves_e)} embeddings and "
          f"{len(moves_r)} renders")
    if dropped:
        print(f"dropped {len(dropped)} pose entries that matched no file")
    stale = [(len(orphans_e), f"{EMBEDS_SUBDIR} .npy in the cache root"),
             (len(orphans_r), f"renders under {old_renders}")]
    for n, what in stale:
        if n:
            print(f"{n} {what} were claimed by no model and are still there — "
                  f"check them, then delete")


if __name__ == "__main__":
    main()
