"""Re-key a cache whose collection root moved.

Usage:
  python migrate_cache_keys.py [/path/to/stls] --cache-dir embed-cache2 [--apply]

Every key here is taken relative to the collection root (identity.py), so a
library that moves between drives keeps its cache for free. What still needs
migrating is the case where the *root itself* changes: a cache built on one kit
and later run against the whole library, or a recorded root that no longer
describes where the collection lives. There the old keys are already relative,
just to a different root, so migrating is recomputing them — pose entries,
`.npy` under `embeds/`, and the render filenames, which hash the relative path
too.

Order is forced: poses first, because an embedding's key contains the pose's
up-token, then embeds, then renders.

Without this every entry misses and the next run re-renders, re-embeds and
re-resolves the whole collection — hours, and real money once a pose entry is
VLM-sourced.

**Old-recipe files are left alone, by construction.** The framing change's
review (2026-08-31) named the hazard: a migration writes destination names with
the *live* `cache_key`, so anything it picks up is promoted into the current
recipe's namespace, and an embedding of pixels nothing renders any more would
start being read as if it were current. What closes that is computing *both*
sides with the same live `cache_key_from_identity` — only the root differs. A
`.npy` written under an older `EMBED_CACHE_VERSION` or `RECIPE_VERSION` carries
a token this code no longer produces, so it matches neither name, is never
moved, and is counted in the unclaimed line at the end. Nothing outside the
collection is deleted either: unclaimed `.npy` and unclaimed renders are
reported and left where they are. The one exception is pose entries that match
no file, which are dropped, because `load_pose_cache` filters on version alone
and would carry them forever.

`cache-meta.json` is stamped **last**, only after every move succeeded, so an
interrupted run stays re-runnable.

Dry run by default; nothing moves without --apply. Safe to re-run — anything
already carrying its new name is left alone.

Note this cuts the other way too: **run the migration before rebuilding, not
after.** Anything it could still rescue is cheaper to migrate than to
re-render.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from src import identity
from src import pose
from src.cachedir import (CACHE_VERSION, EMBEDS_SUBDIR, RENDERS_SUBDIR,
                          add_cache_args, apply_run_params, load_file_list,
                          load_run_params, stamp_cache_version)
from src.identity import cache_key_from_identity, render_key

# "<render key>_view3" / "<render key>_pose", split off the right-hand end
# because the stem itself may contain anything.
SUFFIX = re.compile(r"^(?P<key>.+)_(?P<tail>view\d+|pose)$")


# --- planning ---------------------------------------------------------------

def plan_poses(files, cache_dir, old_root, new_root):
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
        old = pose.file_identity(f, old_root)
        new = pose.file_identity(f, new_root)
        # a file can be filed under both roots if a new-code run happened
        # before this migration. The new-root entry is then the later
        # resolution, and taking the old one would roll it back.
        entry = cache.get(new, cache.get(old))
        if entry is None:
            continue
        claimed.update(k for k in (old, new) if k in cache)
        rekeyed[new] = entry
    dropped = {k: v for k, v in cache.items() if k not in claimed}
    return rekeyed, cache, dropped


def plan_embeds(files, cache_dir, args, poses, old_root, new_root):
    """(moves, already, missing, unclaimed) for the .npy under embeds/.

    Both names come from the live `cache_key_from_identity` with only the root
    changed, which is what keeps an older recipe's embeddings out of current
    names (module docstring). `unclaimed` is where they surface: .npy that
    neither computed name accounts for. Reported, never touched — this decides
    nothing about data it cannot identify, and a half-mounted collection looks
    exactly like a collection that shrank."""
    dst_dir = Path(cache_dir) / EMBEDS_SUBDIR
    moves, already, missing, claimed = [], 0, 0, set()
    for f in files:
        old_ident = pose.file_identity(f, old_root)
        new_ident = pose.file_identity(f, new_root)
        # one token for both names: it comes from the pose's up, which a change
        # of root does not move
        token = pose.embed_cache_token(poses.get(new_ident), args.up_axis)
        src = dst_dir / f"{cache_key_from_identity(old_ident, args, token)}.npy"
        dst = dst_dir / f"{cache_key_from_identity(new_ident, args, token)}.npy"
        claimed.update((src, dst))
        if dst.exists():
            # a part-applied run leaves the source behind; it is spoken for,
            # not unclaimed, and mislabelling it reads as data nothing wants
            already += 1
        elif src.exists():
            moves.append((src, dst))
        else:
            missing += 1
    unclaimed = [p for p in sorted(dst_dir.glob("*.npy"))
                 if p not in claimed] if dst_dir.is_dir() else []
    return moves, already, missing, unclaimed


def plan_renders(files, renders, cache_dir, old_root, new_root):
    """[(src, dst)] moves for saved renders, and the files nothing claims.

    Render filenames hash the path relative to the root, so a root move
    re-keys them in place. Every camera-config directory is migrated, not just
    the current one: a render config is a directory name, and an old config's
    images are still that config's images under the new key.

    The remap carries each new key to itself as well as the old key to it,
    because the move is in place: without that a re-run reads its own output as
    claimed by nobody and reports every migrated render as a stray."""
    remap = {}
    for f in files:
        new = render_key(f, new_root)
        remap[render_key(f, old_root)] = new
        remap[new] = new
    moves, already, unclaimed = [], 0, []
    if not renders or not Path(renders).is_dir():
        return moves, already, unclaimed
    for cfg in sorted(p for p in Path(renders).iterdir() if p.is_dir()):
        for p in sorted(cfg.iterdir()):
            if not p.is_file():
                continue
            m = SUFFIX.match(p.stem)
            new = remap.get(m.group("key")) if m else None
            if new is None:
                unclaimed.append(p)
                continue
            dst = Path(cache_dir) / RENDERS_SUBDIR / cfg.name / \
                f"{new}_{m.group('tail')}{p.suffix}"
            if dst.exists():
                already += 1
            else:
                moves.append((p, dst))
    return moves, already, unclaimed


def move_all(moves):
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def main():
    parser = argparse.ArgumentParser()
    add_cache_args(parser, "STL directory, where the library is now "
                           "(defaults to the last classify_stls.py run)")
    parser.add_argument("--renders", default=None,
                        help="renders directory to re-key (defaults to the "
                             "cache's own renders/)")
    parser.add_argument("--apply", action="store_true",
                        help="actually re-key and move; without it this only reports")
    args = apply_run_params(parser)
    if not args.input:
        sys.exit("no input given, and no directory recorded by classify_stls.py — "
                 "pass the STL directory explicitly")

    params = load_run_params(args.cache_dir)
    new_root = identity.collection_root(Path(args.input))
    anchored = params.get("collection_root")
    old_root = Path(anchored) if anchored else new_root

    print(f"cache      {args.cache_dir}")
    if old_root == new_root:
        print(f"anchored at {new_root} — nothing to migrate")
        return

    print(f"old keys   relative to {old_root}")
    print(f"new keys   relative to {new_root}")

    files = load_file_list(Path(args.input), args.cache_dir, args.rescan)
    print(f"collection {len(files)} models\n")

    rekeyed, old_cache, dropped = plan_poses(
        files, args.cache_dir, old_root, new_root)
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

    moves_e, already_e, missing_e, unclaimed_e = plan_embeds(
        files, args.cache_dir, args, rekeyed, old_root, new_root)
    print(f"embeds     {len(moves_e)} to re-key under {EMBEDS_SUBDIR}/"
          + (f", {already_e} already at their new key" if already_e else "")
          + (f", {missing_e} models have no cached embedding" if missing_e else "")
          + (f", {len(unclaimed_e)} claimed by no model (left alone)" if unclaimed_e else ""))
    if unclaimed_e:
        # the population an older EMBED_CACHE_VERSION or RECIPE_VERSION lands
        # in: their names carry a token this code no longer writes, so they
        # match neither key and are never promoted into a current name
        print(f"           those {len(unclaimed_e)} are embeddings of some other "
              f"recipe or of models the walk cannot see — check, then delete")

    renders = args.renders or Path(args.cache_dir) / RENDERS_SUBDIR
    moves_r, already_r, unclaimed_r = plan_renders(
        files, renders, args.cache_dir, old_root, new_root)
    print(f"renders    {len(moves_r)} to re-key under {RENDERS_SUBDIR}/ from {renders}"
          + (f", {already_r} already there" if already_r else "")
          + (f", {len(unclaimed_r)} claimed by no model (left alone)" if unclaimed_r else ""))
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
    (Path(args.cache_dir) / "run-params.json").write_text(json.dumps(params, indent=2))
    stamp_cache_version(args.cache_dir)   # last: an interrupted run re-runs

    print(f"\nre-keyed {len(rekeyed)} poses, {len(moves_e)} embeddings and "
          f"{len(moves_r)} renders, stamped cache_version {CACHE_VERSION}")
    if dropped:
        print(f"dropped {len(dropped)} pose entries that matched no file")
    for n, what in ((len(unclaimed_e), f".npy under {EMBEDS_SUBDIR}/"),
                    (len(unclaimed_r), f"renders under {renders}")):
        if n:
            print(f"{n} {what} were claimed by no model and are still there — "
                  f"check them, then delete")


if __name__ == "__main__":
    main()
