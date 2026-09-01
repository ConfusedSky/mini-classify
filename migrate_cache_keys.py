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
no file, which are dropped, because `load_pose_cache` filters on version and
shape, never on whether the file still exists, and would carry them forever.

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
                          load_run_params, stamp_cache_version, write_atomic)
from src.identity import cache_key_from_identity, render_key

# "<render key>_view3" / "<render key>_pose", split off the right-hand end
# because the stem itself may contain anything.
SUFFIX = re.compile(r"^(?P<key>.+)_(?P<tail>view\d+|pose)$")


# --- planning ---------------------------------------------------------------

def plan_poses(files, cache_dir, old_root, new_root):
    """(rekeyed, kept, dropped, unreadable) for pose-cache.json.

    `dropped` are entries no walked file claims — a model deleted or renamed
    since the cache was written. They are returned rather than discarded
    quietly so a VLM-sourced one can be called out before it goes.

    `unreadable` are entries this migration re-keys and copies **unchanged**
    that `pose.load_pose_cache` will then drop on the next run: an older
    `v`, or one of the shapes `pose._readable` refuses (adversarial review
    pass 3, 2026-08-31). Counted, never touched — re-keying is a rename, not
    a repair, and a migration that quietly deleted entries would be deciding
    something it was not asked to. Reporting them is the point: a run that
    re-keys 3540 poses and re-resolves 53 of them anyway should say so before
    the re-render bill arrives, not after. Imported rather than mirrored so
    there is exactly one copy of the rule."""
    path = Path(cache_dir) / "pose-cache.json"
    if not path.exists():
        return {}, {}, {}, {}
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
    unreadable = {k: v for k, v in rekeyed.items() if not pose._readable(v)}
    return rekeyed, cache, dropped, unreadable


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


def migration_root(inp, recorded):
    """The root to re-key *to*, refusing the one scoping that re-keying eats.

    Deliberately `identity.resolve_root` and not `identity.collection_root`:
    the anchor is the recorded root's to keep, and a subdir-scoped run is
    exactly where the two answers diverge. `collection_root` would answer the
    subdir, every key would be recomputed relative to it, and `plan_poses`
    walks only that subdir — so every pose outside it is claimed by no file
    and `--apply` drops the lot. Nothing there needs migrating in any case:
    the recorded root still anchors those keys, which is the whole point of
    `resolve_root` keeping it.

    Every other case re-keys onto the resolved root, which is the migration
    this tool exists for: "superdir" (the library grew around the cached kit)
    and "mismatch" (it moved somewhere else entirely) both answer the input's
    own root, as before. A loose-file input is the one place the answer also
    changes — it keeps the recorded anchor, so the run reports nothing to
    migrate rather than re-keying a whole collection onto one file's parent
    directory."""
    root, note = identity.resolve_root(inp, recorded)
    if note == "subdir":
        sys.exit(f"{Path(inp).resolve()} is inside the recorded collection root "
                 f"{recorded} — nothing to migrate: keys under it are still "
                 f"anchored there, and re-keying relative to a subdir would "
                 f"drop every pose outside it on --apply. Re-run against "
                 f"{recorded} if the root itself has moved.")
    return root


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
    anchored = params.get("collection_root")
    new_root = migration_root(args.input, anchored)
    old_root = Path(anchored) if anchored else new_root

    print(f"cache      {args.cache_dir}")
    if old_root == new_root:
        print(f"anchored at {new_root} — nothing to migrate")
        return

    print(f"old keys   relative to {old_root}")
    print(f"new keys   relative to {new_root}")

    files = load_file_list(Path(args.input), args.cache_dir, args.rescan)
    print(f"collection {len(files)} models\n")

    rekeyed, old_cache, dropped, unreadable = plan_poses(
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
    if unreadable:
        # copied unchanged, then dropped by load_pose_cache on the next run:
        # the re-render is owed either way, and this is where it becomes
        # visible instead of arriving as an unexplained re-pose
        print(f"           {len(unreadable)} entries the loader will drop are "
              f"copied unchanged — an older v, or a shape it cannot process; "
              f"they will be re-resolved on the next run")

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

    # poses first: an embedding's key contains the pose's up-token. Written
    # here rather than through `pose.save_pose_cache`, whose bare write_text
    # is the evals': a crash mid-write tears the most expensive file in the
    # cache, and re-running is this tool's whole recovery story. Same
    # one-liner as `Done.flush`, so the two writers stay byte-identical.
    p = Path(args.cache_dir) / "pose-cache.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(p, json.dumps(rekeyed))
    move_all(moves_e)
    move_all(moves_r)
    params["collection_root"] = str(new_root)
    write_atomic(Path(args.cache_dir) / "run-params.json",
                 json.dumps(params, indent=2))
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
