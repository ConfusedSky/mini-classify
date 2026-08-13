"""One-off migration for renders written before render_key() existed.

Usage: python migrate_renders.py [/path/to/stls] --renders-dir my_renders2 [--apply]

Old renders were named "<stem>_view<i>.<ext>", so two STLs sharing a filename
wrote over each other and every tool showed one model's image for both. The new
name is "<stem>_<6 hex of the path>_view<i>.<ext>".

Where the old stem identifies exactly one file in the collection, its images are
renamed and the pixels are kept. Where a stem is shared, the images on disk are
one model's — which one is unknowable, since only the last file walked survived
— so they are deleted and those models re-render on the next
`classify_stls.py --save-renders` run.

Dry run by default; nothing on disk changes without --apply. Safe to re-run:
already-migrated files are recognised and left alone. Every render config
subdirectory under --renders-dir is migrated, not just the current one.
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

from classify_stls import (add_cache_args, apply_run_params, load_file_list,
                           render_key)

# "<stem>_view3" / "<stem>_pose" — the stem itself may contain anything, so the
# split has to come off the right-hand end
SUFFIX = re.compile(r"^(?P<stem>.+)_(?P<tail>view\d+|pose)$")


def plan_dir(rdir, by_stem, new_keys):
    """What to do with each image in one render config directory.

    Returns (renames, deletes, already, orphans): a list of (src, dst) pairs,
    a list of paths to remove, and counts of files needing neither."""
    renames, deletes, already, orphans = [], [], 0, 0
    for p in sorted(rdir.iterdir()):
        if not p.is_file():
            continue
        m = SUFFIX.match(p.stem)
        if not m:
            orphans += 1
            continue
        stem, tail = m.group("stem"), m.group("tail")
        if stem in new_keys:  # already carries a path hash
            already += 1
            continue
        files = by_stem.get(stem, [])
        if len(files) == 1:
            dst = p.with_name(f"{render_key(files[0])}_{tail}{p.suffix}")
            if dst.exists():
                already += 1  # migrated copy already there; the old one is dead weight
                deletes.append(p)
            else:
                renames.append((p, dst))
        elif len(files) > 1:
            deletes.append(p)
        else:
            orphans += 1  # no such STL under this root — someone else's file
    return renames, deletes, already, orphans


def main():
    parser = argparse.ArgumentParser()
    add_cache_args(parser, "STL directory (defaults to the last classify_stls.py run)")
    parser.add_argument("--renders-dir", help="renders saved by classify_stls.py "
                                              "--save-renders (defaults to the last run's)")
    parser.add_argument("--apply", action="store_true",
                        help="actually rename and delete; without it this only reports")
    args = apply_run_params(parser)
    if not args.input:
        sys.exit("no input given, and no directory recorded by classify_stls.py — "
                 "pass the STL directory explicitly")
    if not args.renders_dir:
        sys.exit("no --renders-dir given, and the last run recorded none")

    root = Path(args.renders_dir)
    if not root.is_dir():
        sys.exit(f"{root} is not a directory")

    files = load_file_list(Path(args.input), args.cache_dir)
    by_stem = defaultdict(list)
    for f in files:
        by_stem[f.stem].append(f)
    new_keys = {render_key(f) for f in files}
    shared = sum(len(v) for v in by_stem.values() if len(v) > 1)
    print(f"{len(files)} models, {len(by_stem)} distinct filenames — "
          f"{shared} models share a filename with another")

    # every render config subdir, plus the dir itself if renders were saved flat
    dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if any(p.is_file() for p in root.iterdir()):
        dirs.append(root)
    total = defaultdict(int)
    for d in dirs:
        renames, deletes, already, orphans = plan_dir(d, by_stem, new_keys)
        print(f"\n{d}: {len(renames)} to rename, {len(deletes)} to delete"
              + (f", {already} already migrated" if already else "")
              + (f", {orphans} unrecognised (left alone)" if orphans else ""))
        for src, dst in renames[:3]:
            print(f"  rename {src.name} -> {dst.name}")
        for p in deletes[:3]:
            print(f"  delete {p.name} (filename shared by "
                  f"{len(by_stem.get(SUFFIX.match(p.stem).group('stem'), []))} models)")
        if len(renames) + len(deletes) > 6:
            print(f"  ... {len(renames) + len(deletes) - 6} more")
        if args.apply:
            for src, dst in renames:
                src.rename(dst)
            for p in deletes:
                p.unlink()
        total["renamed"] += len(renames)
        total["deleted"] += len(deletes)

    verb = "renamed" if args.apply else "would rename"
    print(f"\n{verb} {total['renamed']}, "
          f"{'deleted' if args.apply else 'would delete'} {total['deleted']}")
    if not args.apply:
        print("dry run — nothing changed; pass --apply to do it")
    elif total["deleted"]:
        print("deleted images belonged to models sharing a filename; rerun "
              "classify_stls.py --save-renders to redraw them")


if __name__ == "__main__":
    main()
