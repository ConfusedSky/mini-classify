"""Extract the per-model zips some sets ship inside their downloaded archive.

Most sets unzip straight to STLs — <Set>/DownloadAll_32mm/<Category>/<Group>/*.stl
— and the collection walk finds them. Two do not: Panshaw Under Siege and Snowy
Mountain Summit unzip to a tree of *per-model* zips,

    <Set>/DownloadAll_32mm/<Category>/<Model>/<Model>_NoSupports.zip

so the walk sees no STLs at all and every model in the set is silently absent.
Snowy Mountain Summit is the reason this is worth a script rather than a one-off
unzip: it contributes exactly one STL, so it looks present in every count while
~106 models are missing.

Usage: python unpack_models.py <dir> [--all] [--apply]

Dry run by default; nothing is written without --apply. Safe to re-run — a model
whose folder already holds files is skipped. Zips are never deleted.

By default this extracts only the archives whose contents the collection walk
would keep, using the same `naming.skip` the walk itself uses: the supported,
LYCHEE, CHITUBOX, hollow and 75mm variants are left packed, since unpacking them
costs disk for models that are then filtered out anyway (3.14 GB against 0.70 GB
on one set). --all unpacks everything regardless.
"""
import argparse
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

import naming


def entries_escaping(names):
    """Entries that would write outside the destination — absolute paths or ..

    zipfile sanitises these on extract, but a zip carrying them is not the shape
    this script expects and is worth refusing rather than silently mangling."""
    return [n for n in names if n.startswith("/") or ".." in Path(n).parts]


def destination(names, zpath):
    """Where this archive's contents belong, and whether it brings its own root.

    The leaf zips carry a single top-level folder named after themselves
    ("Barrier_NoSupports/32mm_Barrier.stl"), so they extract beside the zip.
    Anything without one common root gets a folder named after the zip instead,
    so two archives in the same directory cannot interleave."""
    tops = {n.split("/")[0] for n in names if n.strip("/")}
    if len(tops) == 1:
        return zpath.parent / tops.pop(), True
    return zpath.parent / zpath.stem, False


def damaged(z, dest, owns_root):
    """Entries the destination is missing, or holds at the wrong size.

    The archive is the authority on what a finished extraction looks like, so
    "already done" is checked against it rather than against the directory
    merely existing. Anything else calls a half-written extraction finished —
    which is what a drive going read-only mid-run leaves behind: the tree in
    place and the files inside it zero bytes long."""
    bad = []
    for info in z.infolist():
        if info.is_dir():
            continue
        rel = info.filename
        if owns_root:                       # entries carry the archive's own root
            rel = rel.split("/", 1)[1] if "/" in rel else rel
        p = dest / rel
        if not p.is_file() or p.stat().st_size != info.file_size:
            bad.append(rel)
    return bad


def plan_zip(zpath, unpack_all=False):
    """(action, destination, bytes) for one zip. action is one of
    skip-tagged / done / repair / unsafe / unreadable / extract."""
    if not unpack_all and naming.skip(zpath.stem):
        return "skip-tagged", None, 0
    try:
        with zipfile.ZipFile(zpath) as z:
            names = z.namelist()
            size = sum(i.file_size for i in z.infolist())
            if entries_escaping(names):
                return "unsafe", None, 0
            dest, owns_root = destination(names, zpath)
            if dest.is_dir() and any(dest.iterdir()):
                return ("repair" if damaged(z, dest, owns_root) else "done"), dest, size
    except (zipfile.BadZipFile, OSError) as e:
        return f"unreadable ({e})", None, 0
    return "extract", dest, size


def extract(zpath, dest):
    """Extract via a .partial sibling, then move into place.

    The collection lives on external media that has already lost one extraction
    half-way. Unpacking beside the target and renaming means an interrupted run
    leaves an obvious .partial directory rather than a folder that looks
    extracted but holds half a model. An existing destination is replaced only
    after the replacement is complete, which is what makes a repair safe."""
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        with zipfile.ZipFile(zpath) as z:
            names = z.namelist()
            z.extractall(tmp)  # raises on a CRC mismatch, so this validates too
        _, owns_root = destination(names, zpath)
        staged = tmp / dest.name if owns_root else tmp
        if dest.exists():
            # a repair: swap only now that the fresh copy is staged and intact,
            # so a failure above costs nothing that was already there
            shutil.rmtree(dest)
        staged.rename(dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="extract per-model zips left inside a downloaded set")
    parser.add_argument("root", help="directory to search, e.g. a set folder or "
                                     "the whole collection")
    parser.add_argument("--all", dest="unpack_all", action="store_true",
                        help="unpack every archive, including the supported, LYCHEE, "
                             "CHITUBOX, hollow and 75mm variants the collection walk "
                             f"filters out anyway (tags: {', '.join(naming.SKIP_TAGS)})")
    parser.add_argument("--apply", action="store_true",
                        help="actually extract; without it this only reports")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"{root} is not a directory")

    zips = sorted(p for p in root.rglob("*.zip") if p.is_file())
    if not zips:
        sys.exit(f"no zips under {root}")

    todo, counts, total = [], Counter(), 0
    for p in zips:
        action, dest, size = plan_zip(p, args.unpack_all)
        counts[action.split(" ")[0]] += 1
        if action in ("extract", "repair"):
            todo.append((p, dest, size, action))
            total += size
        elif action.startswith(("unsafe", "unreadable")):
            print(f"  ! {p.relative_to(root)}: {action}")

    print(f"{len(zips)} zips under {root}")
    for k in ("extract", "repair", "done", "skip-tagged", "unsafe", "unreadable"):
        if counts[k]:
            print(f"  {k:<13} {counts[k]}")
    if not todo:
        print("nothing to do")
        return
    n_rep = sum(1 for *_, a in todo if a == "repair")
    print(f"\n{len(todo)} to write, {total / 1e9:.2f} GB uncompressed"
          + (f" ({n_rep} repairing a destination whose files are missing or the "
             f"wrong size)" if n_rep else "") + ":")
    for p, dest, size, action in todo[:5]:
        verb = "repair" if action == "repair" else "->"
        print(f"  {p.relative_to(root)} {verb} {dest.name}/  ({size / 1e6:.0f} MB)")
    if len(todo) > 5:
        print(f"  ... {len(todo) - 5} more")

    if not args.apply:
        print("\ndry run — nothing changed; pass --apply to extract")
        return

    failed = []
    for i, (p, dest, _, action) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {p.name}"
              + ("  (repair)" if action == "repair" else ""), flush=True)
        try:
            extract(p, dest)
        except Exception as e:  # keep going: one bad zip should not stop the set
            failed.append((p, e))
            print(f"  ! failed: {type(e).__name__}: {e}")
    print(f"\nwrote {len(todo) - len(failed)} of {len(todo)}")
    if failed:
        print(f"{len(failed)} failed — rerun to retry just those:")
        for p, e in failed[:10]:
            print(f"  {p.relative_to(root)}: {e}")
    else:
        print("rerun classify_stls.py to pick up the new models")


if __name__ == "__main__":
    main()
