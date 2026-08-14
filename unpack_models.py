"""Extract the per-model zips some sets ship inside their downloaded archive.

Most sets unzip straight to STLs — <Set>/DownloadAll_32mm/<Category>/<Group>/*.stl
— and the collection walk finds them. Two do not: Panshaw Under Siege and Snowy
Mountain Summit unzip to a tree of *per-model* zips,

    <Set>/DownloadAll_32mm/<Category>/<Model>/<Model>_NoSupports.zip

so the walk sees no STLs at all and every model in the set is silently absent.
Snowy Mountain Summit is the reason this is worth a script rather than a one-off
unzip: it contributes exactly one STL, so it looks present in every count while
~106 models are missing.

Usage: python unpack_models.py <dir> [--all] [--apply] [--repair PATH ...]
                                [--list-all]

Windows-authored archives often use Deflate64, which Python's zipfile cannot
decompress; those are handed to 7z (7zz/7za also work), so it is an optional
dependency you will want for any set zipped by Windows Explorer.

Dry run by default; nothing is written without --apply. Safe to re-run — a model
whose folder already holds files is skipped. Zips are never deleted, and an
existing destination is never replaced unless its zip is named (or covered) by
a --repair path: replacing a directory is destructive, so it is opt-in per zip.
A zip whose models all exist under its parent tree by name and size — extracted
by hand once and curated into a different layout — reports as `elsewhere` and
is left alone rather than re-extracted into a duplicate tree.

By default this extracts only the archives whose contents the collection walk
would keep, using the same `naming.skip` the walk itself uses: the supported,
LYCHEE, CHITUBOX, hollow and 75mm variants are left packed, since unpacking them
costs disk for models that are then filtered out anyway (3.14 GB against 0.70 GB
on one set). --all unpacks everything regardless.
"""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile
import zlib
from collections import Counter, defaultdict
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


# Presentation files whose absence must not force a re-extraction: curation
# routinely deletes renders and pdfs while keeping every model.
JUNK_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".pdf", ".db", ".url"}


def _crc32(p):
    crc = 0
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            crc = zlib.crc32(chunk, crc)
    return crc


def content_elsewhere(z, zpath, cache):
    """True when every model-class entry already exists under the zip's parent
    tree — a collection that was extracted by hand and then curated into a
    different layout. The zip is redundant; re-extracting it would duplicate
    gigabytes one directory deeper. Junk (renders, pdfs) is ignored: curation
    deletes those while keeping models.

    Every entry must match by basename and exact size, and a spread sample is
    verified byte-for-byte against the archive's own CRCs — name+size
    coincidences exist, and a false positive here means a zip is silently
    never unpacked, which is the exact failure this script exists to fix.

    `cache` maps a parent dir to its file index; the caller owns it and its
    lifetime, because this module cannot know when the tree changes."""
    if zpath.parent not in cache:
        idx = defaultdict(list)
        for dp, dn, fn in os.walk(zpath.parent):
            dn[:] = [d for d in dn if not d.endswith(".partial")]
            for f in fn:
                p = Path(dp) / f
                try:
                    idx[(f, p.stat().st_size)].append(p)
                except OSError:
                    pass
        cache[zpath.parent] = idx
    idx = cache[zpath.parent]
    ents = [i for i in z.infolist() if not i.is_dir()
            and Path(i.filename).suffix.lower() not in JUNK_SUFFIXES]
    if not ents:
        return False
    if any((Path(i.filename).name, i.file_size) not in idx for i in ents):
        return False
    step = max(1, len(ents) // 5)
    return all(any(_crc32(p) == i.CRC
                   for p in idx[(Path(i.filename).name, i.file_size)])
               for i in ents[::step][:5])


def plan_zip(zpath, unpack_all=False, cache=None):
    """(action, destination, bytes) for one zip. action is one of
    skip-tagged / done / elsewhere / repair / unsafe / unreadable / extract.

    `cache` (optional) carries the parent-tree index between calls in one
    planning pass; omit it and every call re-reads the tree, which is correct
    but slow across many zips in one directory."""
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
            # not under --all, which promises to unpack everything regardless
            if not unpack_all and content_elsewhere(z, zpath, {} if cache is None else cache):
                return "elsewhere", dest, size
    except (zipfile.BadZipFile, OSError) as e:
        return f"unreadable ({e})", None, 0
    return "extract", dest, size


def repair_authorized(zpath, wanted):
    """Repairs replace an existing directory, so they are opt-in per zip or
    per directory: authorized when the zip is, or sits under, a --repair path."""
    zp = zpath.resolve()
    return any(zp == q or q in zp.parents for q in (Path(w).resolve() for w in wanted))


# What Python's zipfile can decompress. Windows Explorer writes Deflate64
# (method 9) into large archives, which zipfile lists but cannot extract —
# those go through 7z instead.
PY_METHODS = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED,
              zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA}


def unzip_into(zpath, tmp):
    """Decompress the archive into tmp, validating CRCs either way."""
    with zipfile.ZipFile(zpath) as z:
        methods = {i.compress_type for i in z.infolist()}
        if methods <= PY_METHODS:
            z.extractall(tmp)  # raises on a CRC mismatch, so this validates too
            return
    exe = next((n for n in ("7z", "7zz", "7za") if shutil.which(n)), None)
    if not exe:
        odd = ", ".join(str(m) for m in sorted(methods - PY_METHODS))
        raise RuntimeError(f"compression method {odd} needs 7z, "
                           f"which is not installed")
    # 7z verifies CRCs as it goes and exits non-zero on any error
    run = subprocess.run([exe, "x", "-y", f"-o{tmp}", str(zpath)],
                         capture_output=True, text=True)
    if run.returncode:
        tail = (run.stderr or run.stdout).strip().splitlines()[-3:]
        raise RuntimeError(f"{exe}: " + (" | ".join(tail)
                                         or f"exited {run.returncode}"))


def extract(zpath, dest, replace=False):
    """Extract via a .partial sibling, then move into place.

    The collection lives on external media that has already lost one extraction
    half-way. Unpacking beside the target and renaming means an interrupted run
    leaves an obvious .partial directory rather than a folder that looks
    extracted but holds half a model.

    replace=False refuses an existing destination outright, before touching
    anything. Only an authorized repair may replace one — and the swap never
    has a moment where neither copy exists: the original moves aside, the
    replacement moves in, and only then is the old copy deleted, so a failure
    at any statement leaves either the original or the replacement in place.
    The refusal also catches two zips resolving to one destination in a single
    run (thingiverse-style archives share their author's name as the root
    dir), where the second extraction would otherwise destroy the first."""
    if dest.exists() and not replace:
        raise RuntimeError(f"{dest.name}/ already exists — repairs are "
                           f"opt-in (--repair), and two zips may be "
                           f"claiming one destination")
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        with zipfile.ZipFile(zpath) as z:
            names = z.namelist()
        unzip_into(zpath, tmp)
        _, owns_root = destination(names, zpath)
        staged = tmp
        if owns_root:
            # Adopt whatever single root actually got written rather than the
            # name zipfile predicted: 7z decodes no-UTF-8-flag entry names as
            # UTF-8 where zipfile uses cp437, so the two can disagree — and the
            # archives that need 7z (Windows-authored, Deflate64) are exactly
            # the ones that omit the flag.
            inner = list(tmp.iterdir())
            if len(inner) == 1 and inner[0].is_dir():
                staged = inner[0]
        if dest.exists():
            if not replace:   # appeared while we were extracting
                raise RuntimeError(f"{dest.name}/ appeared mid-extraction — "
                                   f"another zip claims this destination")
            aside = dest.with_name(dest.name + ".replaced")
            if aside.exists():
                shutil.rmtree(aside)
            dest.rename(aside)
            try:
                staged.rename(dest)
            except BaseException:
                aside.rename(dest)          # put the original back
                raise
            shutil.rmtree(aside)
        else:
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
    parser.add_argument("--repair", action="append", default=[], metavar="PATH",
                        help="authorize repairs (replacing an existing destination) "
                             "for this zip or anything under this directory; "
                             "repeatable. Without it repairs are reported, never run")
    parser.add_argument("--list-all", action="store_true",
                        help="list every planned zip instead of the first five")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"{root} is not a directory")

    zips = sorted(p for p in root.rglob("*.zip") if p.is_file())
    if not zips:
        sys.exit(f"no zips under {root}")

    todo, held, redundant, problems = [], [], [], []
    counts, total, cache = Counter(), 0, {}
    for p in zips:
        action, dest, size = plan_zip(p, args.unpack_all, cache)
        if action == "repair" and not repair_authorized(p, args.repair):
            counts["repair-held"] += 1
            held.append(p)
            continue
        counts[action.split(" ")[0]] += 1
        if action in ("extract", "repair"):
            todo.append((p, dest, size, action))
            total += size
        elif action == "elsewhere":
            redundant.append(p)
        elif action.startswith(("unsafe", "unreadable")):
            problems.append((p, action))

    print(f"{len(zips)} zips under {root}")
    for k in ("extract", "repair", "repair-held", "elsewhere", "done",
              "skip-tagged", "unsafe", "unreadable"):
        if counts[k]:
            print(f"  {k:<13} {counts[k]}")
    for p, action in problems:
        print(f"  ! {p.relative_to(root)}: {action}")
    shown_r = redundant if args.list_all else redundant[:5]
    for p in shown_r:
        print(f"  = {p.relative_to(root)}: content already on disk under "
              f"another layout — zip is redundant")
    if len(redundant) > len(shown_r):
        print(f"  ... {len(redundant) - len(shown_r)} more redundant "
              f"(--list-all shows them)")
    for p in held:
        print(f"  ~ {p.relative_to(root)}: destination differs from the archive — "
              f"repair with --repair '{p}'")
    if not todo:
        print("nothing to do")
        return
    n_rep = sum(1 for *_, a in todo if a == "repair")
    print(f"\n{len(todo)} to write, {total / 1e9:.2f} GB uncompressed"
          + (f" ({n_rep} repairing a destination whose files are missing or the "
             f"wrong size)" if n_rep else "") + ":")
    shown = todo if args.list_all else todo[:5]
    for p, dest, size, action in shown:
        verb = "repair" if action == "repair" else "->"
        print(f"  {p.relative_to(root)} {verb} {dest.name}/  ({size / 1e6:.0f} MB)")
    if len(todo) > len(shown):
        print(f"  ... {len(todo) - len(shown)} more (--list-all shows them)")

    if not args.apply:
        print("\ndry run — nothing changed; pass --apply to extract")
        return

    failed = []
    for i, (p, dest, _, action) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {p.name}"
              + ("  (repair)" if action == "repair" else ""), flush=True)
        try:
            extract(p, dest, replace=(action == "repair"))
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
