"""Put the original timestamps back on a copy made without -a.

Usage: python restore_mtimes.py <source> <destination> [--apply]

A plain `cp -r` or a file-manager drag copies the bytes and stamps every file
with the time of the copy. The data is fine; the timestamps are not, and this
project's caches key on mtime — so a library copied that way loses every pose,
embedding and render even though nothing about the models changed.

While the source is still around the fix is exact: the two trees correspond by
relative path, so each destination file can take its source's timestamps back.

Refuses to touch anything unless the trees match on every path and every size.
That check is the whole safety argument — restoring timestamps onto a tree that
does *not* correspond would make a bad copy look like a good one, which is worse
than leaving the timestamps wrong, because a wrong timestamp is at least loud.

Directories are restored after their contents, deepest first: writing a file
updates its parent's mtime, so doing it in the other order would undo the work.

Dry run by default; nothing is written without --apply.
"""
import argparse
import os
import sys
from pathlib import Path


# ext4 puts this at the root of every filesystem it makes. It is not part of
# anyone's copy, and letting it count as an extra file would stop the run on
# every drive-to-drive restore.
FS_ARTIFACTS = {"lost+found"}


def manifest(root):
    """{relative path: (is_dir, size, atime_ns, mtime_ns)} for a whole tree."""
    out = {}
    for p in root.rglob("*"):
        if p.parent == root and p.name in FS_ARTIFACTS:
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        out[p.relative_to(root).as_posix()] = (
            p.is_dir(), 0 if p.is_dir() else st.st_size, st.st_atime_ns, st.st_mtime_ns)
    return out


def compare(src, dst):
    """(missing, extra, size_bad) — the reasons this must not run."""
    missing = sorted(set(src) - set(dst))
    extra = sorted(set(dst) - set(src))
    size_bad = sorted(k for k in set(src) & set(dst)
                      if not src[k][0] and src[k][1] != dst[k][1])
    return missing, extra, size_bad


def plan(src, dst):
    """Relative paths whose timestamps differ, files first then deepest dirs."""
    differ = [k for k in set(src) & set(dst) if src[k][3] != dst[k][3]]
    files = sorted(k for k in differ if not src[k][0])
    # deepest first: restoring a directory before its children would be undone
    dirs = sorted((k for k in differ if src[k][0]),
                  key=lambda k: k.count("/"), reverse=True)
    return files, dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="the tree with the original timestamps")
    parser.add_argument("destination", help="the copy to repair")
    parser.add_argument("--apply", action="store_true",
                        help="actually restore; without it this only reports")
    args = parser.parse_args()

    src_root, dst_root = Path(args.source), Path(args.destination)
    for r in (src_root, dst_root):
        if not r.is_dir():
            sys.exit(f"{r} is not a directory")

    print(f"reading {src_root}")
    src = manifest(src_root)
    print(f"reading {dst_root}")
    dst = manifest(dst_root)
    print(f"source {len(src)} entries, destination {len(dst)} entries")

    missing, extra, size_bad = compare(src, dst)
    if missing or extra or size_bad:
        print(f"\n  missing in destination: {len(missing)}")
        for k in missing[:5]:
            print(f"    {k}")
        print(f"  extra in destination:   {len(extra)}")
        for k in extra[:5]:
            print(f"    {k}")
        print(f"  size mismatch:          {len(size_bad)}")
        for k in size_bad[:5]:
            print(f"    {k}  src={src[k][1]} dst={dst[k][1]}")
        sys.exit("\nthe trees do not correspond — refusing to restore timestamps "
                 "onto a copy that is not the same copy")

    files, dirs = plan(src, dst)
    print(f"\ntrees correspond: same paths, same sizes")
    print(f"timestamps to restore: {len(files)} files, {len(dirs)} directories")
    for k in files[:3]:
        print(f"    {k}\n      {dst[k][3]} -> {src[k][3]}")
    if len(files) > 3:
        print(f"    ... {len(files) - 3} more")

    if not files and not dirs:
        print("nothing to do")
        return
    if not args.apply:
        print("\ndry run — nothing changed; pass --apply to restore")
        return

    failed = []
    for rel in files + dirs:
        try:
            os.utime(dst_root / rel, ns=(src[rel][2], src[rel][3]))
        except OSError as e:
            failed.append((rel, e))
    print(f"\nrestored {len(files) + len(dirs) - len(failed)} timestamps")
    if failed:
        print(f"{len(failed)} failed:")
        for rel, e in failed[:10]:
            print(f"    {rel}: {e}")


if __name__ == "__main__":
    main()
