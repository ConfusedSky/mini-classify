"""How a model file is named inside the caches.

Every cache in this project is keyed on a file's identity: the pose cache
(`pose.file_identity`), the embedding cache (`classify_stls.cache_key`) and the
saved renders (`classify_stls.render_key`). All three used the *absolute* path,
which made the whole collection pinned to wherever it happened to be mounted —
moving the library to another drive changed every key at once and threw away
1806 pose resolutions, ~26k renders and every embedding, at a cost of hours and
of real money, since a pose entry sourced from the VLM billed per gemini call.

Keys derive from the path *relative to the collection root* instead, so the
library can move between drives and the caches follow it. The sister project
(model-browser) already keys its thumbnails on root-relative virtual paths;
this brings the two in line.

Two things this does NOT make portable, both deliberate:

* `mtime` and `size` still take part in the identity, because a file whose
  content changed is a different model. A move must therefore preserve mtimes —
  `mv` within a filesystem does, `cp -a`/`rsync -a` do, a plain `cp` does not
  and would invalidate every entry no matter what the path says.
* The walk cache (`load_file_list`) still keys on the absolute root, so a move
  simply rescans. That is the one cheap cache of the four, and rescanning is
  the correct response to a directory tree that moved.
"""
from pathlib import Path


def collection_root(inp):
    """The directory keys are taken relative to.

    A directory input is its own root. A single-file input (`classify_stls.py
    model.stl`) has no collection, so its parent stands in — which keeps that
    file's key stable as long as it stays beside its neighbours.

    The test is `is_file`, not `is_dir`: a path that does not exist yet has to
    read as a directory, or an unmounted drive and a mistyped path both
    silently anchor a whole cache to the *parent* of what was asked for."""
    p = Path(inp).resolve()
    return p.parent if p.is_file() else p


def mtime_key(stat):
    """Modification time truncated to whole seconds.

    Full `st_mtime_ns` does not survive a change of filesystem, which makes it
    the wrong half of an identity that is supposed to let the library move.
    exFAT — where this collection lives — stores timestamps at 10 ms
    granularity; ext4 keeps nanoseconds. So a file written on ext4 (a new kit, a
    re-exported STL) truncates on the way back to exFAT and silently loses its
    entry, `rsync -a` or not.

    Whole seconds survives every filesystem this library is likely to touch;
    FAT32's 2-second rounding is the exception, and nothing is running a 45 GB
    STL library off FAT32. The cost is that an edit landing in the same second
    as the cached one goes unnoticed — but size is in the identity too, so it
    has to be a same-second edit that also preserves the byte count exactly."""
    return stat.st_mtime_ns // 1_000_000_000


def resolve_root(inp, recorded):
    """Which root to key against, and whether that needs saying out loud.

    Returns (root, note). The note is None when there is nothing to report,
    "subdir" when this run is scoped to part of the recorded collection — the
    recorded root is kept, so a run on one kit still hits the cache built by a
    run on the whole library — and "mismatch" when the input is somewhere else
    entirely.

    A mismatch is not decided here. It is either the library having moved,
    where re-keying is exactly right and costs nothing because the keys are
    relative, or a cache directory pointed at the wrong collection, where
    re-keying silently re-renders and re-embeds everything and bills the
    arbiter again. Nothing in the paths distinguishes those, so the caller
    asks."""
    root = collection_root(inp)
    if recorded is None:
        return root, None
    recorded = Path(recorded)
    if root == recorded:
        return root, None
    if Path(inp).resolve().is_file():
        # a loose file is not a collection and must not move the anchor. Its
        # key falls back to an absolute path, which is the honest description
        # of a file that belongs to no library.
        return recorded, None
    if root.is_relative_to(recorded):
        return recorded, "subdir"
    if recorded.is_relative_to(root):
        # the library grew upward: the cache was built on one kit and the run
        # now covers the whole collection. Every existing key is still valid,
        # it just needs the intervening directories on the front, so this is
        # re-keyable rather than lost.
        return root, "superdir"
    return root, "mismatch"


def rel_path(f, root):
    """A file's identity within the collection, as a POSIX-style string.

    Falls back to the absolute path for anything outside the root rather than
    raising: a file reached through a symlink out of the tree still gets a
    stable, correct key, it just is not relocatable. Returning something
    usable matters more here than refusing, because the alternative is a run
    that dies partway through a collection."""
    p = Path(f).resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()
