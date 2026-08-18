"""How a model file is named inside the caches, and the keys built from it.

Every cache in this project is keyed on a file's identity: the pose cache
(`pose.file_identity`), the embedding cache (`cache_key_from_identity`) and the
saved renders (`render_key`). The last two live here rather than beside their
callers because both sides of the pipeline need them and neither may import
the other: the parent's `cache_checker.route` and the child's `renderer`
each grew a byte-identical `render_key` before this move (E-R1-5), and
`classify_stls.py` held a third copy. This module is the deepest leaf — stdlib
only, so it costs nothing to import in the parent, the child, or a bare test —
which is what lets all of them share one definition.

All three used the *absolute* path,
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
import hashlib
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


# --- the keys themselves ------------------------------------------------------

# Versions the *derivation* of an embedding from its file: bump when
# load_mesh -> up_axis_scores -> rank_up_scores changes its answer for
# unchanged bytes, the way POSE_CACHE_VERSION already re-resolves poses. The
# numpy-parser swap was the near-miss (it passed only because triangle counts
# and bounding boxes came out exact). Appended to the key only when bumped, so
# every key from before it existed survives its introduction.
EMBED_CACHE_VERSION = 1

# The --elevations default. It lives here because the key elides it: a single
# 20 degree ring appends nothing, which is what keeps keys written before
# --elevations existed byte-identical. `cachedir.add_cache_args` declares the
# flag with it.
DEFAULT_ELEVATIONS = [20.0]


def cache_key_from_identity(ident, args, up_token):
    """The embedding key for a file already reduced to its identity string.

    `ident` is byte-identical to pose.file_identity — rel|mtime|size — which
    is what makes every embedding key reconstructible from pose-cache.json
    plus run-params.json alone, with no filesystem access (review §P3.1).
    migrate_cache_keys drives the token migration through here so a
    half-mounted collection cannot leave entries behind (S1).

    Every conditional token appends only when non-default, so keys written
    before each flag existed stay byte-identical and those (expensive) caches
    survive."""
    elev = "" if args.elevations == DEFAULT_ELEVATIONS else \
        "|e:" + ",".join(f"{e:g}" for e in args.elevations)
    # "pv" = per-view cache format: (n_views, dim) instead of one pooled vector.
    # up_token is the pose's up vector ("0,0,1"), the only pose input that
    # changes the pixels — pose.embed_cache_token.
    ver = "" if EMBED_CACHE_VERSION == 1 else f"|ev{EMBED_CACHE_VERSION}"
    # torch.compile's kernels drift ~1e-03 from eager, so the two regimes are
    # different numbers under the same pixels; like elev, the token appears
    # only when non-default.
    comp = "|compiled" if getattr(args, "compile", False) else ""
    raw = f"{ident}|{args.views}|{args.render_size}|{up_token}|{args.model}|pv{elev}{comp}{ver}"
    return hashlib.sha1(raw.encode()).hexdigest()


def render_key(f, root):
    """Per-file prefix for saved renders: '<stem>_<6 hex of the rel path>'.

    Stems are not unique — a collection routinely holds one Baal_Flaming_Sword_L
    per kit — and the renders all land in one flat directory, so keying by stem
    alone let the last file walked overwrite the others' images and every tool
    then showed one model's render for all of them. The path disambiguates;
    mtime and size deliberately do not, so re-rendering a file replaces its own
    images instead of accumulating a set per edit. Only the path is hashed, so
    the stem stays readable and searchable in a directory listing.

    The path hashed is relative to the collection root, so moving the library
    does not orphan every render. The child writes these files and the parent's
    render index reads them, which is why the definition is here and not in
    either (module docstring)."""
    return f"{f.stem}_{hashlib.sha1(rel_path(f, root).encode()).hexdigest()[:6]}"
