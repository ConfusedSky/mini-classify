"""Reading the embedding cache back: the .npy files a classify run wrote.

`src/done.py` is the only *writer* of these files (interfaces.md §Done). This
is the read side, and it is the read side for everyone — `test_categories.py`
and `cluster_models.py` both load the whole collection into one array before
they do anything else, and they used to do it by one REPL tool importing the
function out of the other.

Deliberately light: loading cached vectors is not a reason to load SigLIP —
or a renderer. numpy and the three cache-identity functions it takes from
`pose`, and nothing else; `cluster_models.py` reads the whole collection
without importing torch or open3d.

That second half was false when this module was written (2026-08-18): `from
src import pose` pulled open3d at module scope, so a tool that reads `.npy`
files off disk loaded a rendering library — 2602 modules, found by review.
The fix was in `pose`, which now defers its one open3d use into
`up_axis_scores`; this import costs 201 modules.
"""
from pathlib import Path

import numpy as np

from src import pose
from src.cachedir import cache_key, embeds_dir
from src.identity import cache_key_from_identity


def load_embedding_matrix(files, args, root):
    """Every file's cached views as one (n_files, n_views, dim) float32 array.

    Returns (matrix, kept, missing): the files that had no cached entry are
    dropped from `kept` and counted, because a run scoped to part of the
    library legitimately walks files the cache has never seen. The pose cache
    is consulted for each file's up-token, which is part of the embedding key
    (`pose.embed_cache_token`)."""
    cache_dir = embeds_dir(args.cache_dir)
    poses = pose.load_pose_cache(args.cache_dir)
    vecs, kept, missing = [], [], 0
    for f in files:
        token = pose.embed_cache_token(poses.get(pose.file_identity(f, root)), args.up_axis)
        p = cache_dir / f"{cache_key(f, args, token, root)}.npy"
        if p.exists():
            vecs.append(np.load(p))
            kept.append(f)
        else:
            missing += 1
    if not vecs:
        raise SystemExit("no cached embeddings found — run classify_stls.py first")
    return np.stack(vecs).astype(np.float32), kept, missing  # (n_files, n_views, dim)


def _recency(ident, parts):
    """Order two identities of one file: the newer export sorts higher.

    `(mtime, size)` is the identity's own second half, so this orders exports
    the way a stat would without taking one — not the same thing as making the
    walk's choice, which is the file's current identity and not the newest
    known (see `load_embedding_matrix_from_poses`; F5).

    A field that is not an integer is not something this cache wrote and loses
    to any field that is — the same call as dropping an identity that does not
    split into three, and total rather than a raise, since pose-cache.json is
    hand-editable. The identity
    string breaks the remaining ties, which makes the winner deterministic
    across runs rather than a function of dict order."""
    try:
        return (1, int(parts[1]), int(parts[2]), ident)
    except ValueError:
        return (0, 0, 0, ident)


def load_embedding_matrix_from_poses(poses, args, root, prefix=()):
    """The same array, enumerated from the pose cache instead of from a walk —
    the `--no-volume` load (serve_api.py, `Collection.load`).

    Returns (matrix, files, idents, missing): `load_embedding_matrix`'s three,
    plus the identity behind each row. The identities are handed back because
    recomputing them is `pose.file_identity`, which stats the file — the one
    thing this path exists not to do.

    `prefix` is the input's offset from the collection root as parts
    (`collection._input_prefix`), and it buys the parity the mode rests on: a
    walk serves only what is under `args.input`, so an unscoped enumeration of
    the whole pose cache made a server started on `root/a` answer for `root/b`
    as well — the two loaders disagreeing about which models exist, which is
    the one thing they may not do. An identity outside the prefix is **not**
    counted in `missing`: it is out of scope, not absent, exactly as a walk of
    a subdirectory never sees it and never reports it.

    **No filesystem access outside `cache_dir`.** A pose entry is keyed by
    `rel|mtime|size`, which is byte-identical to what `cache_key` builds from a
    file's own stat, so `identity.cache_key_from_identity` rebuilds every
    embedding key from the cache's own records (review §P3.1). The volume is
    never consulted, and the `Path` built here names a file nothing opens.

    Split on the **right**: a filename may contain "|", and the two fields
    after it — the truncated mtime and the size — never do. An identity that
    does not split into three is not one this cache wrote; it is dropped and
    counted like any other row with nothing behind it, since pose-cache.json is
    hand-editable and one bad key must not cost the whole index.

    A pose entry with no `.npy` is dropped and counted in `missing`, which is
    the same field and the same meaning as the walk-driven loader's: known to
    the cache, not in the index. Here it covers a model that has a pose and no
    embedding yet, one whose render failed, one embedded under a different
    cache identity than these args name — and the **superseded** entry below.

    **One row per `rel`, at its newest identity.** Nothing prunes
    pose-cache.json, so re-exporting a file (new mtime, new size) leaves the
    old identity and its `.npy` on disk beside the new pair, and a walk never
    sees the stale one: it stats the file and gets exactly one identity.
    Iterating the pose cache does see both, and two rows for one file is not a
    cosmetic duplicate — `Collection`'s `_row_by_rel`/`_row_by_path` collision
    rule holds only because colliding rows share a pose entry, an embedding key
    and a render key, which two exports of one model do not. `row_of` (so
    `POST /poses`) would answer from whichever row won insertion, and the model
    would appear twice in one result list under two scores. Not hypothetical:
    docs/cache-rebuild.md measured 3540 pose entries against 3396 loaded models
    on embed-cache2 — 144 orphans. Newest wins, by `(mtime, size)` descending
    with an unparseable field losing to any parseable one.

    That matches the walk exactly when the file on disk is its latest export,
    which is the ordinary case and the one worth optimising for — but it is not
    the walk's rule, and saying so overclaimed (F5). Restore a file to an older
    export (old bytes *and* old mtime back, as a backup restore gives) and the
    two diverge: the walk stats the file and serves the older identity, this
    loader serves the newest identity it has, and only a walk can know which
    one is current. Keep-newest is still the right call without a disk — the
    alternative is guessing — it is just a guess, not a reproduction.

    **The residual, which is the trade the flag opts into.** A file *deleted or
    renamed* in the library leaves an orphaned identity at a rel that exists
    nowhere, and with no disk to ask, that row stays in the index and answers
    queries. That is exactly the "answer from a snapshot it cannot check" cost
    `VolumeUnavailable`'s docstring names, and it is priced rather than hidden:
    the walk-driven mode is what trims such a row, because only a walk can know
    the file is gone."""
    if not poses:
        raise SystemExit(
            f"no pose entries in {args.cache_dir} — --no-volume builds the "
            f"index from pose-cache.json, and a cache built entirely under a "
            f"forced --up-axis records no poses to build it from.\n"
            f"  run: serve_api.py without --no-volume to index from the STL "
            f"volume instead")
    cache_dir = embeds_dir(args.cache_dir)
    # Compared as a posix string rather than by parts, because the stored rel
    # *is* a posix string (`identity.rel_path`) and splitting it back into
    # parts to compare would only invent a second spelling of the same test.
    # The trailing "/" is what keeps `a` from claiming `ab`.
    scope = "/".join(prefix)
    by_rel, missing = {}, 0
    for ident, entry in poses.items():
        parts = ident.rsplit("|", 2)
        if len(parts) != 3:
            missing += 1
            continue
        if scope and not (parts[0] == scope
                          or parts[0].startswith(scope + "/")):
            continue                    # out of scope, not missing (F3)
        # An identity for a file outside the root carries an absolute posix
        # path (`identity.rel_path`'s fallback), and joining an absolute path
        # onto the root yields that path — which is the right answer for a
        # model the collection does not contain.
        f = Path(root) / parts[0]
        p = cache_dir / (cache_key_from_identity(
            ident, args, pose.embed_cache_token(entry, args.up_axis)) + ".npy")
        if not p.exists():
            missing += 1
            continue
        row = (_recency(ident, parts), f, ident, p)
        prior = by_rel.get(parts[0])
        if prior is None:
            by_rel[parts[0]] = row
            continue
        # A second live identity for one rel: the file was re-exported and
        # nothing pruned the old entry. Whichever loses is superseded, not
        # absent, and `missing` is where a row known to the cache and not in
        # the index is counted.
        missing += 1
        if row[0] > prior[0]:
            by_rel[parts[0]] = row
    rows = [r[1:] for r in by_rel.values()]
    if not rows:
        # Not the walk loader's "run classify_stls.py first": classify cannot
        # run where the volume does not go, so that is advice this mode's
        # operator can never take (F7). Pose entries were read, and none of
        # them keyed a `.npy` — which is the signature of args that name a
        # different cache identity than the one on disk, not of a cache that
        # was never built. `run:` is the line `Collection.load` harvests into
        # `CacheUnusable.hint`.
        raise SystemExit(
            f"pose entries were read from {args.cache_dir}, and none names a "
            f"cached embedding under the keys these args build — most likely "
            f"the cache-identity flags do not name the cache that is there "
            f"(--views/--elevations/--render-size/--model/--compile).\n"
            f"  run: compare those flags against {args.cache_dir}/"
            f"run-params.json, which records what the classify run used")
    # The walk's order, reproduced. The two loaders index the same collection,
    # and a row number that names a different model depending on how the server
    # was started is not one anything can hold — `Scope.rows` and every hit
    # carry row indices.
    #
    # The key is the **Path**, deliberately, because that is `find_stls`' own
    # comparison (`sorted(found)` over Path objects) rather than something that
    # happens to agree with it: `Path.__lt__` orders by the parts tuple, and a
    # str sort disagrees with it on exactly `Kits/A-B` vs `Kits/A` — '-' sorts
    # below '/', so the string puts `Kits/A-B` first and the parts put `Kits/A`
    # first. That is a real shape in this collection, not a hypothetical (see
    # `collection.in_rel_order`), and it is what the parametrized parity case
    # in tests/test_collection.py guards.
    rows.sort(key=lambda r: r[0])
    matrix = np.stack([np.load(p) for _, _, p in rows]).astype(np.float32)
    return (matrix, [f for f, _, _ in rows], [ident for _, ident, _ in rows],
            missing)
