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
import numpy as np

from src import pose
from src.cachedir import cache_key, embeds_dir


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
