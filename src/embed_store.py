"""Reading the embedding cache back: the .npy files a classify run wrote.

`src/done.py` is the only *writer* of these files (interfaces.md §Done). This
is the read side, and it is the read side for everyone — `test_categories.py`
and `cluster_models.py` both load the whole collection into one array before
they do anything else, and they used to do it by one REPL tool importing the
function out of the other.

Deliberately numpy-only: loading cached vectors is not a reason to load
SigLIP. `cluster_models.py` never touches torch, and this is the module that
lets it stay that way.
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
