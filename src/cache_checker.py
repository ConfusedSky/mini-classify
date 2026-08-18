"""The Cache Checker: `route()`, the admission decision (interfaces.md
§"Cache Checker — a pure decision").

Extracted from the cache-decision half of `classify_stls.process()`
(classify_stls.py:1096-1160): the pose-cache lookup, the embedding-cache
`.npy` presence test, the saved-render presence/redraw decision, and the
`--skip-embed` / `--save-renders` / `--up-axis` flag interplay. `route`
**reads** caches and never writes, renders, or embeds; the whole decision is
in the return value (I14), and it raises rather than guarding — the J3 error
boundary at the call site is the driver's job.

Order matters (actors_proposal.md §Cache Checker): the embedding key includes
the resolved pose token, so under `--up-axis auto` the embedding cache cannot
be consulted until the pose is known. A forced `--up-axis z|y` is the one
path that skips the pose-cache lookup and goes straight to the embedding key
(the migration-notes shortcut).

Import discipline: no torch, no renderer, no classify_stls (which imports
both). Cache keying goes through `src.identity` / `src.pose` —
`pose.file_identity` is byte-identical to the embedding key's identity half
(review §P3.1), so this module never re-derives an identity string.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from src import identity, pose
from src.messages import (
    CacheContext,
    CachedHit,
    EmbedRenderTask,
    PoseRenderTask,
    Redraw,
    Retired,
)

# --- Embedding / render cache keys ------------------------------------------
# Extracted verbatim from classify_stls.py (EMBED_CACHE_VERSION,
# DEFAULT_ELEVATIONS, cache_key_from_identity:608, render_key:321). The
# constants are replicated rather than imported because classify_stls imports
# torch and open3d.rendering at module scope — exactly what this module must
# not pull in. classify_stls delegating to this copy is a later wave's move;
# until then the parity test in tests/test_cache_checker.py pins the two
# byte-identical.

EMBED_CACHE_VERSION = 1
DEFAULT_ELEVATIONS = [20.0]


def cache_key_from_identity(ident, args, up_token):
    """The embedding key for a file already reduced to its identity string.

    `ident` is `pose.file_identity` — rel|mtime|size. Every conditional token
    appends only when non-default so keys written before each flag existed
    stay byte-identical (see the source's comments for the full history)."""
    elev = "" if args.elevations == DEFAULT_ELEVATIONS else \
        "|e:" + ",".join(f"{e:g}" for e in args.elevations)
    ver = "" if EMBED_CACHE_VERSION == 1 else f"|ev{EMBED_CACHE_VERSION}"
    comp = "|compiled" if getattr(args, "compile", False) else ""
    raw = f"{ident}|{args.views}|{args.render_size}|{up_token}|{args.model}|pv{elev}{comp}{ver}"
    return hashlib.sha1(raw.encode()).hexdigest()


def render_key(f, root):
    """Per-file prefix for saved renders: '<stem>_<6 hex of the rel path>'."""
    return f"{f.stem}_{hashlib.sha1(identity.rel_path(f, root).encode()).hexdigest()[:6]}"


# --- The decision ------------------------------------------------------------

def route(f: Path, index: int, ctx: CacheContext) \
        -> PoseRenderTask | EmbedRenderTask | CachedHit | Redraw | Retired:
    """One file's cache decision. Raises on I/O errors (a vanished file, an
    unreadable directory) — the driver converts to `Failure` (J3)."""
    args = ctx.args

    # Pose: forced axis skips the pose-cache lookup entirely (migration-notes
    # shortcut; classify_stls.py:1100-1102). Otherwise a miss — or a
    # geometry-only pose this run's ensemble can upgrade — is a PoseRenderTask.
    entry = None
    if args.up_axis in pose.FORCED_UPS:
        resolved = pose.Pose(up=pose.FORCED_UPS[args.up_axis], confidence=0.0,
                             source="forced", v=pose.POSE_CACHE_VERSION)
    else:
        entry = ctx.poses.get(pose.file_identity(f, ctx.root))
        if not pose.pose_is_sufficient(entry, bool(args.up_ensemble)):
            return PoseRenderTask(file=f, index=index)
        resolved = pose.Pose.from_cache(entry)

    # Embedding cache: entry=None on the forced path — embed_cache_token then
    # derives the token from the flag itself ("a forced --up-axis needs no
    # pose entry; its up is the flag").
    token = pose.embed_cache_token(entry, args.up_axis)
    cache_file = None
    if ctx.embeds_dir is not None:
        ident = pose.file_identity(f, ctx.root)
        cache_file = Path(ctx.embeds_dir) / \
            f"{cache_key_from_identity(ident, args, token)}.npy"
    cached = cache_file is not None and cache_file.exists()
    need_embeds = not cached and not args.skip_embed

    # Renders: today's need_renders rule (classify_stls.py:1157) minus
    # pose_changed, which is always False here — every pose route sees came
    # from the cache or the flag, and a fresh resolution's redraw is the
    # Poser's path, not this one.
    renders_wanted = bool(args.save_renders) and bool(args.cache_dir)
    need_renders = False
    if renders_wanted:
        rkey = render_key(f, ctx.root)
        n_views = args.views * len(args.elevations)
        need_renders = not all(f"{rkey}_view{i}" in ctx.render_index
                               for i in range(n_views))

    if need_embeds:
        # The child saves renders whenever it renders (config at spawn), so a
        # fresh-embed task never needs a separate render decision.
        return EmbedRenderTask(file=f, index=index, pose=resolved,
                               needs_embed=True)
    if args.skip_embed:
        # No CachedHit doing scoring work the flag exists to skip (J1/Q2):
        # renders wanted -> a plain task that retires on its Rendered ack;
        # nothing wanted -> retire directly, no row.
        if need_renders:
            return EmbedRenderTask(file=f, index=index, pose=resolved,
                                   needs_embed=False)
        return Retired(file=f, index=index)
    # Embedding cached, scoring wanted: the hit writes the row. On the redraw
    # path it carries retires=False — the row comes from the hit, retirement
    # from the child's Rendered ack (§P2.3, D8).
    hit = CachedHit(file=f, index=index, pose=resolved, cache_file=cache_file,
                    retires=not need_renders)
    if need_renders:
        return Redraw(task=EmbedRenderTask(file=f, index=index, pose=resolved,
                                           needs_embed=False),
                      hit=hit)
    return hit
