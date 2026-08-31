"""The Cache Checker: `route()`, the admission decision (interfaces.md
§"Cache Checker — a pure decision").

The whole cache decision: the pose-cache lookup, the embedding-cache
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

Import discipline: no torch, no renderer, no top-level script. Cache keying
goes through `src.identity` / `src.pose` —
`pose.file_identity` is byte-identical to the embedding key's identity half
(review §P3.1), so this module never re-derives an identity string, and the
key builders themselves are `identity`'s: the child's renderer needs
`render_key` too and neither module may import the other (E-R1-5).
"""
from __future__ import annotations

from pathlib import Path

from src import pose
from src.identity import cache_key_from_identity, render_key
from src.messages import (
    CacheContext,
    CachedHit,
    EmbedRenderTask,
    PoseRenderTask,
    Redraw,
    Retired,
)

REPOSE_ARBITER_ATTR = "repose_arbiter"
"""The namespace attribute `--repose` crosses on, owned here because `route`
is the reader.

`classify_stls.main` sets it, `route` reads it back off `ctx.args`, and
nothing else carries the flag — `CacheContext.args` is the only thing that
crosses. Before this was a shared constant the name was spelled out on both
sides, and a rename on either one degraded silently into "the flag does
nothing", which is indistinguishable from a cache holding no foreign
judgments. Naming it once makes a rename break an import instead (adversarial
review, 2026-08-31)."""

# --- The decision ------------------------------------------------------------

def route(f: Path, index: int, ctx: CacheContext, pose_changed: bool = False,
          settled: bool = False, *, arbiter_available: bool) \
        -> PoseRenderTask | EmbedRenderTask | CachedHit | Redraw | Retired:
    """One file's cache decision. Raises on I/O errors (a vanished file, an
    unreadable directory) — the driver converts to `Failure` (J3).

    `route` runs twice for a file that needed a fresh pose: cold (→
    `PoseRenderTask`), then again on the Poser's `Resolved` with the pose
    store warm. `pose_changed` is the driver's one extra input on that second
    call — true when the fresh source is `vlm`/`siglip` — and only the
    renders-wanted arm reads it (interfaces.md §Cache Checker).

    `settled` marks that second call: the pose this run could decide is
    decided, so the sufficiency re-check is skipped. Without it the check
    turned the tri-state's `arbitrated: false` — written by `_fold` for a
    rate-limited, transiently failed or cancelled call moments before the
    re-route — back into a `PoseRenderTask`, and the run re-rendered,
    re-escalated and re-billed the same file in an unbounded loop, each lap
    bumping the stall clock (review, 2026-08-20). `false` means "ask again
    on a later run"; this flag is what confines it to one.

    `PoseRenderTask.force_escalate` is the one thing this decision carries
    *forward* rather than returning: a miss caused by `--repose` re-opening a
    settled judgment must reach the arbiter whatever the fresh ensemble margin
    turns out to be, and `route` is the only actor that can tell that kind of
    miss from an ordinary one (the Poser holds no store — J6). It rides the
    task to the child and back on `PoseTiles`, because the Poser is handed
    tiles, never tasks.

    `arbiter_available` — the driver's `cfg.poser.can_arbitrate()`, and
    **keyword-only with no default** so every caller breaks loudly
    (docs/archive/tri-state-pass-2.md, 2026-08-21). It is what makes a marked entry a
    miss only in a run that can actually escalate it: without it an
    arbiterless run re-rendered the marked model, re-resolved it with no gate
    and erased the marker — and production runs `--pose-vlm off`. At the
    `settled=True` site the parameter is dead by construction (`settled or
    ...` short-circuits before the sufficiency check), so nobody should read
    the breaker as able to flip a settled re-route into a re-render."""
    args = ctx.args

    # Pose: forced axis skips the pose-cache lookup entirely. Otherwise a miss — or a
    # geometry-only pose this run's ensemble can upgrade — is a PoseRenderTask.
    entry = None
    if args.up_axis in pose.FORCED_UPS:
        resolved = pose.Pose(up=pose.FORCED_UPS[args.up_axis], confidence=0.0,
                             source="forced", v=pose.POSE_CACHE_VERSION)
    else:
        entry = ctx.poses.get(pose.file_identity(f, ctx.root))
        # The ensemble always runs now (`--no-up-ensemble`/`--up-conf` retired
        # 2026-08-17, actors_proposal.md Migration notes), so a geometry-only
        # entry — margin None, written by some older pass — always reads
        # insufficient and is upgraded in place. The availability flag
        # `pose_is_sufficient` takes is a different one: the *arbiter's*, plus
        # this run's gate (docs/archive/tri-state-pass-2.md, 2026-08-21).
        # `repose_arbiter` via getattr: it is `main`'s, set on the namespace
        # there, and every other namespace that reaches here — the tools', the
        # tests' — predates the flag and means "off". The name is
        # REPOSE_ARBITER_ATTR on both sides so it cannot drift.
        repose_arbiter = getattr(args, REPOSE_ARBITER_ATTR, None)
        if entry is None or not (settled or pose.pose_is_sufficient(
                entry, arbiter_available, args.up_margin,
                repose_arbiter=repose_arbiter)):
            # Why the entry is a miss, not just that it is. Only `route` knows
            # — the Poser holds no store (J6) and sees only a fresh ensemble
            # margin — so a re-opened judgment whose fresh margin clears the
            # gate would make no call, record `arbitrated=None`, be refused by
            # `Done.record_pose`'s guard, and be re-opened again by the next
            # `--repose` run: re-rendered forever, never re-judged, silently.
            # `pose.repose_reopens` is the same predicate sufficiency used one
            # line above, so the two cannot disagree about which misses these
            # are; every ordinary miss (no entry, geometry-only, an owed
            # escalation) leaves it False and buys no call it was not owed.
            return PoseRenderTask(
                file=f, index=index,
                force_escalate=pose.repose_reopens(entry, arbiter_available,
                                                   repose_arbiter))
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

    # Renders: `pose_changed or not renders_ok`. Saved renders predate a fresh
    # override, so they show the old pose; the embedding re-keys on its own
    # because the override moves up_token, but the debug files do not.
    renders_wanted = bool(args.save_renders) and bool(args.cache_dir)
    need_renders = False
    if renders_wanted:
        rkey = render_key(f, ctx.root)
        n_views = args.views * len(args.elevations)
        need_renders = pose_changed or not all(
            f"{rkey}_view{i}" in ctx.render_index for i in range(n_views))

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
