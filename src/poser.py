"""The Poser — ensemble + continuations + arbiter calls (docs/actor-refactor/
interfaces.md §Poser, actors_proposal.md §Poser).

Never blocks, returns what to do next: `on_tiles` stashes the child's
evidence and asks the Embedder for tile embeddings; `on_tile_embeds` runs the
ensemble (the math lives in src/pose — `upright_scores`, `rank_up_scores`,
`combine_up` — never reimplemented here) and either hands back `Resolved` or
parks the file on an arbiter `Future`; `poll` resumes parked files each driver
iteration. `fold_done`/`settle` are the abort pair that owns emptying
`parked`, and `drop` is the driver's Failure arm forgetting a file.

**The Poser decides poses, never cache admission.** Every answer leaves here
as `Resolved(file, index, pose_changed)` and the *driver* re-routes it
through `cache_checker.route` — the second-call rule (interfaces §route):
the pose store is warm by then, so the warm-`.npy` shortcut and the redraw
arm apply instead of an unconditional re-embed. `pose_changed` rides along
because the Poser is the one that knows which tier moved the answer. That re-route is main's
post-resolution check (main:classify_stls.py:1148-1155), whose loss was the
regression escalated against the first draft of this module; a `_next_step`
here building the task itself is exactly what reintroduces it.

The Poser consumes geometry *scores* (computed child-side) and tiles, never
the mesh; it holds no root and derives no identities — resolved poses go out
through `record_pose`, Done's write API (I9/J6). The one torch touch is
pulling `TileEmbeds.embeds` off the GPU (`.float().cpu().numpy()`), which
needs no torch import. The contact sheet's PIL conversion happens here
(`Image.fromarray` over the grid's first column) because arrays are what
cross the process boundary.
"""
from __future__ import annotations

import sys
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Callable

import numpy as np
from PIL import Image

from src import pose
from src.messages import EmbedTilesRequest, Failure, PoseTiles, Resolved, \
    TileEmbeds
from src.pose import Pose

VLM_BACKENDS = (None, "gemini", "claude")
"""What `--pose-vlm` may be. `ollama` is retired at construction (C-R1-4):
the Arbiter has no inline arm, and a pooled ollama call would overlap SigLIP
on the 4060 — 10.1 s of model reload against 0.49 s of inference (CLAUDE.md's
hard constraint). A serialized-inline ollama mode can return as its own
design item; failing here is what keeps it from arriving as a silent
eight-way regression."""

MOVED_SOURCES = ("vlm", "siglip")
"""Sources that make `Resolved.pose_changed` true — parity with
`main:classify_stls.py:1146`. Saved renders predate a fresh override, so they show
the old pose and `route`'s renders-wanted arm must force the redraw; the
*embedding* re-keys on its own, because the override moves `up_token`. Read
off the source of the pose actually recorded, never off which method built
the `Resolved`: a cancelled or failed fold keeps the park-time pose, and its
source is whatever the ensemble wrote — often `siglip`, which still needs the
redraw."""


@dataclass
class ParkedFile:
    """Continuation state (data_structures.md §Poser continuation state),
    replacing today's `deferred` list + single-slot `pending_box`. The
    `Future` IS the Arbiter → Poser transport (Q1)."""
    file: Path
    resolved: tuple                # (up, ratio, source, margin) — ensemble's answer
    future: Future


@dataclass
class VlmConfig:
    """Everything the Poser needs to build one arbiter call — and nothing
    more. The call construction came from the `ask_vlm_up` closure inside
    the CLI's old `resolve_up` (deleted 2026-08-18; this is the only
    arrangement of it now). No run-mode flags: `skip_embed`/`save_renders`
    lived here only to pick EmbedRenderTask-vs-Retired, and that decision is
    `route`'s again (the second-call rule), so the Poser no longer needs to
    know what mode the run is in."""
    backend: str | None = None     # None: never escalate — ambiguity keeps
                                   # the ensemble's pose, as today with no VLM
    model: str | None = None       # backend default applies when None
    scratch_dir: str | Path = "."  # the claude backend's sheet drop
    project: str | None = None     # --gemini-project
    margin_threshold: float = pose.MARGIN_THRESHOLD
    sheet_path: Callable[[Path], Path | None] | None = None
                                   # per-file save_to for the contact sheet
                                   # (today: rdir / f"{rkey}_pose.png") — the
                                   # Poser derives no identities, so the path
                                   # rule is injected
    ask: Callable[[list], int | None] | None = None
                                   # test seam: called with the sheet tiles in
                                   # place of pose.ask_vlm_up; the real network
                                   # call is the None default

    def __post_init__(self):
        if self.backend not in VLM_BACKENDS:
            raise ValueError(
                f"unsupported arbiter backend {self.backend!r}; "
                f"expected one of {VLM_BACKENDS}")


class Poser:
    def __init__(self, up_T: np.ndarray, down_T: np.ndarray, arbiter,
                 record_pose: Callable[[Path, int, Pose], None],
                 vlm_cfg: VlmConfig):
        self.up_T = up_T
        self.down_T = down_T
        self.arbiter = arbiter
        self.record_pose = record_pose
        self.cfg = vlm_cfg
        # continuation state, written only here — the abort pair owns its
        # emptying — and READ by the driver (P4): quiescence and the M4/N1
        # subtractions need membership, hence an exposed dict, not a predicate
        self.parked: dict[int, ParkedFile] = {}
        self._stash: dict[int, tuple[np.ndarray, list[list[np.ndarray]]]] = {}

    # --- the ensemble ------------------------------------------------------

    def on_tiles(self, m: PoseTiles) -> EmbedTilesRequest:
        """Stash (geo_scores, tiles-grid) keyed by index and return the embed
        request — the ensemble cannot finish without the Embedder."""
        self._stash[m.index] = (m.geo_scores, m.tiles)
        flat = [im for row in m.tiles for im in row]   # candidate-major, the
        return EmbedTilesRequest(file=m.file, index=m.index,   # grid's order
                                 tiles=np.stack(flat))

    def on_tile_embeds(self, m: TileEmbeds) -> Resolved | None:
        """Run the ensemble and return `Resolved` for the driver to re-route,
        or None having parked the file on a submitted arbiter Future.

        The ensemble is the CLI's old `resolve_up`'s, and now the only one
        (that single-process arrangement was deleted 2026-08-18), with the
        geometry half already computed child-side: rank geo_scores, average
        SigLIP's upright margin per candidate, combine, and record which tier
        *moved* the answer. The resolved pose is recorded through record_pose
        BEFORE any park — that recording is settle's abandonment floor (I15)."""
        geo_scores, grid = self._stash.pop(m.index)
        embeds = m.embeds.float().cpu().numpy()        # the one GPU pull
        sig = pose.upright_scores(embeds, self.up_T, self.down_T) \
                  .reshape(len(grid), -1).mean(axis=1)
        geo_idx, ratio, _best = pose.rank_up_scores(geo_scores)
        idx, margin = pose.combine_up(geo_scores, sig)
        source = "geometry" if idx == geo_idx else "siglip"
        up = tuple(float(v) for v in pose.UP_CANDIDATES[idx])
        resolved = (up, ratio, source, margin)
        p = self._make_pose(*resolved)
        self.record_pose(m.file, m.index, p)

        if self.cfg.backend and pose.needs_arbiter_margin(
                margin, self.cfg.margin_threshold):
            sheet_tiles = [Image.fromarray(row[0]) for row in grid]
            future = self.arbiter.submit(self._vlm_call(m.file, sheet_tiles))
            self.parked[m.index] = ParkedFile(m.file, resolved, future)
            return None
        return self._resolved(m.file, m.index, source)

    # --- resuming parked files --------------------------------------------

    def poll(self) -> list[Resolved | Failure]:
        """Fold resolved futures, called every driver iteration. Each resumed
        file yields its `Resolved`, re-routed by the driver like any other.
        Its own error boundary (J3): a fold that raises yields Failure for
        that file rather than ending the run, because a raise inside poll
        cannot be attributed by the driver. Failed *calls* keep the ensemble's
        pose and resume normally, as does a cancelled one — the park-time
        answer stands (re-recorded with `arbitrated=False` when the failure
        was transient or the call cancelled), so its `pose_changed` comes
        from the source recorded at park."""
        out = []
        for index in [i for i, pf in self.parked.items() if pf.future.done()]:
            pf = self.parked.pop(index)
            try:
                p = self._fold(index, pf)
                source = pf.resolved[2] if p is None else p.source
                out.append(self._resolved(pf.file, index, source))
            except Exception as e:
                out.append(Failure(pf.file, index, str(e)))
        return out

    def drop(self, index: int) -> None:
        """Forget a file the driver has failed (C-R1-5) — its Failure arm
        calls this, and ONLY that arm: fail_outstanding must never drop a
        parked file, whose in-flight answer is already paid for and must
        still fold before flush (N3, C-R2-2).
        Pops the stash and the park, cancelling a queued future
        so no worker buys an answer nobody will read; a running call is not
        cancellable and its result is simply never folded. A no-op on an
        unknown index, so the driver can call it unconditionally — the same
        reason `Release` no-ops child-side (K1)."""
        self._stash.pop(index, None)
        pf = self.parked.pop(index, None)
        if pf is not None:
            pf.future.cancel()

    def fold_done(self) -> int:
        """Abort step 1: fold every ALREADY-resolved future — no wait, so no
        exposure. What it leaves in `parked` is the wait's size. Cancelled
        futures (arbiter.shutdown ran first) are dropped uncounted: queued,
        never billed; their files keep the pose recorded at park. Inherits
        poll's error boundary — a raising fold costs that one file's answer,
        never the flush behind it."""
        folded = 0
        for index in [i for i, pf in self.parked.items() if pf.future.done()]:
            pf = self.parked.pop(index)
            try:
                folded += self._fold(index, pf) is not None
            except Exception as e:
                print(f"  fold failed for {pf.file.stem}: {e}", file=sys.stderr)
        return folded

    def settle(self, timeout: float) -> int:
        """Abort step 2: wait out the in-flight calls and fold them (I15) —
        free in wall-clock, since the pool's atexit join blocks on the same
        threads regardless. Returns how many were abandoned to their
        ensemble pose: the abort's closing line, when non-zero. Cancelled
        futures are skipped, not abandoned — they were queued, never billed.

        An abandoned call is re-recorded `arbitrated=False` before it is left
        behind: it was asked and the answer never arrived, exactly the state
        the tri-state's `false` names. Without the record the park-time pose
        stood with the key absent — on disk indistinguishable from "never
        asked", so a Ctrl-C landing on up to --arbiter-workers in-flight
        calls silently pinned each one forever, the same hole the
        CancelledError branch in `_fold` closes for the queued ones
        (review, 2026-08-20)."""
        deadline = monotonic() + timeout
        abandoned = 0
        for index in list(self.parked):
            pf = self.parked.pop(index)
            if not pf.future.done():
                try:
                    pf.future.result(timeout=max(0.0, deadline - monotonic()))
                except CancelledError:
                    pass                     # _fold skips it below
                except Exception:
                    if not pf.future.done():  # the wait ran out, not the call
                        abandoned += 1        # keeps the park-time pose (I15)
                        up, ratio, source, margin = pf.resolved
                        self.record_pose(pf.file, index,
                                         self._make_pose(up, ratio, source,
                                                         margin,
                                                         arbitrated=False))
                        continue
            try:
                self._fold(index, pf)
            except Exception as e:
                print(f"  fold failed for {pf.file.stem}: {e}", file=sys.stderr)
        return abandoned

    # --- internals ---------------------------------------------------------

    def _fold(self, index: int, pf: ParkedFile) -> Pose | None:
        """Fold one done future: apply the arbiter's answer to the pose
        resolved without it (apply_arbiter, main:classify_stls.py:508-512 —
        retired with the deferral it closed) and record the result. A failed
        call keeps the ensemble's answer (main:classify_stls.py:1233-1235).
        May raise; the caller is the boundary.

        Four outcomes, three records, and the split is the retry rule
        (`pose.pose_is_sufficient`):

        * an answer — `arbitrated=True`, settled either way;
        * a **transient failure** (`pose.VLMUnavailable`: rate limit, network,
          5xx, CLI timeout) or a **cancellation** — `arbitrated=False`, because
          both mean "asked and not answered *yet*" and a later run should ask
          again;
        * any other failure — the key is left **absent**, because a request the
          API rejects on its merits cannot succeed on a retry, and re-escalating
          it forever would pay a call per run for an answer that cannot come.

        "A later run", not this one: the driver re-routes the Resolved with
        `settled=True`, so an `arbitrated=False` record cannot re-escalate
        inside the run that just wrote it (review, 2026-08-20).

        Cancellation records and still returns None: `settle` counts folds with
        `is not None`, so returning a Pose here would inflate that count
        (review, 2026-08-19). Before this, a cancelled call re-recorded nothing
        at all — leaving the ensemble row with the key absent, a permanent hit
        indistinguishable from "never asked". `arbiter.shutdown()` cancels the
        queued futures on every Ctrl-C, so one interrupt pinned a whole queue
        of models to their ensemble answers forever."""
        arbitrated = None
        try:
            idx = pf.future.result(timeout=0)
            arbitrated = idx is not None
        except CancelledError:
            up, ratio, source, margin = pf.resolved
            self.record_pose(pf.file, index,
                             self._make_pose(up, ratio, source, margin,
                                             arbitrated=False))
            return None
        except pose.VLMUnavailable as e:     # transient: ask again next run
            print(f"  arbiter unavailable for {pf.file.stem}: {e}")
            idx, arbitrated = None, False
        except Exception as e:               # one bad call must not sink the rest
            print(f"  arbiter failed for {pf.file.stem}: {e}")
            idx, arbitrated = None, None     # not retryable; leave the key off
        up, ratio, source, margin = pf.resolved
        if idx is not None and not np.allclose(pose.UP_CANDIDATES[idx],
                                               np.asarray(up)):
            up = tuple(float(v) for v in pose.UP_CANDIDATES[idx])
            source = "vlm"                   # the arbiter MOVED the answer;
                                             # a confirmation keeps the label
        # `arbitrated` was decided above with the outcome; a confirmation and
        # a refusal are otherwise identical here, since both leave source and
        # margin untouched.
        p = self._make_pose(up, ratio, source, margin, arbitrated=arbitrated)
        self.record_pose(pf.file, index, p)
        return p

    def _resolved(self, file: Path, index: int, source: str) -> Resolved:
        # the one place the source → pose_changed mapping lives, so the
        # ensemble exit and the fold exits cannot drift apart
        return Resolved(file, index, pose_changed=source in MOVED_SOURCES)

    def _make_pose(self, up, ratio, source, margin, arbitrated=None) -> Pose:
        # main's fresh-entry shape (main:classify_stls.py:1138-1141): rounded
        # confidence/margin, explicit POSE_CACHE_VERSION (D10)
        return Pose(up=tuple(float(v) for v in up),
                    confidence=round(float(ratio), 4), source=source,
                    margin=None if margin is None else round(float(margin), 4),
                    arbitrated=arbitrated,
                    v=pose.POSE_CACHE_VERSION)

    def _vlm_call(self, file: Path, sheet_tiles: list) -> Callable[[], int | None]:
        cfg = self.cfg
        if cfg.ask is not None:
            return lambda: cfg.ask(sheet_tiles)
        save_to = cfg.sheet_path(file) if cfg.sheet_path else None
        # raise_failures: the Future carries the last attempt's exception,
        # whatever its type, so `_fold` can tell a transient refusal (retry)
        # from a request the API rejects on its merits (do not re-pay)
        return lambda: pose.ask_vlm_up(
            sheet_tiles, cfg.backend, cfg.scratch_dir,
            cfg.model or pose.DEFAULT_VLM_MODELS.get(cfg.backend),
            save_to=save_to, project=cfg.project, raise_failures=True)
