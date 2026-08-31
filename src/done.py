"""Done — scoring, rows, the pose store, retirement, Release, flush
(interfaces.md §"Done — the only writer, and the owner of retirement").

The scoring and writing half of the pipeline: the cache load, the torn-write
guard on the `.npy` save, the score block, the CSV fields and writer, and the
pose-cache save — which is where the `save_pose_cache` atomicity fix (temp +
`os.replace`) landed, closing the last of the three defects the proposal
listed (J7; `pose.save_pose_cache` keeps its bare `write_text` for the evals,
and nothing in the pipeline calls it).

Ownership (I9/I10):
* the canonical pose store — THE dict `CacheContext.poses` aliases, written
  only here (`record_pose` is the Poser's sole write API; `Done` derives
  `file_identity` itself, J6) and via the score block's `front_view` merge;
* `Admission.retired` — bumped once per index; `retired_ids` ignores repeats
  (J2), so a double retirement cannot drive `in_flight()` negative;
* `rows` — the output record, not the retirement record: `Retired` and
  `Rendered` retire without a row, a `CachedHit(retires=False)` writes its row
  without retiring, and a later `Failure` overwrites a hit's row (K5).

`Done` holds the `tasks` transport for exactly one thing: an unconditional
`Release(file, index)` on every retirement (K1) — the second parent-side
writer on `tasks` (L3). The child no-ops on cleared/unknown indices, and FIFO
on `tasks` means a `Release` can never overtake the task it follows.

`flush` is idempotent and runs on the main thread on both the drain and abort
paths — twice on abort, on both sides of `poser.settle` (interfaces.md
§Shutdown): pose cache first (the artifact whose loss costs money), then the
partial rows CSV. No narration here; the wind-down is the driver's to tell.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.instrument import stage
# `view_config` keys the front_view entries this module writes, but it is a
# piece of cache identity rather than scoring — its home is the module that
# owns the cache layout, so the CLI and the read-only tools can name a view
# config without importing a module that loads torch.
from src.cachedir import view_config, write_atomic
from src.identity import cache_key_from_identity
from src.messages import (
    CacheContext,
    CachedHit,
    Embedded,
    Failure,
    Release,
    Rendered,
    ResultRow,
    Retired,
)
# The view pooling this module's score block runs, which is a query decision
# and not a writer's: it moved to `src/query.py` when the REPL and the API
# needed it without importing the pipeline's terminal stage to get it.
from src.query import pool_sims
from src.pose import (
    FORCED_UPS,
    Pose,
    file_identity,
    front_view,
    front_view_index,
    up_str,
)
from src.transport import Transport

if TYPE_CHECKING:
    from src.driver import Admission

# `index` orders the flush; it is not a column.
CSV_FIELDS = ["file", "top1", "score1", "top2", "score2", "top3", "score3",
              "up", "pose_conf", "pose_source", "front_view"]


class Done:
    """The terminal stage: every admitted index ends here, exactly once."""

    def __init__(self, admission: "Admission", text_embeds, cache_ctx: CacheContext,
                 tasks: Transport, *, categories=None, front_embeds=None,
                 back_embeds=None):
        """The four interfaces.md parameters, plus the scoring assets the
        Embedder computes at startup (None under --skip-embed, where no row
        is ever scored): `categories` names the top-3 columns, and
        `front_embeds`/`back_embeds` are the numpy prompt banks
        `front_view_index` ranks against (`src/embedder.py`'s `front_T`/
        `back_T`, built once at startup)."""
        self.admission = admission
        self.text_embeds = text_embeds          # torch, read-only after startup
        self.ctx = cache_ctx
        self.tasks = tasks                      # for Release on retirement (K1)
        self.categories = categories
        self.front_embeds = front_embeds
        self.back_embeds = back_embeds
        # THE canonical store (I9): the same object CacheContext.poses holds,
        # so route() sees this run's resolutions. file_identity -> JSON entry
        # dict, exactly as load_pose_cache returns and save_pose_cache expects.
        self.poses: dict = cache_ctx.poses
        self.rows: dict[int, ResultRow | Failure] = {}
        self.retired_ids: set[int] = set()      # J2: repeats are no-ops
        self.view_cfg = view_config(cache_ctx.args)

    # --- The pose store's only write API (I9) --------------------------------

    def record_pose(self, file: Path, index: int, pose: Pose) -> None:
        """The Poser's write path — a Pose, never a dict reference. `index` is
        the caller's identity for the file; the store keys on file_identity,
        which Done derives itself (J6: the Poser has no root).

        **A record that claims no judgment does not overwrite one, and a
        rejection never replaces an answer.** The whole rule, by
        (stored, incoming) — `judged` is `True` or `"rejected"`:

            stored          incoming            outcome
            ------          --------            -------
            unjudged        anything            replaces
            judged          False / absent      REFUSED
            True            "rejected"          REFUSED  (2026-08-31, pass 3)
            "rejected"      "rejected"          replaces (re-judged refusal)
            "rejected"      True                replaces
            True            True                replaces

        The first refusal: `False` or absent says "asked and not answered
        *yet*", or "never asked at all" — neither is a verdict, and a judgment
        is replaced only by another judgment.

        The second is a design ruling (adversarial review pass 3,
        2026-08-31). `"rejected"` is a verdict, but the verdict is "this judge
        will not answer" — worth recording exactly where there is no answer to
        keep, and nowhere else. Under `--repose` an incoming rejection over a
        stored `True` trades a paid, *answered* judgment (whose `up` the old
        judge may well have MOVED) for a refusal: the same trade the guard
        above exists to refuse, differing only in that the thing arriving is
        typed as settled. The cost is symmetric and bounded — a `--repose` run
        under a rejecting judge re-opens that entry on every run, exactly as a
        same-judge rejection already stays put, and `--repose` runs are
        deliberate.

        The guard lives here rather than in the Poser because Done owns the
        store and the Poser may not read it (J6) — which is precisely why the
        Poser cannot see that it is about to park over a judgment. The owner
        is the only actor in a position to refuse.

        What it prevents (adversarial review, 2026-08-31): `--repose` re-opens
        a settled entry *before* it has secured the replacement. The Poser
        records the ensemble's answer at park time — `arbitrated=False`, no
        arbiter, `poser.on_tile_embeds` — and that record used to replace the
        stored judgment wholesale. So anything going wrong after the park
        downgraded the paid judgment to the very answer the old judge had
        overruled: a `VLMUnavailable`, a 429, a Ctrl-C, the breaker tripping,
        each of which records `arbitrated=False` and used to land. Worse, a
        fresh margin that cleared this run's gate made *no call at all* and
        wrote `arbitrated=None` over the judgment — erased permanently, with
        nothing bought in exchange. `--repose` re-judges a judgment; it must
        never spend one.

        Outside `--repose` the refused case is unreachable for every shape the
        loader admits, so this changes no existing behaviour: a judged entry
        that gets past `pose.load_pose_cache` carries a margin, and a judged
        entry with a margin is sufficient (`pose.pose_is_sufficient`), so
        `route` never re-opens it, so the Poser never resolves that file and
        never writes over it. Truthy `arbitrated` is not on its own enough —
        sufficiency's `arbitrated` branch answers `margin is not None`, and a
        judgment recording no margin would be insufficient every run *and*
        unwritable through this guard, re-rendering forever. That deadlock is
        why the loader drops the shape at load rather than passing it here.
        Pinned by test, since "unreachable" is a claim about another module
        and nothing here can enforce it.

        The in-run consequence of holding the old entry — nothing inconsistent,
        and worth stating because the run has already re-rendered by then.
        **Safe because `settled` skips only the sufficiency check: `route`
        re-reads the store either way** (`src/cache_checker.py`, the
        `settled or pose_is_sufficient(...)` short-circuit sits *after* the
        `ctx.poses.get`). So the re-route sees the **preserved** entry and the
        rest of the run runs on the judged pose: the views are rendered under
        its up, the `.npy` is keyed on its up, the CSV row reports it, and the
        `_score` front_view merge below writes an index computed from renders
        of that same judged pose back into the judged entry — where it belongs.
        The fresh ensemble answer is discarded where it was made and never
        reaches this stage, so there is no orphan embedding either. A failed
        re-judgment therefore behaves as if the re-open never happened; the
        entry is re-opened again by the next `--repose` run, which is the
        intended retry. The cost is one wasted pose-tile render and ensemble
        pass, plus — under `--save-renders`, and only when the discarded
        answer's source is `siglip` or `vlm` — a redraw of already-correct
        views: `pose_changed` is `source in poser.MOVED_SOURCES`, so a
        geometry-sourced discard forces no redraw at all.

        A future refactor of `route` that "optimizes away" that second store
        read is what breaks this; `tests/test_cache_checker.py` pins it."""
        ident = file_identity(file, self.ctx.root)
        stored = self.poses.get(ident)
        was = stored.get("arbitrated") if stored is not None else None
        # The two refusals of the table above, in the order they were found:
        # nothing over a judgment, and a refusal over an *answer*.
        if was in (True, "rejected") and not pose.arbitrated:
            return
        if was is True and pose.arbitrated == "rejected":
            return
        self.poses[ident] = pose.to_cache()

    # --- The one entry point -------------------------------------------------

    def on(self, m: CachedHit | Embedded | Failure | Retired | Rendered) -> None:
        match m:
            case CachedHit():
                # Score before retiring: a corrupt .npy raises out of here and
                # the drain arm converts it to a Failure, which retires (I4).
                with stage("cache-load"):
                    img_embeds = torch.from_numpy(np.load(m.cache_file)).to(
                        self.text_embeds.device, dtype=self.text_embeds.dtype
                    )
                self.rows[m.index] = self._score(m.file, m.index, m.pose, img_embeds)
                if m.retires:
                    self._retire(m.file, m.index)
                # retires=False: the redraw path — the row comes from here,
                # retirement from the child's Rendered ack (§P2.3).
            case Embedded():
                self._save_embeds(m)            # cache-save first, as today
                self.rows[m.index] = self._score(m.file, m.index, m.pose, m.embeds)
                self._retire(m.file, m.index)
            case Failure():
                self.rows[m.index] = m          # K5: overwrites a hit's row
                self._retire(m.file, m.index)
            case Retired() | Rendered():        # no row (I3/Q2, §P2.3)
                self._retire(m.file, m.index)
            case _:
                raise TypeError(f"Done.on: unexpected message {m!r}")

    # --- Retirement (I10, J2, K1) --------------------------------------------

    def _retire(self, file: Path, index: int) -> None:
        if index in self.retired_ids:           # J2: idempotent — no second
            return                              # count, no second Release
        self.retired_ids.add(index)
        self.admission.retired += 1
        self.tasks.send(Release(file=file, index=index))    # K1: unconditional

    # --- Scoring ------------------------------------------------------------

    def _score(self, file: Path, index: int, pose: Pose, img_embeds) -> ResultRow:
        with stage("score"):        # front_view resolution is inside the stage
            view_np = img_embeds.float().cpu().numpy()
            # A forced --up-axis never consults the store (E-R1-1, the same
            # rule `cache_checker.route` applies): the run's up is the flag, so a warm
            # auto entry's front_view describes a *different* pose's renders —
            # reading it would report the wrong hero view, and merging into it
            # would poison the auto entry with an index resolved under the
            # override. Forced fv is recomputed per run and never persisted.
            entry = None
            if self.ctx.args.up_axis not in FORCED_UPS:
                entry = self.poses.get(file_identity(file, self.ctx.root))
            fv = front_view(entry, self.view_cfg)
            if fv is None:
                fv = front_view_index(view_np, self.front_embeds, self.back_embeds)
                if entry is not None:
                    # The front_view merge writes through the canonical entry
                    # (D9): other configs' indices stay, this run's is added.
                    entry["front_view"] = {**entry.get("front_view", {}),
                                           self.view_cfg: fv}
            view_sims = (img_embeds @ self.text_embeds.T).float().cpu().numpy()
            sims = torch.from_numpy(pool_sims(view_sims, self.ctx.args.pool))
            order = sims.argsort(descending=True)
            top = []
            for rank in range(min(3, len(self.categories))):
                idx = order[rank]
                top.append((self.categories[idx], round(sims[idx].item(), 4)))
            return ResultRow(index=index, file=str(file), up=up_str(pose.up),
                             pose_conf=pose.confidence, pose_source=pose.source,
                             front_view=fv, top=tuple(top))

    # --- Fresh-embedding cache write -----------------------------------------

    def _save_embeds(self, m: Embedded) -> None:
        if self.ctx.embeds_dir is None:
            return
        ident = file_identity(m.file, self.ctx.root)
        # The token comes from the pose these embeddings were rendered under,
        # not from a store round-trip (E-R1-1): only `up` changes the pixels
        # (embed_cache_token), and `m.pose.up` is authoritative. On the auto
        # path this is exactly what the store holds — record_pose wrote
        # `pose.to_cache()`, which round-trips `up` unchanged — while on the
        # forced path it is the flag's up, where `route` looks
        # (embed_cache_token(None, axis) == up_str(FORCED_UPS[axis])) instead
        # of a stale auto entry's.
        token = up_str(m.pose.up)
        cache_file = Path(self.ctx.embeds_dir) / \
            f"{cache_key_from_identity(ident, self.ctx.args, token)}.npy"
        try:
            with stage("cache-save"):
                np.save(cache_file, m.embeds.float().cpu().numpy())
        except BaseException:
            # a torn write (full disk, Ctrl-C) would read as a cache hit next
            # run and crash cache-load (:1191-1195)
            cache_file.unlink(missing_ok=True)
            raise

    # --- Flush: pose cache first (temp+replace), then rows CSV ---------------

    def flush(self) -> None:
        """Idempotent, main-thread only; both writes are full rewrites, so a
        second call replays them byte-identically. Pose cache first — the
        only artifact whose loss costs money (§Shutdown) — via temp +
        os.replace, the atomicity fix save_pose_cache never got
        (`pose.save_pose_cache`); then the rows CSV, partial on abort.

        Each write is attempted even if the other's raises
        (the full-disk incident): the nested
        finallys chain rather than swallow, so the last failure re-raises
        after every write has had its try, with the earlier failure kept
        visible as `__context__`. A failed `os.replace` leaves no `.tmp`
        behind."""
        args = self.ctx.args
        try:
            if args.up_axis == "auto" and args.cache_dir:
                p = Path(args.cache_dir) / "pose-cache.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                # cachedir.write_atomic: unique temp name + finally-unlink,
                # the one copy of the idiom (review, 2026-08-20); the JSON
                # itself keeps byte-parity with save_pose_cache
                write_atomic(p, json.dumps(self.poses))
        finally:
            with open(args.out, "w", newline="") as fh:      # :1263-1266
                writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(self.rows[i].to_csv() for i in sorted(self.rows))
