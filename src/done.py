"""Done — scoring, rows, the pose store, retirement, Release, flush
(interfaces.md §"Done — the only writer, and the owner of retirement").

Extracted from the scoring/writing half of `classify_stls.process()` and the
`finally` epilogue: the cache load (classify_stls.py:1179-1182), the torn-write
guard on the `.npy` save (:1187-1195), the score block (:1197-1217), the CSV
fields and writer (:1261-1266), and the pose-cache save (:1255-1256) — which is
where the still-open `save_pose_cache` atomicity fix (temp + `os.replace`)
finally lands (J7, data_structures.md's last paragraph).

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
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.cache_checker import cache_key_from_identity
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

# Today's CSV columns, verbatim (classify_stls.py:1261-1262). `index` orders
# the flush; it is not a column.
CSV_FIELDS = ["file", "top1", "score1", "top2", "score2", "top3", "score3",
              "up", "pose_conf", "pose_source", "front_view"]


def view_config(args):
    """The token keying `front_view` entries in the pose cache. Extracted
    verbatim from classify_stls.view_config (:300-308); the parity test in
    tests/test_done.py pins the two identical until classify_stls delegates."""
    elev = ",".join(f"{e:g}" for e in args.elevations)
    return f"{args.views}v-e{elev}"


def pool_sims(view_sims, mode, axis=-2):
    """Pool per-view similarity scores (..., n_views, n_categories) over views.
    Extracted verbatim from classify_stls.pool_sims (:553-566); parity-pinned
    by tests/test_done.py until classify_stls delegates to this copy."""
    if mode == "mean":
        return view_sims.mean(axis)
    if mode == "max":
        return view_sims.max(axis)
    BETA = 50.0
    w = np.exp(BETA * (view_sims - view_sims.max(axis, keepdims=True)))
    return (w * view_sims).sum(axis) / w.sum(axis)


class Done:
    """The terminal stage: every admitted index ends here, exactly once."""

    def __init__(self, admission: "Admission", text_embeds, cache_ctx: CacheContext,
                 tasks: Transport, *, categories=None, front_embeds=None,
                 back_embeds=None):
        """The four interfaces.md parameters, plus the scoring assets the
        Embedder computes at startup (None under --skip-embed, where no row
        is ever scored): `categories` names the top-3 columns, and
        `front_embeds`/`back_embeds` are the numpy prompt banks
        `front_view_index` ranks against (classify_stls.py:1040-1041)."""
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
        which Done derives itself (J6: the Poser has no root)."""
        self.poses[file_identity(file, self.ctx.root)] = pose.to_cache()

    # --- The one entry point -------------------------------------------------

    def on(self, m: CachedHit | Embedded | Failure | Retired | Rendered) -> None:
        match m:
            case CachedHit():
                # Score before retiring: a corrupt .npy raises out of here and
                # the drain arm converts it to a Failure, which retires (I4).
                img_embeds = torch.from_numpy(np.load(m.cache_file)).to(
                    self.text_embeds.device, dtype=self.text_embeds.dtype
                )  # classify_stls.py:1181
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

    # --- Scoring (classify_stls.py:1197-1217, extracted faithfully) ----------

    def _score(self, file: Path, index: int, pose: Pose, img_embeds) -> ResultRow:
        view_np = img_embeds.float().cpu().numpy()
        # A forced --up-axis never consults the store (E-R1-1, parity with
        # classify_stls.py:1100-1102): the run's up is the flag, so a warm
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
                # (D9), exactly today's shape (:1203-1206); a legacy int
                # carries no config record and is replaced.
                old = entry.get("front_view")
                entry["front_view"] = \
                    {**(old if isinstance(old, dict) else {}), self.view_cfg: fv}
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

    # --- Fresh-embedding cache write (classify_stls.py:1187-1195) ------------

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
        (src/pose.py:200-205); then the rows CSV, partial on abort.

        Each write is attempted even if the other's raises
        (classify_stls.py:1249-1268, the full-disk incident): the nested
        finallys chain rather than swallow, so the last failure re-raises
        after every write has had its try, with the earlier failure kept
        visible as `__context__`. A failed `os.replace` leaves no `.tmp`
        behind."""
        args = self.ctx.args
        try:
            if args.up_axis == "auto" and args.cache_dir:  # classify_stls.py:1255
                p = Path(args.cache_dir) / "pose-cache.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_name(p.name + ".tmp")
                try:
                    tmp.write_text(json.dumps(self.poses))   # byte-parity with
                    os.replace(tmp, p)                       # save_pose_cache
                finally:
                    # a successful replace already consumed it; a failed one
                    # would otherwise strand a half-written cache next to the
                    # real file
                    tmp.unlink(missing_ok=True)
        finally:
            with open(args.out, "w", newline="") as fh:      # :1263-1266
                writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(self.rows[i].to_csv() for i in sorted(self.rows))
