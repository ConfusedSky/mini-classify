"""The actor refactor's frozen message types — one dataclass per edge, no
`kind` field; routing is `match`/`isinstance` (docs/actor-refactor/
data_structures.md, calling conventions in interfaces.md).

`index` is the file's position in the Walker's list: what `Done` sorts by and
what the Supervisor counts — the identity, everywhere.

torch appears under TYPE_CHECKING only (interfaces review I8): the render
child unpickles its tasks from this module, and a real `import torch` here
would hand the child exactly the dependency the import-rule table forbids.
The two tensor-typed messages (`TileEmbeds`, `Embedded`) never cross a queue,
so the name is annotation-only — kept lazy by `from __future__ import
annotations`.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.pose import Pose

if TYPE_CHECKING:
    import torch


# --- Parent → child ---------------------------------------------------------

@dataclass(frozen=True)
class PoseRenderTask:              # pose unknown → render candidate tiles
    file: Path
    index: int


@dataclass(frozen=True)
class EmbedRenderTask:             # pose resolved → render classification views
    file: Path
    index: int
    pose: Pose
    needs_embed: bool              # False: the redrawn path (D8) — embedding
                                   # cached but a saved render is missing/stale;
                                   # the child renders, saves, returns nothing
                                   # but its Rendered ack


@dataclass(frozen=True)
class Release:                     # control message (not a task — no result):
    file: Path                     # clears a resident mesh's in_flight flag.
    index: int                     # Done sends one per retirement,
                                   # unconditionally; the child no-ops on
                                   # cleared or unknown indices (K1)


@dataclass(frozen=True)
class EndOfInput:                  # terminates the child. A message, not None:
    pass                           # recv's None means "nothing arrived yet"
                                   # (interfaces review I5). The child then
                                   # flushes stdio and exits via os._exit(0) —
                                   # interpreter teardown would destroy the
                                   # renderer, the one hard-constraint abort
                                   # (K2/L4)


# --- Child → parent ---------------------------------------------------------

@dataclass(frozen=True)
class PoseTiles:                   # → Poser
    file: Path
    index: int
    geo_scores: np.ndarray         # up_axis_scores from the child's mesh: the
                                   # mesh never crosses the boundary, so its
                                   # geometry evidence must
    tiles: list[list[np.ndarray]]  # [candidate][azimuth] — the grid, not a
                                   # flat list (D7)


@dataclass(frozen=True)
class EmbedViews:                  # → Embedder
    file: Path
    index: int
    pose: Pose                     # read-only echo for the row's pose columns;
                                   # Done writes front_view through the
                                   # canonical pose dict, not this copy (D9)
    views: list[np.ndarray]


@dataclass(frozen=True)
class Rendered:                    # → Done: the needs_embed=False ack. The
    file: Path                     # child always sends exactly one result per
    index: int                     # task (interfaces §P2.3) — this retires a
                                   # render-only file (J1/J2/J4)


@dataclass(frozen=True)
class ChildStages:                 # → the parent's instrument, once, in reply
    rows: tuple                    # to EndOfInput and only under --instrument
                                   # (F-7). NOT a task result: it carries no
                                   # file/index because it belongs to no file,
                                   # and it is sent after quiescence, so the
                                   # drain has already stopped reading results
                                   # by the time it exists. `rows` is
                                   # instrument.stage_totals()' plain tuples —
                                   # the child must not pickle anything the
                                   # parent would have to import to unpickle


# --- Poser ↔ Embedder (the ensemble) — D5 -----------------------------------

@dataclass(frozen=True)
class EmbedTilesRequest:           # Poser → Embedder
    file: Path
    index: int
    tiles: np.ndarray              # stacked, order-preserving; the Poser keeps
                                   # the [candidate][azimuth] grouping


@dataclass(frozen=True)
class TileEmbeds:                  # Embedder → Poser (back-edge; parent-only,
    file: Path                     # never pickled)
    index: int
    embeds: torch.Tensor           # on device; the Poser pulls it off the GPU


# --- Into Done — D5 ---------------------------------------------------------

@dataclass(frozen=True)
class CachedHit:                   # Cache Checker → Done: embedding cache hit
    file: Path
    index: int
    pose: Pose
    cache_file: Path               # Done loads the .npy (today's cache-load)
    retires: bool = True           # False on the redraw path: the row comes
                                   # from here, retirement from the child's
                                   # Rendered ack (§P2.3)


@dataclass(frozen=True)
class Embedded:                    # Embedder → Done: fresh embeddings to score
    file: Path                     # (parent-only, never pickled)
    index: int
    pose: Pose
    embeds: torch.Tensor           # stays on device: Done's scoring matmul
                                   # against the text embeddings runs on the GPU


@dataclass(frozen=True)
class Failure:                     # any stage → Done; becomes a RENDER_ERROR row
    file: Path
    index: int
    error: str

    def to_csv(self) -> dict:
        # Parity with main's error rows (main:classify_stls.py:1127, :1169):
        # DictWriter fills the missing columns.
        return {"file": str(self.file), "top1": f"RENDER_ERROR: {self.error}"}


@dataclass(frozen=True)
class Retired:                     # → Done: retire with no row (interfaces
    file: Path                     # review I3/Q2) — the --skip-embed paths,
    index: int                     # where pose resolution was the whole job


# --- Driver-side shapes (never cross a queue) -------------------------------

@dataclass(frozen=True)
class Resolved:                    # Poser → driver: this file's pose is settled
    file: Path                     # and recorded. The driver re-routes it
    index: int                     # through route(f, index, pose_changed=...) —
                                   # the pose store is warm by then, so the
                                   # warm-.npy and redraw arms apply (the
                                   # second-call rule, interfaces §route). The
                                   # Poser decides poses, never cache admission
    pose_changed: bool             # true when the fresh source is vlm/siglip
                                   # (`src/poser.py`'s MOVED_SOURCES). The Poser knows
                                   # the source it just recorded, so it rides
                                   # here and the driver passes it straight to
                                   # route rather than re-deriving it from the
                                   # store. No default: guessing it False is a
                                   # silently un-redrawn override


@dataclass(frozen=True)
class Redraw:                      # route's redraw return: both halves of the
    task: EmbedRenderTask          # decision in one value, so a test of route
    hit: CachedHit                 # covers what the driver dispatches (I14)


@dataclass(frozen=True)
class RenderConfig:                # handed whole to the child at spawn — it
    render_size: int               # crosses the spawn boundary, so everything
    views: int                     # here must stay picklable (I13)
    elevations: tuple[float, ...]
    save_renders_dir: Path | None
    render_format: str
    budget_bytes: int
    collection_root: Path
    instrument_path: str | None = None   # --instrument's path, or None. A str,
                                         # not a Path or a live handle: the
                                         # child re-derives its own timing from
                                         # it (F-7) and everything in here has
                                         # to survive the pickle (I13)


@dataclass                         # NOT frozen — live references, not a message
class CacheContext:                # route()'s read-only world: the pose store
    poses: dict                    # (THE object Done owns, not a copy — route
    embeds_dir: Path | None        # must see this run's resolutions), the
    render_index: dict             # render index, and the parsed args the
    args: argparse.Namespace       # cache keys derive from
    root: Path                     # the collection anchor — Done derives
                                   # file_identity from it, so the Poser
                                   # never has to (J6). The ONLY sanctioned
                                   # parent-side root: never re-derive it
                                   # from args (K7). RenderConfig carries
                                   # its own copy because it crosses spawn.


# --- Rows -------------------------------------------------------------------

@dataclass(frozen=True)
class ResultRow:
    index: int
    file: str
    up: str
    pose_conf: float
    pose_source: str
    front_view: int
    top: tuple[tuple[str, float], ...]     # up to 3 of (category, score)

    def to_csv(self) -> dict:
        # Main's CSV columns (main:classify_stls.py:1261-1262); `index` orders the
        # flush, it is not a column.
        d = {"file": self.file, "up": self.up, "pose_conf": self.pose_conf,
             "pose_source": self.pose_source, "front_view": self.front_view}
        for rank, (category, score) in enumerate(self.top[:3], start=1):
            d[f"top{rank}"] = category
            d[f"score{rank}"] = score
        return d
