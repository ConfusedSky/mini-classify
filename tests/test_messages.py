"""src/messages.py: frozenness, pickleability of everything that crosses a
queue (or the spawn boundary), and the I8 rule — importing the module must
not pull torch into the process, because the render child unpickles its
tasks from it."""
import argparse
import dataclasses
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src import messages as M
from src.pose import Pose

REPO = Path(__file__).resolve().parent.parent


def pose():
    return Pose(up=(0.0, 0.0, 1.0), confidence=0.83, source="geometry",
                v=4, margin=0.51, front_view={"8v20,-20": 3})


def tile():
    return np.zeros((4, 4, 3), dtype=np.uint8)


# Every message type that crosses the tasks/results queues, plus RenderConfig,
# which crosses the spawn boundary (I13). TileEmbeds/Embedded are parent-only
# (torch tensors, never pickled) and are deliberately absent — they get
# frozenness coverage via parent_only() instead.
def queue_crossers():
    return [
        M.PoseRenderTask(file=Path("a.stl"), index=0),
        M.EmbedRenderTask(file=Path("a.stl"), index=1, pose=pose(),
                          needs_embed=False),
        M.Release(file=Path("a.stl"), index=2),
        M.EndOfInput(),
        M.PoseTiles(file=Path("a.stl"), index=3,
                    geo_scores=np.arange(6, dtype=float),
                    tiles=[[tile(), tile()] for _ in range(6)]),
        M.EmbedViews(file=Path("a.stl"), index=4, pose=pose(),
                     views=[tile() for _ in range(16)]),
        M.Rendered(file=Path("a.stl"), index=5),
        M.Failure(file=Path("a.stl"), index=6, error="boom"),
        M.RenderConfig(render_size=384, views=8, elevations=(20.0, -20.0),
                       save_renders_dir=None, render_format="jpg",
                       budget_bytes=1 << 28, collection_root=Path("/stl")),
    ]


def parent_only():
    task = M.EmbedRenderTask(file=Path("a.stl"), index=7, pose=pose(),
                             needs_embed=True)
    hit = M.CachedHit(file=Path("a.stl"), index=7, pose=pose(),
                      cache_file=Path("a.npy"), retires=False)
    return [
        M.EmbedTilesRequest(file=Path("a.stl"), index=8,
                            tiles=np.zeros((12, 4, 4, 3), dtype=np.uint8)),
        # torch.Tensor is annotation-only in src.messages (TYPE_CHECKING, I8):
        # any object stands in for the tensor at runtime, so these construct
        # without torch — asserted below
        M.TileEmbeds(file=Path("a.stl"), index=8, embeds=object()),
        M.Embedded(file=Path("a.stl"), index=8, pose=pose(), embeds=object()),
        hit,
        M.Retired(file=Path("a.stl"), index=9),
        # driver-side: the Poser hands it straight to the driver, which
        # re-routes it through route() — it never goes near a queue
        M.Resolved(file=Path("a.stl"), index=9, pose_changed=True),
        M.Redraw(task=task, hit=hit),
        M.ResultRow(index=10, file="a.stl", up="0,0,1", pose_conf=0.83,
                    pose_source="geometry", front_view=3,
                    top=(("dragon", 0.31), ("knight", 0.22))),
    ]


def eq(a, b):
    """Field-wise equality that tolerates numpy members (dataclass __eq__
    on an ndarray field yields an ambiguous array truth value)."""
    assert type(a) is type(b)
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray):
            assert np.array_equal(x, y)
        elif isinstance(x, list) and x and isinstance(x[0], (list, np.ndarray)):
            assert np.array_equal(np.asarray(x), np.asarray(y))
        elif dataclasses.is_dataclass(x) and not isinstance(x, type):
            eq(x, y)
        else:
            assert x == y


def all_messages():
    return queue_crossers() + parent_only()


@pytest.mark.parametrize("msg", all_messages(), ids=lambda m: type(m).__name__)
def test_every_message_is_frozen(msg):
    field = dataclasses.fields(msg)[0].name if dataclasses.fields(msg) else None
    if field is None:                      # EndOfInput has no fields; setattr
        field = "anything"                 # on a frozen class still raises
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(msg, field, "nope")


def test_pose_is_frozen_and_unhashable():
    p = pose()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.source = "vlm"
    with pytest.raises(TypeError):         # shallow freeze: front_view is a
        hash(p)                            # dict, so Pose must never be a key
    hash(M.Rendered(file=Path("a.stl"), index=0))   # dict-free messages hash


@pytest.mark.parametrize("msg", queue_crossers(), ids=lambda m: type(m).__name__)
def test_queue_crossers_pickle_round_trip(msg):
    eq(msg, pickle.loads(pickle.dumps(msg)))


def test_importing_messages_does_not_import_torch():
    """I8: a fresh interpreter that imports src.messages must not have torch
    in sys.modules — a real torch import would hand the render child a torch
    dependency when it unpickles tasks. Constructing the two tensor-carrying
    types must not pull it in either: the annotation is TYPE_CHECKING-only."""
    code = ("import sys; from pathlib import Path; import src.messages as M; "
            "from src.pose import Pose; "
            "p = Pose(up=(0.0, 0.0, 1.0), confidence=0.5, source='geometry', v=4); "
            "M.TileEmbeds(file=Path('a.stl'), index=0, embeds=object()); "
            "M.Embedded(file=Path('a.stl'), index=0, pose=p, embeds=object()); "
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr


def test_resolved_requires_its_pose_changed_verdict():
    """No default: the driver hands this straight to route(), and a silent
    False is a fresh override whose stale renders never get redrawn."""
    with pytest.raises(TypeError):
        M.Resolved(file=Path("a.stl"), index=0)


def test_cache_context_is_deliberately_mutable():
    """data_structures.md's driver-side block marks CacheContext @dataclass,
    NOT frozen — it bundles live references (the pose store Done owns), not a
    message. Pin that so nobody 'fixes' it to match its frozen neighbours."""
    ctx = M.CacheContext(poses={}, embeds_dir=None, render_index={},
                         args=argparse.Namespace(), root=Path("/stl"))
    ctx.embeds_dir = Path("embeds")        # must not raise
    assert [f.name for f in dataclasses.fields(ctx)] == \
        ["poses", "embeds_dir", "render_index", "args", "root"]


def test_failure_row_matches_todays_error_shape():
    f = M.Failure(file=Path("bad.stl"), index=0, error="no header")
    assert f.to_csv() == {"file": "bad.stl", "top1": "RENDER_ERROR: no header"}


def test_result_row_matches_todays_columns():
    row = M.ResultRow(index=0, file="a.stl", up="0,0,1", pose_conf=0.83,
                      pose_source="geometry", front_view=3,
                      top=(("dragon", 0.31), ("knight", 0.22), ("orc", 0.11)))
    assert row.to_csv() == {"file": "a.stl", "up": "0,0,1", "pose_conf": 0.83,
                            "pose_source": "geometry", "front_view": 3,
                            "top1": "dragon", "score1": 0.31,
                            "top2": "knight", "score2": 0.22,
                            "top3": "orc", "score3": 0.11}
