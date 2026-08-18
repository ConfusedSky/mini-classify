"""src/render_child.py: the recv -> dispatch -> send loop, against in-memory
transports and a fake loader/renderer (the real ones need the GPU). Pins the
child's contract from interfaces.md §Render child: exactly one result per
task, always; every exception between recv and send becomes Failure;
Release is a control no-op on unknown indices (K1); the Rendered ack follows
save_renders (K6); EndOfInput flushes stdio then os._exit(0), never
returning (K2/L4)."""
import os
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from src import render_child
from src.loader import LoadedMesh
from src.messages import (ChildStages, EmbedRenderTask, EmbedViews, EndOfInput,
                          Failure, PoseRenderTask, PoseTiles, Release,
                          Rendered, RenderConfig)
from src.pose import Pose

POSE = Pose(up=(0.0, 0.0, 1.0), confidence=0.9, source="geometry", v=4)
CFG = RenderConfig(render_size=64, views=8, elevations=(20.0, -20.0),
                   save_renders_dir=None, render_format="png",
                   budget_bytes=1_000_000, collection_root=Path("/nowhere"))
BOX = o3d.geometry.TriangleMesh.create_box()
BOX.compute_vertex_normals()


class ScriptedTasks:
    """The tasks transport, pre-scripted. Refuses to run dry: a child still
    recv-ing after the script means EndOfInput did not terminate the loop."""

    def __init__(self, msgs):
        self.msgs = list(msgs)

    def recv(self, timeout=None):
        assert self.msgs, "child kept recv-ing after the script ended"
        return self.msgs.pop(0)

    def recv_nowait(self):
        return self.recv()

    def send(self, msg):
        raise AssertionError("child must not send on tasks")

    def close(self):
        pass


class RecordingResults:
    def __init__(self, log):
        self.log = log
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)
        self.log.append(("send", type(msg).__name__, getattr(msg, "index", None)))

    def recv(self, timeout=None):
        raise AssertionError("child must not recv on results")

    recv_nowait = recv

    def flush(self):
        self.log.append(("results.flush",))

    def close(self):
        pass


class FakeLoader:
    """Raises for files whose stem is 'bad' — the malformed-input path."""

    def __init__(self, log):
        self.log = log

    def get(self, file):
        self.log.append(("load", Path(file).stem))
        if Path(file).stem == "bad":
            raise ValueError("no triangles")
        return LoadedMesh(file=Path(file), mesh=BOX, nbytes=100)


class FakeRenderer:
    def __init__(self, log, resident=(), fail_tiles=False):
        self.log = log
        self._resident = set(resident)
        self.fail_tiles = fail_tiles

    def is_resident(self, index):
        return index in self._resident

    def pose_tiles(self, lm, index):
        self.log.append(("pose_tiles", index))
        if self.fail_tiles:
            raise RuntimeError("filament aborted on an empty AABB")
        self._resident.add(index)
        return [[np.zeros((4, 4, 3), np.uint8)] * 2 for _ in range(6)]

    def views(self, lm, index, up):
        self.log.append(("views", index, lm is None))
        return [np.zeros((4, 4, 3), np.uint8)] * 16

    def save_renders(self, file, images):
        self.log.append(("save", Path(file).stem))

    def release(self, index):
        self.log.append(("release", index))


def run(monkeypatch, msgs, renderer=None, loader=None, log=None):
    log = [] if log is None else log
    renderer = renderer or FakeRenderer(log)
    loader = loader or FakeLoader(log)
    monkeypatch.setattr(render_child, "Renderer", lambda cfg: renderer)
    monkeypatch.setattr(render_child, "loader", loader)
    results = RecordingResults(log)
    render_child.run_child(ScriptedTasks(msgs), results, CFG)
    return results, log


class ExitCalled(Exception):
    def __init__(self, code):
        self.code = code


@pytest.fixture
def trap_exit(monkeypatch):
    """os._exit must not kill the test process; record and raise instead."""
    log = []

    def fake_exit(code):
        log.append(("os._exit", code))
        raise ExitCalled(code)

    monkeypatch.setattr(os, "_exit", fake_exit)
    return log


# --- Ctrl-C is the parent's (the child must not die of it) ------------------

def test_the_child_ignores_sigint(monkeypatch, trap_exit):
    """A terminal delivers SIGINT to the whole foreground process group, so
    this child gets it too — and the loop's `except Exception` cannot catch a
    KeyboardInterrupt, so an un-shielded child dies mid-render. The parent then
    sees an exitcode and writes every outstanding file to the CSV as a render
    failure it never had. The parent owns the lifecycle: EndOfInput on the
    drain path, kill() on the abort path."""
    import signal
    before = signal.getsignal(signal.SIGINT)
    try:
        with pytest.raises(ExitCalled):
            run(monkeypatch, [EndOfInput()])
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, before)      # never leave it ignored


# --- one result per task, always (§P2.3) ------------------------------------

def test_one_result_per_task_types_and_order(monkeypatch, trap_exit):
    f = Path("/c/a.stl")
    log = []
    results = RecordingResults(log)
    renderer = FakeRenderer(log)
    monkeypatch.setattr(render_child, "Renderer", lambda cfg: renderer)
    monkeypatch.setattr(render_child, "loader", FakeLoader(log))
    msgs = [None,
            PoseRenderTask(f, 0),
            EmbedRenderTask(f, 1, POSE, needs_embed=True),
            EmbedRenderTask(f, 2, POSE, needs_embed=False),
            EndOfInput()]
    with pytest.raises(ExitCalled):
        render_child.run_child(ScriptedTasks(msgs), results, CFG)
    assert [type(m) for m in results.sent] == [PoseTiles, EmbedViews, Rendered]
    assert [m.index for m in results.sent] == [0, 1, 2]
    assert len(results.sent) == 3            # three tasks, three results


def test_pose_task_carries_geometry_evidence_and_the_grid(monkeypatch, trap_exit):
    """The mesh never crosses the boundary, so its geometry evidence must:
    the real pose.up_axis_scores runs on the loaded mesh."""
    log = []
    results = RecordingResults(log)
    monkeypatch.setattr(render_child, "Renderer",
                        lambda cfg: FakeRenderer(log))
    monkeypatch.setattr(render_child, "loader", FakeLoader(log))
    with pytest.raises(ExitCalled):
        render_child.run_child(
            ScriptedTasks([PoseRenderTask(Path("/c/a.stl"), 7), EndOfInput()]),
            results, CFG)
    (tiles,) = results.sent
    assert isinstance(tiles, PoseTiles)
    assert isinstance(tiles.geo_scores, np.ndarray)
    assert tiles.geo_scores.shape == (6,)    # one score per UP_CANDIDATE
    assert len(tiles.tiles) == 6             # [candidate][azimuth]


# --- Failure conversion ------------------------------------------------------

def test_a_bad_mesh_becomes_failure_and_the_run_continues(monkeypatch, trap_exit):
    log = []
    results = RecordingResults(log)
    monkeypatch.setattr(render_child, "Renderer",
                        lambda cfg: FakeRenderer(log))
    monkeypatch.setattr(render_child, "loader", FakeLoader(log))
    msgs = [PoseRenderTask(Path("/c/bad.stl"), 0),
            PoseRenderTask(Path("/c/good.stl"), 1),
            EndOfInput()]
    with pytest.raises(ExitCalled):
        render_child.run_child(ScriptedTasks(msgs), results, CFG)
    failure, tiles = results.sent
    assert isinstance(failure, Failure)
    assert (failure.file, failure.index) == (Path("/c/bad.stl"), 0)
    assert failure.error == "no triangles"   # str(e), the message alone
    assert isinstance(tiles, PoseTiles) and tiles.index == 1


def test_a_renderer_exception_becomes_failure_too(monkeypatch, trap_exit):
    log = []
    results = RecordingResults(log)
    monkeypatch.setattr(render_child, "Renderer",
                        lambda cfg: FakeRenderer(log, fail_tiles=True))
    monkeypatch.setattr(render_child, "loader", FakeLoader(log))
    with pytest.raises(ExitCalled):
        render_child.run_child(
            ScriptedTasks([PoseRenderTask(Path("/c/a.stl"), 4), EndOfInput()]),
            results, CFG)
    (failure,) = results.sent
    assert isinstance(failure, Failure) and failure.index == 4
    assert failure.error == "filament aborted on an empty AABB"


def test_an_unknown_message_is_a_protocol_crash(monkeypatch):
    """Per-file errors become Failure; protocol errors are bugs and crash."""
    with pytest.raises(TypeError):
        run(monkeypatch, ["garbage"])


# --- Release: control, not a task (K1) ---------------------------------------

def test_release_produces_no_result_and_unknown_index_is_a_noop(monkeypatch, trap_exit):
    log = []
    results = RecordingResults(log)
    renderer = FakeRenderer(log)
    monkeypatch.setattr(render_child, "Renderer", lambda cfg: renderer)
    monkeypatch.setattr(render_child, "loader", FakeLoader(log))
    msgs = [Release(Path("/c/a.stl"), 5),
            Release(Path("/c/a.stl"), 999),  # unknown: no-op, no crash
            EndOfInput()]
    with pytest.raises(ExitCalled):
        render_child.run_child(ScriptedTasks(msgs), results, CFG)
    assert results.sent == []                # no result for control messages
    assert ("release", 5) in log and ("release", 999) in log


# --- ack ordering (K6) and the residency skip --------------------------------

def test_rendered_ack_is_sent_strictly_after_save_renders(monkeypatch, trap_exit):
    log = []
    results = RecordingResults(log)
    monkeypatch.setattr(render_child, "Renderer",
                        lambda cfg: FakeRenderer(log))
    monkeypatch.setattr(render_child, "loader", FakeLoader(log))
    with pytest.raises(ExitCalled):
        render_child.run_child(
            ScriptedTasks([EmbedRenderTask(Path("/c/a.stl"), 2, POSE,
                                           needs_embed=False),
                           EndOfInput()]),
            results, CFG)
    ops = [e for e in log if e[0] in ("views", "save", "send")]
    assert [e[0] for e in ops] == ["views", "save", "send"]
    assert ops[-1][1] == "Rendered"


def test_embed_views_also_saves_before_sending(monkeypatch, trap_exit):
    """The child owns saving renders in every case (data_structures Q2)."""
    log = []
    results = RecordingResults(log)
    monkeypatch.setattr(render_child, "Renderer",
                        lambda cfg: FakeRenderer(log))
    monkeypatch.setattr(render_child, "loader", FakeLoader(log))
    with pytest.raises(ExitCalled):
        render_child.run_child(
            ScriptedTasks([EmbedRenderTask(Path("/c/a.stl"), 3, POSE,
                                           needs_embed=True),
                           EndOfInput()]),
            results, CFG)
    ops = [e[0] for e in log if e[0] in ("views", "save", "send")]
    assert ops == ["views", "save", "send"]
    assert isinstance(results.sent[0], EmbedViews)


def test_a_resident_mesh_skips_the_loader(monkeypatch, trap_exit):
    log = []
    results = RecordingResults(log)
    renderer = FakeRenderer(log, resident={4})
    monkeypatch.setattr(render_child, "Renderer", lambda cfg: renderer)
    monkeypatch.setattr(render_child, "loader", FakeLoader(log))
    with pytest.raises(ExitCalled):
        render_child.run_child(
            ScriptedTasks([EmbedRenderTask(Path("/c/a.stl"), 4, POSE,
                                           needs_embed=True),
                           EndOfInput()]),
            results, CFG)
    assert not any(e[0] == "load" for e in log)   # the residency win
    assert ("views", 4, True) in log              # lm was None


# --- EndOfInput (K2/L4) -------------------------------------------------------

class FlushProbe:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def flush(self):
        self.log.append(("flush", self.name))

    def write(self, s):                      # in case anything prints
        pass


def test_end_of_input_flushes_stdio_then_exits_and_never_returns(monkeypatch):
    log = []

    def fake_exit(code):
        log.append(("os._exit", code))
        raise ExitCalled(code)

    monkeypatch.setattr(os, "_exit", fake_exit)
    monkeypatch.setattr(sys, "stdout", FlushProbe("stdout", log))
    monkeypatch.setattr(sys, "stderr", FlushProbe("stderr", log))
    results = RecordingResults(log)
    monkeypatch.setattr(render_child, "Renderer",
                        lambda cfg: FakeRenderer(log))
    monkeypatch.setattr(render_child, "loader", FakeLoader(log))
    with pytest.raises(ExitCalled) as exc:
        render_child.run_child(ScriptedTasks([EndOfInput()]), results, CFG)
    assert exc.value.code == 0
    # both flushes happen, and strictly before os._exit — os._exit skips
    # buffered stdio on a pipe (L4)
    assert log == [("flush", "stdout"), ("flush", "stderr"), ("os._exit", 0)]


# --- the instrument reply (F-7) ----------------------------------------------

def test_no_stage_totals_are_sent_when_the_run_is_not_instrumented(monkeypatch,
                                                                   trap_exit):
    """Silence is the contract off the flag: the parent waits for this message
    only under --instrument, so a child that sent one anyway would leave it on
    the queue, and one that stayed silent under the flag would cost the parent
    STAGES_S of waiting for nothing."""
    log = []
    results = RecordingResults(log)
    monkeypatch.setattr(render_child, "Renderer", lambda cfg: FakeRenderer(log))
    monkeypatch.setattr(render_child, "loader", FakeLoader(log))
    with pytest.raises(ExitCalled):
        render_child.run_child(ScriptedTasks([EndOfInput()]), results, CFG)
    assert results.sent == []
    assert ("results.flush",) not in log


def test_the_config_is_what_turns_child_timing_on(monkeypatch, trap_exit):
    """`--instrument` reaches the child only through RenderConfig — the flag
    itself is parsed in a process this one never sees. It enables timing and
    *not* sampling: one nvidia-smi per run belongs to the parent."""
    calls = []
    monkeypatch.setattr(render_child.instrument, "enable",
                        lambda path, **kw: calls.append((path, kw)))
    log = []
    monkeypatch.setattr(render_child, "Renderer", lambda cfg: FakeRenderer(log))
    monkeypatch.setattr(render_child, "loader", FakeLoader(log))
    with pytest.raises(ExitCalled):
        render_child.run_child(ScriptedTasks([EndOfInput()]),
                               RecordingResults(log), CFG)
    assert calls == []                            # instrument_path=None
    with pytest.raises(ExitCalled):
        render_child.run_child(
            ScriptedTasks([EndOfInput()]), RecordingResults(log),
            replace(CFG, instrument_path="run.json"))
    assert calls == [("run.json", {"sample": False})]


def test_stage_totals_go_home_flushed_before_the_exit(monkeypatch, trap_exit):
    """Under --instrument the child times its own stages and ships the totals
    as the last thing it does — and flushes the queue first, because os._exit
    drops the feeder's buffer exactly the way it drops stdio's (F-7)."""
    monkeypatch.setattr(render_child.instrument, "enabled", lambda: True)
    monkeypatch.setattr(render_child.instrument, "stage_totals",
                        lambda: (("main", "view-render", 1.5, 2),))
    log = []
    results = RecordingResults(log)
    monkeypatch.setattr(render_child, "Renderer", lambda cfg: FakeRenderer(log))
    monkeypatch.setattr(render_child, "loader", FakeLoader(log))
    with pytest.raises(ExitCalled):
        render_child.run_child(ScriptedTasks([EndOfInput()]), results, CFG)
    (stats,) = results.sent
    assert isinstance(stats, ChildStages)
    assert stats.rows == (("main", "view-render", 1.5, 2),)
    assert [e[0] for e in log if e[0] in ("send", "results.flush")] \
        == ["send", "results.flush"]
    assert trap_exit == [("os._exit", 0)]        # and then, only then, the exit


def test_the_child_times_the_stages_the_parent_cannot_see(monkeypatch, trap_exit):
    """The flag's promise: mesh-load/pose-geometry/pose-render on the pose
    task, mesh-load/view-render/save-renders on the embed task. These are the
    stages that moved into another process with the refactor, and nothing in
    the parent can time them."""
    timed = []
    monkeypatch.setattr(render_child, "stage",
                        lambda name: (timed.append(name), nullcontext())[1])
    log = []
    results = RecordingResults(log)
    monkeypatch.setattr(render_child, "Renderer", lambda cfg: FakeRenderer(log))
    monkeypatch.setattr(render_child, "loader", FakeLoader(log))
    f = Path("/c/a.stl")
    with pytest.raises(ExitCalled):
        render_child.run_child(
            ScriptedTasks([PoseRenderTask(f, 0),
                           EmbedRenderTask(f, 1, POSE, needs_embed=True),
                           EndOfInput()]),
            results, CFG)
    assert timed == ["mesh-load", "pose-geometry", "pose-render",
                     "mesh-load", "view-render", "save-renders"]
