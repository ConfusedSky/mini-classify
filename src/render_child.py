"""The render child's entry point: one recv -> dispatch -> send loop
(docs/actor-refactor/interfaces.md §Render child).

The contract, verbatim from the interfaces note:

* **Exactly one result per task, always** (§P2.3): `PoseTiles`, `EmbedViews`,
  `Rendered`, or `Failure`. A task that raises sends `Failure` *instead of*
  its result — the exception never crosses raw.
* `Release` and `EndOfInput` are control messages, not tasks — no result.
  `Release` clears a resident mesh's `in_flight` flag; unknown or cleared
  indices are a no-op (K1).
* The `Rendered` ack is sent strictly **after** `save_renders` returns (K6):
  quiescence-means-idle and the parent's untimed join both rest on the ack
  being last.
* `EndOfInput` flushes stdout/stderr and exits via `os._exit(0)`, never by
  returning (K2/L4): interpreter teardown would destroy the
  `OffscreenRenderer` — the one hard-constraint abort (CLAUDE.md). The stdio
  flush matters because `os._exit` skips it and the child's diagnostics are
  block-buffered on a pipe. Skipping the queue feeder's delivery guarantee is
  safe only because `EndOfInput` follows quiescence (I1) — every result is
  already received by then.
* The child never crashes on a per-file error; it crashes only on protocol
  errors (which are bugs) — an unknown message type raises out of the loop.

`--instrument` is where the child's stage timings come from (F-7): every stage
the old single-process pipeline attributed below the recv — mesh-load,
pose-geometry, pose-render, view-render, save-renders — is timed here, in the
process that actually does the work, and the totals ride back on the
`EndOfInput` reply so the parent prints one table. The timing lives in this
file rather than in `renderer.py`/`loader.py` so those stay free of everything
but their job, and `instrument` is torch-free, which is what lets the child
import it at all.

Import rule (interfaces.md row 1): child side imports open3d/PIL/numpy/
messages/pose — never torch.
"""
import os
import signal
import sys

from src import instrument
from src.instrument import stage
from src import loader
from src import pose
from src.messages import (ChildStages, EmbedRenderTask, EmbedViews, EndOfInput,
                          Failure, PoseRenderTask, PoseTiles, Release,
                          Rendered, RenderConfig)
from src.renderer import Renderer
from src.transport import Transport

# recv wake-up period. Not a liveness mechanism — EndOfInput is what ends the
# loop — just a bound on how long the child sleeps in one recv call.
RECV_TIMEOUT_S = 1.0


def _handle(msg, renderer: Renderer):
    """One task -> its one result. Raises on failure; the loop converts."""
    if isinstance(msg, PoseRenderTask):
        with stage("mesh-load"):
            lm = loader.get(msg.file)
        with stage("pose-geometry"):
            geo_scores = pose.up_axis_scores(lm.mesh)  # the mesh never crosses
        with stage("pose-render"):                     # the boundary, so its
            tiles = renderer.pose_tiles(lm, msg.index)  # geometry evidence must
        return PoseTiles(msg.file, msg.index, geo_scores, tiles)
    # EmbedRenderTask. The pose->embed revisit is the residency win: a
    # resident mesh needs no re-parse, so loader.get runs only on a miss.
    lm = None
    if not renderer.is_resident(msg.index):
        with stage("mesh-load"):                # the residency hit shows up as
            lm = loader.get(msg.file)           # a missing call, not a fast one
    with stage("view-render"):
        views = renderer.views(lm, msg.index, msg.pose.up)  # up rotated into a
                                                            # copy of the mesh
    with stage("save-renders"):                             # (I11)
        renderer.save_renders(msg.file, views)  # child owns saving (Q2)
    if not msg.needs_embed:
        return Rendered(msg.file, msg.index)            # ack after save (K6)
    return EmbedViews(msg.file, msg.index, msg.pose, views)


def run_child(tasks: Transport, results: Transport, cfg: RenderConfig) -> None:
    """Spawned once per run (spawn context, daemon — the parent's job). Loops
    until `EndOfInput`; never returns."""
    # Ctrl-C is the parent's to handle. A terminal delivers SIGINT to the whole
    # foreground process group, which includes this child, and the loop's
    # `except Exception` cannot catch the resulting KeyboardInterrupt (it is a
    # BaseException) — so an un-shielded child dies mid-render, the parent's
    # liveness check sees an exitcode, and every outstanding file is written to
    # the CSV as a render failure it never had. Measured: an unshielded spawn
    # child raises KeyboardInterrupt and exits 1; SIG_IGN leaves it untouched.
    # The parent still owns the lifecycle — EndOfInput on the drain path,
    # kill() on the abort path — and this is a daemon, so a hard second Ctrl-C
    # still takes it down with the parent.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if cfg.instrument_path:
        # times stages, samples nothing: one nvidia-smi per run is the parent's
        # (instrument.py), and the child's totals go home on EndOfInput
        instrument.enable(cfg.instrument_path, sample=False)
    renderer = Renderer(cfg)
    while True:
        msg = tasks.recv(timeout=RECV_TIMEOUT_S)
        if msg is None:                         # nothing arrived, not the end:
            continue                            # EndOfInput is a message (I5)
        if isinstance(msg, EndOfInput):
            if instrument.enabled():            # the last message of the run,
                results.send(ChildStages(instrument.stage_totals()))
                results.flush()                 # and os._exit would drop it in
                                                # the feeder (F-7) — the same
                                                # loss the stdio flush below
                                                # prevents
            sys.stdout.flush()                  # os._exit skips buffered stdio
            sys.stderr.flush()                  # on a pipe (L4)
            os._exit(0)                         # never return: teardown would
                                                # destroy the renderer (K2)
        if isinstance(msg, Release):
            renderer.release(msg.index)         # control: no result (K1)
            continue
        if not isinstance(msg, (PoseRenderTask, EmbedRenderTask)):
            raise TypeError(f"unknown message on tasks: {type(msg).__name__}")
        try:                                    # every exception between recv
            out = _handle(msg, renderer)        # and send becomes Failure —
        except Exception as e:                  # one bad mesh must not end
            out = Failure(msg.file, msg.index, str(e))  # the run
        results.send(out)                       # exactly one result per task
