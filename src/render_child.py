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

Import rule (interfaces.md row 1): child side imports open3d/PIL/numpy/
messages/pose — never torch.
"""
import os
import sys

from src import loader
from src import pose
from src.messages import (EmbedRenderTask, EmbedViews, EndOfInput, Failure,
                          PoseRenderTask, PoseTiles, Release, Rendered,
                          RenderConfig)
from src.renderer import Renderer
from src.transport import Transport

# recv wake-up period. Not a liveness mechanism — EndOfInput is what ends the
# loop — just a bound on how long the child sleeps in one recv call.
RECV_TIMEOUT_S = 1.0


def _handle(msg, renderer: Renderer):
    """One task -> its one result. Raises on failure; the loop converts."""
    if isinstance(msg, PoseRenderTask):
        lm = loader.get(msg.file)
        geo_scores = pose.up_axis_scores(lm.mesh)   # the mesh never crosses the
        tiles = renderer.pose_tiles(lm, msg.index)  # boundary, so its geometry
        return PoseTiles(msg.file, msg.index, geo_scores, tiles)  # evidence must
    # EmbedRenderTask. The pose->embed revisit is the residency win: a
    # resident mesh needs no re-parse, so loader.get runs only on a miss.
    lm = None if renderer.is_resident(msg.index) else loader.get(msg.file)
    views = renderer.views(lm, msg.index, msg.pose.up)  # up rotated into a copy
                                                        # of the mesh (I11)
    renderer.save_renders(msg.file, views)              # child owns saving (Q2)
    if not msg.needs_embed:
        return Rendered(msg.file, msg.index)            # ack after save (K6)
    return EmbedViews(msg.file, msg.index, msg.pose, views)


def run_child(tasks: Transport, results: Transport, cfg: RenderConfig) -> None:
    """Spawned once per run (spawn context, daemon — the parent's job). Loops
    until `EndOfInput`; never returns."""
    renderer = Renderer(cfg)
    while True:
        msg = tasks.recv(timeout=RECV_TIMEOUT_S)
        if msg is None:                         # nothing arrived, not the end:
            continue                            # EndOfInput is a message (I5)
        if isinstance(msg, EndOfInput):
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
