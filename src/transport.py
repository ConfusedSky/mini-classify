"""The boundary protocol (docs/actor-refactor/interfaces.md): the one
interface between the parent and the render child, small enough that the
measured shared-memory variant can drop in behind the same four signatures.

v1 wiring: `tasks` (parent→child) unbounded, `results` (child→parent)
bounded at the admission window (I2/Q1). Admission is the only forward
pressure — the parent never blocks on a send.
"""
import multiprocessing as mp
import queue as _queue
from typing import Any, Protocol


class Transport(Protocol):
    def send(self, msg) -> None:        # blocks only if bounded and full
        ...

    def recv_nowait(self):              # a message, or None if empty
        ...

    def recv(self, timeout=None):       # a message; None = nothing arrived
        ...

    def close(self) -> None:            # drop: cancel_join_thread + close —
        ...                             # unflushed pickles are abandoned, so an
                                        # aborting parent cannot be held open by
                                        # the queue's feeder thread (I6)


class MpQueueTransport:
    """`mp.Queue` behind the protocol. `maxsize=0` is unbounded.

    Built on a spawn context by default — spawn is load-bearing for the
    render child with CUDA initialised in the parent (interfaces.md, child
    lifecycle). The instance is picklable across that spawn boundary the
    same way a bare `mp.Queue` is: pass it in the child's `Process` args.
    """

    def __init__(self, maxsize: int = 0, ctx=None):
        self._q = (ctx or mp.get_context("spawn")).Queue(maxsize)

    def send(self, msg: Any) -> None:
        self._q.put(msg)

    def recv_nowait(self):
        try:
            return self._q.get_nowait()
        except _queue.Empty:
            return None

    def recv(self, timeout=None):
        try:
            return self._q.get(timeout=timeout)
        except _queue.Empty:
            return None

    def close(self) -> None:
        self._q.cancel_join_thread()
        self._q.close()
