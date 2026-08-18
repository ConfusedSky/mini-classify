"""src/driver.py: the sequential loop, against fake transports, a fake child
and fake stages (interfaces.md §Driver, §Shutdown, and the invariants section).

Nothing real is constructed here — no process, no GPU, no arbiter thread. What
is pinned is the calling convention the whole refactor rests on:

* the drain's dispatch table, including the `Resolved` re-route through
  `route(..., pose_changed=...)` — the warm-`.npy`/redraw parity arm, whose
  loss was the regression escalated against wave 1;
* every drain arm being an error boundary that retires AND calls `poser.drop`
  (C-R1-5) — and `fail_outstanding` never calling `drop` (N3/C-R2-2);
* admission bounding the child's backlog, and quiescence (in_flight AND
  parked) preceding `EndOfInput`;
* the abort's fixed order with a flush on both sides of `settle`, and the
  narration that prices the second Ctrl-C;
* the dead-child and wedged-child paths, the second gated on `child_owed()`
  so an arbiter tail can never trip it;
* invariant 5, both halves (F-3): the transports' bounds, against the real
  queues, and *when* the parent is allowed to block on a recv — which needs a
  fake that tells `recv` and `recv_nowait` apart, or the flag the driver
  threads through `drain` is invisible to everything here.
"""
import signal
import threading
from pathlib import Path

import numpy as np

from src import driver
from src.driver import Admission, DriverConfig, DriverState
from src.messages import (CachedHit, CacheContext, EmbedRenderTask,
                          EmbedTilesRequest, EmbedViews, Embedded, EndOfInput,
                          Failure, PoseRenderTask, PoseTiles, Redraw, Rendered,
                          Resolved, Retired, TileEmbeds)
from src.pose import Pose

POSE = Pose(up=(0.0, 0.0, 1.0), confidence=0.9, source="geometry", v=4, margin=0.7)


def f(i):
    return Path(f"/lib/m{i}.stl")


# --- fakes -------------------------------------------------------------------

class FakeTasks:
    """The parent->child queue. Records what was sent and, with each message,
    what the accounting looked like at that moment — which is how the
    quiescence-before-EndOfInput ordering is checked without a real child."""

    def __init__(self, log):
        self.log = log
        self.sent = []
        self.at_send = []          # (in_flight, parked) per message
        self.closed = False
        self.probe = None          # set by the rig once the state exists

    def send(self, m):
        self.sent.append(m)
        self.at_send.append(self.probe() if self.probe else None)
        self.log.append(("tasks.send", type(m).__name__))

    def recv(self, timeout=None):
        raise AssertionError("the driver never receives on tasks")

    recv_nowait = recv

    def close(self):
        self.closed = True
        self.log.append(("tasks.close",))


class FakeResults:
    """The child->parent queue, and **the two recv methods are not the same
    method** (F-3). The `block` flag the driver threads through `drain` decides
    whether the parent is allowed to go to sleep holding work it could be
    dispatching (invariant 5b), and a fake where `recv is recv_nowait` cannot
    see the difference — with one, flipping the post-admit `drain(block=False)`
    to True passes every test in this file.

    So they differ the way the real transport differs:

    * `recv(timeout)` **sleeps**. Time passes and the world moves while the
      parent is in here — which is what `on_empty` models: a clock advancing, a
      child answering, a signal arriving. Every call is recorded with the
      accounting at that moment, so a test can ask *when* the parent chose to
      sleep.
    * `recv_nowait()` returns whatever has already arrived, or None, and
      changes nothing. The world does not move while the parent is walking.

    Refuses to be polled forever: a driver that cannot make progress must fail
    the test, not hang it."""

    def __init__(self, script=(), on_empty=None, limit=500):
        self.script = list(script)
        self.on_empty = on_empty
        self.empties = 0
        self.limit = limit
        self.blocking_at = []      # (in_flight, admitted) per blocking recv
        self.probe = None          # set by the rig once the state exists

    def recv(self, timeout=None):
        self.blocking_at.append(self.probe() if self.probe else None)
        if self.script:
            return self.script.pop(0)
        self.empties += 1
        assert self.empties < self.limit, "driver spun without making progress"
        if self.on_empty:          # the world moves only while the parent
            self.on_empty()        # sleeps in here
        return None

    def recv_nowait(self):
        return self.script.pop(0) if self.script else None

    def send(self, m):
        raise AssertionError("the driver never sends on results")

    def close(self):
        pass


class FakeChild:
    def __init__(self, log, exitcode=None):
        self.log = log
        self.exitcode = exitcode
        self.joins = []

    def kill(self):
        self.log.append(("child.kill",))
        self.exitcode = -9

    def join(self, timeout=None):
        self.log.append(("child.join", timeout))
        self.joins.append(timeout)
        if self.exitcode is None:
            self.exitcode = 0


class FakeDone:
    """Retirement, rows and flush — Done's contract as the driver depends on
    it: idempotent retirement (J2), `retired_ids` readable by name (M4/N1),
    and an `admission` the driver takes its one `Admission` from (P2)."""

    def __init__(self, log, admission=None):
        self.log = log
        self.admission = admission or Admission()
        self.rows = {}
        self.retired_ids = set()
        self.flushes = 0

    def on(self, m):
        self.log.append(("done.on", type(m).__name__, m.index))
        if isinstance(m, (Embedded, Failure)) or \
                (isinstance(m, CachedHit) and m.retires):
            self.rows[m.index] = m
            self._retire(m.index)
        elif isinstance(m, CachedHit):
            self.rows[m.index] = m          # retires=False: row now, ack later
        elif isinstance(m, (Retired, Rendered)):
            self._retire(m.index)
        else:
            raise TypeError(f"Done.on: unexpected {m!r}")

    def _retire(self, index):
        if index in self.retired_ids:
            return
        self.retired_ids.add(index)
        self.admission.retired += 1

    def record_pose(self, file, index, p):
        self.log.append(("record_pose", index))

    def flush(self):
        self.flushes += 1
        self.log.append(("flush",))


class FakePoser:
    """`answers` maps index -> what on_tile_embeds returns (a Resolved, or None
    for a park). `polls` is a list of lists: one drain's worth of poll output
    per entry, then empty forever."""

    def __init__(self, log, answers=None, polls=(), parked=None):
        self.log = log
        self.answers = dict(answers or {})
        self.polls = [list(p) for p in polls]
        self.parked = {} if parked is None else parked   # by reference: a test
                                                         # clears it to model a
                                                         # fold, as poll does
        self.dropped = []
        self.folded = 0
        self.abandoned = 0

    def on_tiles(self, m):
        self.log.append(("on_tiles", m.index))
        return EmbedTilesRequest(file=m.file, index=m.index,
                                 tiles=np.zeros((2, 4, 4, 3), np.uint8))

    def on_tile_embeds(self, m):
        self.log.append(("on_tile_embeds", m.index))
        return self.answers.get(m.index)

    def poll(self):
        out = self.polls.pop(0) if self.polls else []
        if out:
            self.log.append(("poll", [o.index for o in out]))
        return out

    def drop(self, index):
        self.log.append(("poser.drop", index))
        self.dropped.append(index)

    def fold_done(self):
        self.log.append(("fold_done",))
        return self.folded

    def settle(self, timeout):
        self.log.append(("settle", timeout))
        self.parked.clear()
        return self.abandoned


class FakeEmbedder:
    def __init__(self, log, fail_tiles=(), fail_views=()):
        self.log = log
        self.fail_tiles = set(fail_tiles)
        self.fail_views = set(fail_views)

    def embed_tiles(self, m):
        self.log.append(("embed_tiles", m.index))
        if m.index in self.fail_tiles:
            raise RuntimeError("CUDA out of memory")
        return TileEmbeds(file=m.file, index=m.index, embeds=object())

    def embed_views(self, m):
        self.log.append(("embed_views", m.index))
        if m.index in self.fail_views:
            raise RuntimeError("CUDA out of memory")
        return Embedded(file=m.file, index=m.index, pose=m.pose, embeds=object())


class FakeArbiter:
    def __init__(self, log):
        self.log = log

    def shutdown(self):
        self.log.append(("arbiter.shutdown",))


class Rig:
    def __init__(self, monkeypatch, files=1, routes=None, script=(),
                 answers=None, polls=(), parked=None, on_empty=None,
                 window=2, skip_embed=False, exitcode=None,
                 fail_tiles=(), fail_views=()):
        self.log = []
        self.tasks = FakeTasks(self.log)
        self.results = FakeResults(script, on_empty=on_empty)
        self.child = FakeChild(self.log, exitcode)
        self.done = FakeDone(self.log)
        self.poser = FakePoser(self.log, answers, polls, parked)
        self.embedder = FakeEmbedder(self.log, fail_tiles, fail_views)
        self.arbiter = FakeArbiter(self.log)
        self.route_calls = []
        self.routes = routes or {}
        self.tasks.probe = lambda: (self.done.admission.in_flight(),
                                    len(self.poser.parked))
        self.results.probe = lambda: (self.done.admission.in_flight(),
                                      self.done.admission.admitted)
        monkeypatch.setattr(driver, "route", self._route)
        self.cfg = DriverConfig(
            walker=[f(i) for i in range(files)],
            ctx=CacheContext(poses={}, embeds_dir=None, render_index={},
                             args=None, root=Path("/lib")),
            tasks=self.tasks, results=self.results, child=self.child,
            poser=self.poser, embedder=self.embedder, done=self.done,
            arbiter=self.arbiter, skip_embed=skip_embed, window=window)

    def _route(self, file, index, ctx, pose_changed=False):
        self.route_calls.append((index, pose_changed))
        out = self.routes.get((index, pose_changed),
                              self.routes.get(index, PoseRenderTask(file, index)))
        if isinstance(out, Exception):
            raise out
        return out

    def run(self):
        driver.run(self.cfg)
        return self.log

    def kinds(self, *names):
        return [e for e in self.log if e[0] in names]


# --- the drain's dispatch table ----------------------------------------------

def test_pose_tiles_walks_poser_embedder_poser(monkeypatch):
    """PoseTiles -> on_tiles -> embed_tiles -> on_tile_embeds, in that order,
    and the Resolved it returns re-routes into a task the driver sends."""
    rig = Rig(monkeypatch, files=1,
              routes={0: PoseRenderTask(f(0), 0),
                      (0, True): EmbedRenderTask(f(0), 0, POSE, needs_embed=True)},
              script=[PoseTiles(f(0), 0, np.zeros(6), [[np.zeros((2, 2, 3))]]),
                      Rendered(f(0), 0)],
              answers={0: Resolved(f(0), 0, pose_changed=True)})
    rig.run()
    assert [e[0] for e in rig.log if e[0] in
            ("on_tiles", "embed_tiles", "on_tile_embeds")] == \
        ["on_tiles", "embed_tiles", "on_tile_embeds"]
    assert isinstance(rig.tasks.sent[-2], EmbedRenderTask)


def test_resolved_reroutes_with_pose_changed_verbatim(monkeypatch):
    """The driver passes the Poser's flag straight through and never re-derives
    it from the store: the cold call is (0, False), the second (0, True)."""
    rig = Rig(monkeypatch, files=1,
              routes={(0, False): PoseRenderTask(f(0), 0),
                      (0, True): Retired(f(0), 0)},
              script=[PoseTiles(f(0), 0, np.zeros(6), [[np.zeros((2, 2, 3))]])],
              answers={0: Resolved(f(0), 0, pose_changed=True)})
    rig.run()
    assert rig.route_calls == [(0, False), (0, True)]
    assert ("done.on", "Retired", 0) in rig.log


def test_a_parked_file_produces_no_task_until_poll_resolves_it(monkeypatch):
    """on_tile_embeds returning None is a park: nothing dispatched, and the
    file resumes through poll — re-routed like any other Resolved (J3/O5)."""
    parked = {0: "future"}
    rig = Rig(monkeypatch, files=1,
              routes={(0, False): PoseRenderTask(f(0), 0),
                      (0, True): Retired(f(0), 0)},
              script=[PoseTiles(f(0), 0, np.zeros(6), [[np.zeros((2, 2, 3))]])],
              answers={0: None}, parked=parked,
              polls=([], [Resolved(f(0), 0, pose_changed=True)]))
    # the park clears when poll yields its answer, as the real Poser's does
    rig.poser.polls[1] = [Resolved(f(0), 0, pose_changed=True)]
    original_poll = rig.poser.poll

    def poll():
        out = original_poll()
        if out:
            parked.clear()
        return out

    rig.poser.poll = poll
    rig.run()
    assert rig.route_calls == [(0, False), (0, True)]


def test_embed_views_and_the_ack_arms(monkeypatch):
    rig = Rig(monkeypatch, files=2,
              routes={0: EmbedRenderTask(f(0), 0, POSE, needs_embed=True),
                      1: EmbedRenderTask(f(1), 1, POSE, needs_embed=False)},
              script=[EmbedViews(f(0), 0, POSE, [np.zeros((2, 2, 3))]),
                      Rendered(f(1), 1)])
    rig.run()
    assert ("embed_views", 0) in rig.log
    assert ("done.on", "Embedded", 0) in rig.log
    assert ("done.on", "Rendered", 1) in rig.log
    assert rig.done.admission.in_flight() == 0


def test_child_failure_message_goes_straight_to_done(monkeypatch):
    rig = Rig(monkeypatch, files=1,
              routes={0: PoseRenderTask(f(0), 0)},
              script=[Failure(f(0), 0, "no triangles")])
    rig.run()
    assert ("done.on", "Failure", 0) in rig.log
    assert rig.done.rows[0].error == "no triangles"


def test_redraw_writes_the_row_and_sends_the_task(monkeypatch):
    hit = CachedHit(f(0), 0, POSE, Path("/c/0.npy"), retires=False)
    rig = Rig(monkeypatch, files=1,
              routes={0: Redraw(task=EmbedRenderTask(f(0), 0, POSE, needs_embed=False),
                                hit=hit)},
              script=[Rendered(f(0), 0)])
    rig.run()
    assert rig.done.rows[0] is hit                 # the row comes from the hit
    assert rig.done.retired_ids == {0}             # retirement from the ack
    assert isinstance(rig.tasks.sent[0], EmbedRenderTask)


def test_warm_hit_and_retired_need_no_child(monkeypatch):
    rig = Rig(monkeypatch, files=2,
              routes={0: CachedHit(f(0), 0, POSE, Path("/c/0.npy")),
                      1: Retired(f(1), 1)})
    rig.run()
    assert rig.done.retired_ids == {0, 1}
    assert [type(m).__name__ for m in rig.tasks.sent] == ["EndOfInput"]


# --- the error boundaries (I4, C-R1-5) ---------------------------------------

def test_a_raising_embed_retires_the_file_and_drops_its_tiles(monkeypatch):
    """The stash is ~9 MB of tiles at 512 px and nothing else would ever free
    it (C-R1-5): the Failure arm retires AND drops."""
    rig = Rig(monkeypatch, files=1,
              routes={0: PoseRenderTask(f(0), 0)},
              script=[PoseTiles(f(0), 0, np.zeros(6), [[np.zeros((2, 2, 3))]])],
              fail_tiles=(0,))
    rig.run()
    assert rig.done.rows[0].error == "CUDA out of memory"
    assert rig.poser.dropped == [0]
    assert rig.done.admission.in_flight() == 0     # retired, so no hang (I1)


def test_a_raising_score_still_retires(monkeypatch):
    rig = Rig(monkeypatch, files=1,
              routes={0: EmbedRenderTask(f(0), 0, POSE, needs_embed=True)},
              script=[EmbedViews(f(0), 0, POSE, [np.zeros((2, 2, 3))])],
              fail_views=(0,))
    rig.run()
    assert rig.done.rows[0].error == "CUDA out of memory"
    assert rig.poser.dropped == [0]


def test_route_raising_on_the_walk_becomes_a_row(monkeypatch):
    """J3: route stats a file the walk cache lists but that vanished."""
    rig = Rig(monkeypatch, files=2,
              routes={0: FileNotFoundError("m0.stl"),
                      1: Retired(f(1), 1)})
    rig.run()
    assert "m0.stl" in rig.done.rows[0].error
    assert rig.done.retired_ids == {0, 1}


def test_route_raising_on_the_re_route_becomes_a_row(monkeypatch):
    """The ensemble's own exit: the Resolved is dispatched inside the match
    arm, so that arm's Failure+drop guard covers the re-route's raise."""
    rig = Rig(monkeypatch, files=1,
              routes={(0, False): PoseRenderTask(f(0), 0),
                      (0, True): FileNotFoundError("gone")},
              script=[PoseTiles(f(0), 0, np.zeros(6), [[np.zeros((2, 2, 3))]])],
              answers={0: Resolved(f(0), 0, pose_changed=True)})
    rig.run()
    assert "gone" in rig.done.rows[0].error
    assert rig.poser.dropped == [0]


def test_route_raising_on_a_polled_resolved_becomes_a_row(monkeypatch):
    """W2-R1-2: the same raise reached through `poll` — a parked file resumed
    by the arbiter — gets the identical guard, Failure AND drop. `drop` is a
    no-op here (poll popped the park first), which is the point: the two arms
    are literally the same guard, not the same guard minus a line."""
    parked = {0: "future"}

    def poll_once():
        if parked:
            parked.clear()
            return [Resolved(f(0), 0, pose_changed=True)]
        return []

    rig = Rig(monkeypatch, files=1,
              routes={(0, False): PoseRenderTask(f(0), 0),
                      (0, True): FileNotFoundError("gone")},
              script=[PoseTiles(f(0), 0, np.zeros(6), [[np.zeros((2, 2, 3))]])],
              answers={0: None}, parked=parked)
    rig.poser.poll = poll_once
    rig.run()
    assert "gone" in rig.done.rows[0].error
    assert rig.poser.dropped == [0]
    assert rig.done.admission.in_flight() == 0     # retired, never a hang (I1)


# --- admission ---------------------------------------------------------------

def test_admission_bounds_the_window(monkeypatch):
    """The parent never blocks on a send; admission is the only forward
    pressure, so in_flight can never exceed the window."""
    pending = []

    def on_empty():
        # the "child": answers the oldest outstanding task, one per poll
        if pending:
            rig.results.script.append(Rendered(*pending.pop(0)))

    rig = Rig(monkeypatch, files=6, window=2,
              routes={i: EmbedRenderTask(f(i), i, POSE, needs_embed=False)
                      for i in range(6)},
              on_empty=on_empty)
    original_send = rig.tasks.send

    def send(m):
        original_send(m)
        if isinstance(m, EmbedRenderTask):
            pending.append((m.file, m.index))

    rig.tasks.send = send
    rig.cfg.tasks = rig.tasks
    rig.run()
    assert max(inflight for inflight, _ in rig.tasks.at_send) <= 2
    assert rig.done.retired_ids == set(range(6))


def test_the_walk_never_sleeps_while_it_could_admit_another_file(monkeypatch):
    """Invariant 5b, the half F-3 found unpinned: the parent must never make a
    *blocking* recv while it is holding work it could dispatch instead. The
    post-admit drain is that call — `drain(block=False)` — and its whole point
    is to pick up cheap results without stalling the walk behind the child.

    Pinned by *when* the parent chose to sleep rather than by the flag: every
    blocking recv has to find either a full window (the admission gate, where
    waiting is the pressure) or an exhausted walker (quiescence). A blocking
    post-admit drain lands with room in the window and files left, and this
    fails — which flipping that one argument to True is enough to produce."""
    pending = []

    def on_empty():
        if pending:                     # the "child", one answer per sleep
            rig.results.script.append(Rendered(*pending.pop(0)))

    rig = Rig(monkeypatch, files=3, window=3,
              routes={i: EmbedRenderTask(f(i), i, POSE, needs_embed=False)
                      for i in range(3)},
              on_empty=on_empty)
    original_send = rig.tasks.send

    def send(m):
        original_send(m)
        if isinstance(m, EmbedRenderTask):
            pending.append((m.file, m.index))

    rig.tasks.send = send
    rig.run()
    assert rig.results.blocking_at, "the run never blocked at all — rig broken"
    assert all(in_flight >= 3 or admitted == 3
               for in_flight, admitted in rig.results.blocking_at), \
        f"blocked with the window open and files left: {rig.results.blocking_at}"


def test_make_transports_leaves_tasks_unbounded_and_bounds_results_at_the_window():
    """Invariant 5a: the parent never blocks on a send. It is a property of
    two constructor arguments, so it lives in one function (`make_transports`)
    and is pinned here against the real queues — the fakes above cannot see it,
    and `maxsize=1` on `tasks` passes all of them while deadlocking a real run.

    `tasks` must be unbounded because `Done._retire` sends a `Release` on it
    from the parent's own thread, inside the drain: the only thread that could
    relieve a full `tasks` is the one that would be blocked filling it, and the
    child cannot help — it consumes `tasks` only between its own sends, which
    the parent would no longer be reading. `results` is bounded at exactly
    WINDOW, one slot per admitted file."""
    tasks, results = driver.make_transports()

    def sends_within(transport, msg, seconds):
        done = threading.Event()

        def put():
            transport.send(msg)
            done.set()

        threading.Thread(target=put, daemon=True).start()
        return done.wait(seconds)

    try:
        for i in range(driver.WINDOW + 5):        # well past any window
            assert sends_within(tasks, Retired(f(i), i), 5), \
                "a send on `tasks` blocked — the parent can deadlock itself"
        for i in range(driver.WINDOW):
            assert sends_within(results, Rendered(f(i), i), 5)
        assert not sends_within(results, Rendered(f(9), 9), 0.3), \
            "`results` is not bounded at WINDOW"
        assert results.recv(timeout=5) is not None    # unblock that last send
    finally:
        tasks.close()
        results.close()


def test_every_admitted_index_retires_exactly_once(monkeypatch):
    """Invariant 1, across a mixed run: a warm hit, a redraw, an ack, a
    failure and a skip-embed retirement."""
    rig = Rig(monkeypatch, files=4, window=4,
              routes={0: CachedHit(f(0), 0, POSE, Path("/c/0.npy")),
                      1: Redraw(task=EmbedRenderTask(f(1), 1, POSE, needs_embed=False),
                                hit=CachedHit(f(1), 1, POSE, Path("/c/1.npy"),
                                              retires=False)),
                      2: EmbedRenderTask(f(2), 2, POSE, needs_embed=True),
                      3: Retired(f(3), 3)},
              script=[Rendered(f(1), 1), Failure(f(2), 2, "boom")])
    rig.run()
    a = rig.done.admission
    assert (a.admitted, a.retired) == (4, 4) and a.in_flight() == 0


# --- quiescence --------------------------------------------------------------

def test_end_of_input_follows_quiescence(monkeypatch):
    """I1: EndOfInput is sent only once in_flight is 0 AND nothing is parked —
    the arbiter tail resolves files long after the walker runs dry."""
    parked = {0: "future"}

    def poll_once():
        if rig.results.empties >= 2 and parked:
            parked.clear()
            return [Resolved(f(0), 0, pose_changed=True)]
        return []

    rig = Rig(monkeypatch, files=1,
              routes={(0, False): PoseRenderTask(f(0), 0),
                      (0, True): Retired(f(0), 0)},
              script=[PoseTiles(f(0), 0, np.zeros(6), [[np.zeros((2, 2, 3))]])],
              answers={0: None}, parked=parked)
    rig.poser.poll = poll_once
    rig.run()
    eoi = [i for i, m in enumerate(rig.tasks.sent) if isinstance(m, EndOfInput)]
    assert len(eoi) == 1 and eoi[0] == len(rig.tasks.sent) - 1
    assert rig.tasks.at_send[eoi[0]] == (0, 0)     # nothing in flight, none parked
    assert rig.child.joins == [None]               # UNTIMED on the drain path
    assert rig.done.flushes >= 1


def test_flush_follows_the_join_on_the_drain_path(monkeypatch):
    rig = Rig(monkeypatch, files=1, routes={0: Retired(f(0), 0)})
    rig.run()
    order = [e[0] for e in rig.log if e[0] in ("tasks.send", "child.join", "flush")]
    assert order[-3:] == ["tasks.send", "child.join", "flush"]


# --- the dead and wedged child ----------------------------------------------

def test_a_dead_child_fails_what_it_owed_and_stops_the_walk(monkeypatch):
    rig = Rig(monkeypatch, files=6, window=2, exitcode=1,
              routes={i: PoseRenderTask(f(i), i) for i in range(6)})
    rig.run()
    # clause 2: the walk stops within WINDOW files of the first unserved file
    assert rig.done.admission.admitted <= 3
    assert all(isinstance(r, Failure) for r in rig.done.rows.values())
    assert rig.done.admission.in_flight() == 0
    assert ("child.kill",) not in rig.log          # dead, not wedged


def test_a_warm_run_completes_over_a_dead_child(monkeypatch):
    """Clause 1, and the reason the front edge is inexact on purpose: a run
    that needs nothing from the child finishes normally however it died."""
    rig = Rig(monkeypatch, files=3, exitcode=1,
              routes={i: CachedHit(f(i), i, POSE, Path(f"/c/{i}.npy"))
                      for i in range(3)})
    rig.run()
    assert rig.done.retired_ids == {0, 1, 2}
    assert not any(isinstance(r, Failure) for r in rig.done.rows.values())


def test_a_wedged_child_is_killed_after_stall_s(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(driver, "now", lambda: clock["t"])

    def on_empty():
        clock["t"] += driver.STALL_S / 2 + 1       # two empty polls cross it

    rig = Rig(monkeypatch, files=2, window=1,
              routes={i: PoseRenderTask(f(i), i) for i in range(2)},
              on_empty=on_empty)
    rig.run()
    assert ("child.kill",) in rig.log              # kill BEFORE any join, so
    kill = rig.log.index(("child.kill",))          # the drain join never meets
    joins = [i for i, e in enumerate(rig.log) if e[0] == "child.join"]
    assert all(j > kill for j in joins)            # a live wedge (M3)
    assert all(isinstance(r, Failure) for r in rig.done.rows.values())
    assert "stopped responding" in rig.done.rows[0].error


def test_the_stall_clock_ignores_a_healthy_arbiter_tail(monkeypatch):
    """N1: the deadline runs against child_owed(), which a parked file leaves —
    otherwise 45 s p95 of arbiter latency would read as child silence on the
    one state this pipeline enters by design."""
    clock = {"t": 0.0}
    monkeypatch.setattr(driver, "now", lambda: clock["t"])
    parked = {0: "future"}

    def on_empty():
        clock["t"] += driver.STALL_S               # far past the deadline
        if clock["t"] > 3 * driver.STALL_S and parked:
            parked.clear()
            rig.poser.polls.append([Resolved(f(0), 0, pose_changed=True)])

    rig = Rig(monkeypatch, files=1,
              routes={(0, False): PoseRenderTask(f(0), 0),
                      (0, True): Retired(f(0), 0)},
              script=[PoseTiles(f(0), 0, np.zeros(6), [[np.zeros((2, 2, 3))]])],
              answers={0: None}, parked=parked, on_empty=on_empty)
    rig.run()
    assert ("child.kill",) not in rig.log
    assert not any(isinstance(r, Failure) for r in rig.done.rows.values())


def test_the_stall_clock_restarts_when_the_child_owes_nothing(monkeypatch):
    """W2-R1-1: an arbiter tail is silence the clock must not accumulate. With
    nothing owed the clock is re-read every drain, so when the fold finally
    un-parks the file — here past STALL_S into the run — the task it sends
    gets its own full window instead of being killed with 0 ms to answer."""
    clock = {"t": 0.0}
    monkeypatch.setattr(driver, "now", lambda: clock["t"])
    parked = {0: "future"}
    state = {"unparked_at": None, "answered": False}

    def on_empty():
        clock["t"] += 60.0
        if clock["t"] > driver.STALL_S + 20 and parked:   # the arbiter, late
            parked.clear()
            state["unparked_at"] = clock["t"]
            rig.poser.polls.append([Resolved(f(0), 0, pose_changed=True)])
        elif state["unparked_at"] and not state["answered"] and \
                clock["t"] - state["unparked_at"] >= driver.STALL_S / 2:
            state["answered"] = True                      # well inside the
            rig.results.script.append(                    # fresh window
                EmbedViews(f(0), 0, POSE, [np.zeros((2, 2, 3))]))

    rig = Rig(monkeypatch, files=1,
              routes={(0, False): PoseRenderTask(f(0), 0),
                      (0, True): EmbedRenderTask(f(0), 0, POSE, needs_embed=True)},
              script=[PoseTiles(f(0), 0, np.zeros(6), [[np.zeros((2, 2, 3))]])],
              answers={0: None}, parked=parked, on_empty=on_empty)
    rig.run()
    assert state["unparked_at"] > driver.STALL_S          # the tail outlasted it
    assert ("child.kill",) not in rig.log
    assert not any(isinstance(r, Failure) for r in rig.done.rows.values())
    assert rig.done.retired_ids == {0}


def test_fail_outstanding_never_drops_a_parked_file(monkeypatch):
    """N3/C-R2-2: a parked file's in-flight answer is already paid for and must
    still fold before flush — `drop` would cancel exactly that future. Under
    --skip-embed it is not even failed: it needs nothing further from the
    child (M4)."""
    parked = {0: "future"}

    def on_empty():
        if rig.results.empties > 3 and parked:
            parked.clear()          # the fold, arriving after the child's death
            rig.poser.polls.append([Resolved(f(0), 0, pose_changed=True)])

    rig = Rig(monkeypatch, files=3, window=2, skip_embed=True, exitcode=1,
              routes={(0, False): PoseRenderTask(f(0), 0),
                      (0, True): Retired(f(0), 0),
                      1: PoseRenderTask(f(1), 1), 2: PoseRenderTask(f(2), 2)},
              script=[PoseTiles(f(0), 0, np.zeros(6), [[np.zeros((2, 2, 3))]])],
              answers={0: None}, parked=parked, on_empty=on_empty)
    rig.run()
    assert rig.poser.dropped == []                 # never, on this path
    assert not isinstance(rig.done.rows.get(0), Failure)   # excluded (M4)
    assert isinstance(rig.done.rows[1], Failure)
    assert rig.done.retired_ids >= {0, 1}          # the fold retired 0 itself


# --- abort (§Shutdown) -------------------------------------------------------

def abort_on_second_poll(rig):
    """Deliver a real SIGINT mid-run, so the driver's own handler is what sets
    the flag — and the handler restores the previous disposition, which is what
    makes the *second* Ctrl-C a hard exit."""
    def on_empty():
        if rig.results.empties == 2:
            signal.raise_signal(signal.SIGINT)
    rig.results.on_empty = on_empty


def test_a_ctrl_c_that_kills_the_child_writes_no_failure_rows(monkeypatch):
    """A terminal Ctrl-C reaches the whole foreground process group, so the
    render child takes the SIGINT too. Whatever it was mid-render on was
    *interrupted*, not broken: the abort discards in-flight work by design and
    the next run picks those files up, so nothing may be written to the CSV as
    a render failure. Error rows are retirements — they would outlive the run
    that invented them."""
    rig = Rig(monkeypatch, files=3, window=1,
              routes={i: PoseRenderTask(f(i), i) for i in range(3)})

    def on_empty():                      # the real sequence: signal first,
        if rig.results.empties == 1:     # then the child dies of it, then the
            signal.raise_signal(signal.SIGINT)   # parent's liveness check runs
            rig.child.exitcode = 1
    rig.results.on_empty = on_empty

    rig.run()
    assert not any(isinstance(r, Failure) for r in rig.done.rows.values()), \
        "an interrupted file was recorded as a render failure"
    assert ("child.kill",) not in rig.log        # it died of the signal, not a wedge


def test_abort_runs_the_fixed_order_with_a_flush_on_both_sides(monkeypatch,
                                                               capsys):
    rig = Rig(monkeypatch, files=3, window=1,
              routes={i: PoseRenderTask(f(i), i) for i in range(3)},
              parked={7: "future"})
    abort_on_second_poll(rig)
    rig.poser.abandoned = 2
    rig.run()
    order = [e[0] for e in rig.log if e[0] in
             ("arbiter.shutdown", "fold_done", "flush", "settle",
              "tasks.close", "child.join")]
    assert order == ["arbiter.shutdown", "fold_done", "flush", "settle",
                     "flush", "tasks.close", "child.join", "flush"]
    #                                                       ^ the finally's,
    # idempotent by contract and the reason a bug above still leaves artifacts
    assert ("settle", driver.FOLD_S) in rig.log
    assert rig.child.joins == [driver.JOIN_S]      # timed, unlike the drain's
    assert rig.tasks.closed
    assert not any(isinstance(m, EndOfInput) for m in rig.tasks.sent)


def test_the_wind_down_prices_the_second_ctrl_c(monkeypatch, capsys):
    rig = Rig(monkeypatch, files=2, window=1,
              routes={i: PoseRenderTask(f(i), i) for i in range(2)},
              parked={7: "future", 8: "future"})
    abort_on_second_poll(rig)
    rig.poser.abandoned = 1
    rig.run()
    err = capsys.readouterr().err
    # the count is len(parked) AFTER fold_done — what the wait is actually for
    assert "folding 2 in-flight VLM calls already paid for" in err
    assert "forfeits only those 2" in err
    assert "1 calls did not answer within 60s" in err


def test_a_clean_abort_does_not_editorialise(monkeypatch, capsys):
    rig = Rig(monkeypatch, files=2, window=1,
              routes={i: PoseRenderTask(f(i), i) for i in range(2)})
    abort_on_second_poll(rig)
    rig.run()
    assert capsys.readouterr().err == ""
    assert rig.done.flushes >= 2


def test_the_second_ctrl_c_is_not_swallowed(monkeypatch):
    """The handler restores the previous disposition on its way out, so a
    second SIGINT is the interpreter's again (§Shutdown: hard exit)."""
    before = signal.getsignal(signal.SIGINT)
    rig = Rig(monkeypatch, files=1, window=1,
              routes={0: PoseRenderTask(f(0), 0)})
    abort_on_second_poll(rig)
    rig.run()
    assert signal.getsignal(signal.SIGINT) is before


# --- the shapes --------------------------------------------------------------

def test_admission_is_one_object_shared_with_done(monkeypatch):
    """P2: a second Admission yields an in_flight() that never decreases —
    I1's hang. The driver takes Done's rather than being handed a copy."""
    rig = Rig(monkeypatch, files=1, routes={0: Retired(f(0), 0)})
    rig.run()
    assert rig.done.admission.admitted == 1 and rig.done.admission.retired == 1


def test_driver_state_defaults():
    a = Admission()
    drv = DriverState(admission=a)
    assert (a.in_flight(), drv.admitted_files, drv.child_failed) == (0, {}, False)
    a.admitted += 2
    assert drv.admission.in_flight() == 2


def test_instrumented_wrap_is_a_decorator():
    """C-R1-2: instrument.arbiter_call is a context manager and Arbiter's wrap
    is a decorator — the driver is where the two meet (today's `timed`)."""
    assert driver.instrumented(lambda: 41 + 1)() == 42
