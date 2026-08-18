"""The Driver — the sequential loop that is the v1 architecture
(docs/actor-refactor/interfaces.md §Driver, §Shutdown; the accounting shapes
are data_structures.md's §Supervisor accounting).

Everything the parent does happens here: admission (the only forward
pressure), the drain that routes every result, the quiescence epilogue, the
dead/wedged-child walk, and the abort sequence. Nothing else in the parent
loops — every other stage is a function call returning the message that says
what to do next.

Three properties this file exists to keep, each with the finding that bought
it:

* **Every admitted index retires exactly once** (invariant 1). Every drain arm
  is an error boundary that converts to `Failure` and retires (I4), and
  `Done.retired_ids` makes the count idempotent (J2), so the accounting cannot
  go negative and finish the run early.
* **The parent never blocks on a send** (invariant 5): `tasks` is unbounded,
  `results` is bounded at the admission window, and admission is the only
  thing that waits.
* **A dead or wedged child ends the run rather than hanging it** (L1/M1/M3),
  and a healthy arbiter tail is never mistaken for either (N1) — the stall
  deadline watches `child_owed()`, which a parked file leaves.

`Admission` and `DriverState` live here because they are the driver's
bookkeeping, not messages (`Done` imports `Admission` under TYPE_CHECKING for
its annotation). Driver state is written as `drv.*` attributes: three review
passes found the same bug arriving one field at a time — state called "driver
state" in prose while the code showed a bare binding a closure silently
rebinds as a local (M2, N4, O1) — and an attribute write cannot be shadowed
that way.
"""
from __future__ import annotations

import signal
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    # Annotation-only: a runtime import of done/embedder would hand
    # `import src.driver` their module-scope torch (seconds and gigabytes
    # the fake-driven tests never need), and done <-> driver stays acyclic
    # only while one direction is annotation-only.
    from multiprocessing.process import BaseProcess

    from src.arbiter import Arbiter
    from src.done import Done
    from src.embedder import Embedder
    from src.poser import Poser

from instrument import arbiter_call, stage
from src.cache_checker import route
from src.messages import (
    CachedHit,
    CacheContext,
    EmbedRenderTask,
    EmbedViews,
    Embedded,
    EndOfInput,
    Failure,
    PoseRenderTask,
    PoseTiles,
    Redraw,
    RenderConfig,
    Rendered,
    Resolved,
    Retired,
)
from src.transport import Transport

# --- the driver's constants (named together so nobody hunts for them, N6) ----

WINDOW = 3
"""Admitted-but-unretired files. One knob: it bounds the child's backlog
(every task holds its slot until its result or `Rendered` ack returns, §P2.3),
it is the `results` queue depth, and residency depth follows it — the
roundtrip spike resolved to three resident meshes at 88% busy, and the hard
worst case is WINDOW x the heaviest mesh (data_structures.md §residency)."""

SHORT = 0.25
"""How long a blocking drain sleeps in one `recv` before looking at the world
again (the `Transport` protocol has no iterator, I16). Not a deadline: the
liveness checks below run once per pass, so this is only how often they run."""

STALL_S = 240.0
"""No progress on work the child *owes*, past this, is treated as death
(M3/N2/O4). The child's unit of work is 3-28 s per model
(actors_proposal.md:196), so this sits ~8.5x above the documented top of
range: the error is one-sided — a wedge is permanent, so four minutes is paid
once, while a false positive kills a healthy child. Deliberately not 300 s,
which is the arbiter's transport deadline (src/pose.py:506) and unrelated."""

FOLD_S = 60.0
"""The abort's wait on in-flight arbiter calls — above the 45 s p95, well
under the 300 s transport deadline. Free in wall-clock: `concurrent.futures`'
atexit hook joins those same non-daemon threads regardless (I7), so the choice
is not "wait or don't" but "read the answers or throw them away"."""

JOIN_S = 5.0
"""Abort-path join on the child. Timed, unlike the drain path's: quiescence
means the child is idle (§P2.3), an abort means nothing of the sort, and its
half-finished renders are debug artifacts."""


def now() -> float:
    return time.monotonic()


# --- the accounting (data_structures.md §Supervisor accounting) --------------

@dataclass
class Admission:
    """One counter doing three jobs — admission, quiescence, and bounding the
    in-flight window. NOT frozen, and single-writer per field: `admitted` is
    the driver's, `retired` is `Done`'s (I10, invariant 3). Exactly one
    instance exists per run; a second yields an `in_flight()` that never
    decreases, which is I1's hang."""
    admitted: int = 0
    retired: int = 0

    def in_flight(self) -> int:
        return self.admitted - self.retired


@dataclass
class DriverState:
    """The container three passes asked for one field at a time (M2, N4, O1).
    `admitted_files` is new bookkeeping, deliberately unpruned: pruning would
    need `Done` to call back into the driver on retirement — the coupling I10
    avoided — and unpruned it holds one `Path` per admitted file (~1758 at the
    end of a full run, nothing)."""
    admission: Admission
    admitted_files: dict[int, Path] = field(default_factory=dict)
    last_progress: float = 0.0
    child_failed: bool = False


@dataclass
class DriverConfig:
    """The constructed world. `classify_stls.py` builds every one of these —
    args, run-params and cache guards are the CLI's (interfaces.md:42) — and
    hands them over; `run` wires nothing itself.

    `child` is the already-spawned render child (`spawn_render_child` below):
    the driver needs only `exitcode`, `kill()` and `join()` from it, which is
    what lets a fake stand in for the whole process in tests."""
    walker: Iterable[Path]
    ctx: CacheContext
    tasks: Transport
    results: Transport
    child: BaseProcess              # or any fake with exitcode/kill()/join()
    poser: Poser
    embedder: Embedder
    done: Done
    arbiter: Arbiter
    skip_embed: bool = False
    window: int = WINDOW


def spawn_render_child(tasks: Transport, results: Transport, cfg: RenderConfig,
                       ctx=None):
    """The one child: spawn context, daemon, handed its config whole.

    Spawn is load-bearing with CUDA initialised in the parent, and `daemon`
    means an aborting parent is never held open by it. `run_child` is imported
    here rather than at module scope so the parent does not pull
    `open3d.visualization.rendering` in just to spawn (no renderer is ever
    created parent-side)."""
    import multiprocessing as mp

    from src.render_child import run_child

    ctx = ctx or mp.get_context("spawn")
    child = ctx.Process(target=run_child, args=(tasks, results, cfg), daemon=True)
    child.start()
    return child


def instrumented(call):
    """The Arbiter's `wrap` (C-R1-2): the timer and the in-flight gauge, applied
    on the worker path so they measure the call and not its queue wait.

    `instrument.arbiter_call` is a context manager, so the driver adapts it to
    the decorator `wrap` expects — structurally today's `timed` closure
    (classify_stls.py:1132-1134). Injected rather than imported by the Arbiter,
    which owns no metrics and must stay importable without them."""
    def wrapped():
        with arbiter_call():
            return call()
    return wrapped


@contextmanager
def _first_ctrl_c(stopping: threading.Event):
    """First Ctrl-C sets the flag the loops check per iteration; the handler
    then restores the previous disposition, so the **second** Ctrl-C is the
    hard exit §Shutdown promises. Silent by design — the wind-down's one line
    is priced against the `settle` wait, and a clean abort should not
    editorialise. A no-op off the main thread (a harness importing `run`)."""
    previous = signal.getsignal(signal.SIGINT)   # read before installing: a
                                                 # Ctrl-C between the two must
    def handler(signum, frame):                  # find the handler's restore
        signal.signal(signal.SIGINT, previous)   # target already bound
        stopping.set()

    try:
        signal.signal(signal.SIGINT, handler)
    except ValueError:                    # not the main thread
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def run(cfg: DriverConfig) -> None:
    """The loop. Returns when the walk is finished and quiescent, when the
    child has died and everything it owed has been failed, or when a Ctrl-C
    has wound the run down — never by raising a per-file error."""
    done, poser, embedder = cfg.done, cfg.poser, cfg.embedder
    tasks, results, child = cfg.tasks, cfg.results, cfg.child
    # ONE Admission (P2): taken from Done rather than constructed beside it, so
    # there is no second source for a copy to come from. `admitted` stays the
    # driver's field, `retired` stays Done's.
    drv = DriverState(admission=done.admission, admitted_files={},
                      last_progress=now())   # the stall clock starts at spawn
    stopping = threading.Event()             # (N4), not at the first result

    # --- the routing table ---------------------------------------------------

    def dispatch(out) -> None:
        """Everything a decision can produce, routed to the two sinks. A
        `Resolved` is re-routed through `route` — the second-call rule
        (interfaces §route): the pose store is warm by then, so the warm-`.npy`
        shortcut and the redraw arm apply instead of an unconditional
        re-embed, and `pose_changed` rides through verbatim because the Poser
        is the one that knows which tier moved the answer."""
        match out:
            case Resolved():
                dispatch(route(out.file, out.index, cfg.ctx,
                               pose_changed=out.pose_changed))
            case Redraw():
                done.on(out.hit)             # the row (retires=False) ...
                tasks.send(out.task)         # ... retirement is the child's ack
            case PoseRenderTask() | EmbedRenderTask():
                tasks.send(out)
            case CachedHit() | Retired() | Rendered() | Embedded() | Failure():
                done.on(out)
            case _:
                raise TypeError(f"driver.dispatch: unexpected {out!r}")

    def _next(block: bool):
        if not block:
            return results.recv_nowait()
        with stage("results-wait"):
            return results.recv(SHORT)

    def drain(block: bool) -> None:
        while (m := _next(block)) is not None:      # not truthiness (J8)
            drv.last_progress = now()
            try:
                match m:
                    case PoseTiles():
                        with stage("pose-embed"):
                            embeds = embedder.embed_tiles(poser.on_tiles(m))
                        out = poser.on_tile_embeds(embeds)
                        if out is not None:         # None: parked on the arbiter
                            dispatch(out)
                    case EmbedViews():
                        with stage("embed"):
                            embedded = embedder.embed_views(m)
                        done.on(embedded)
                    case Rendered() | Failure():
                        done.on(m)
                    case _:
                        raise TypeError(
                            f"unexpected message on results: {type(m).__name__}")
            except Exception as e:
                # retire, never crash (I4) — and unpin the file's ~9 MB of
                # stashed tiles, which nothing else will ever collect (C-R1-5).
                done.on(Failure(m.file, m.index, str(e)))
                poser.drop(m.index)
        for out in poser.poll():          # arbiter answers; poll is its own
            try:                          # error boundary and yields Failure
                dispatch(out)             # per file (J3) — but dispatch of a
            except Exception as e:        # Resolved re-routes through route(),
                done.on(Failure(out.file, out.index, str(e)))   # which raises
                poser.drop(out.index)     # on a vanished file, so this arm
                                          # carries the same Failure+drop guard
                                          # as the match arms above (W2-R1-2).
                                          # NO progress bump here (O2):
                                          # child_owed() already silences the
                                          # clock in the tail (N1), and a bump
                                          # would let a mid-run fold reset a
                                          # wedged child's deadline.
        owed = child_owed()               # read once: the reset below and the
        if not owed:                      # stall half agree by construction
            drv.last_progress = now()     # (W2-R1-1). Nothing owed means the
                                          # clock has nothing to measure —
                                          # without this an arbiter tail ages it
                                          # past STALL_S and the child is killed
                                          # the instant the file un-parks, with
                                          # zero ms to answer the task just
                                          # sent. While the child owes anything
                                          # — the wedge O2 protects — owed is
                                          # non-empty and no reset happens, so a
                                          # mid-run fold still cannot save one.
        if outstanding() and (child.exitcode is not None
                              or (owed
                                  and now() - drv.last_progress > STALL_S)):
            fail_outstanding(wedged=child.exitcode is None)
            # LAST, after the recv loop AND poll (M1): a result already in the
            # pipe is consumed first and never mis-blamed. The outer
            # outstanding() guard keeps a dead child's exitcode from re-firing
            # over an empty set every drain (N6).

    # --- the two subtractions, differing by exactly the parked set -----------

    def outstanding() -> list[tuple[int, Path]]:
        """Who to fail when the child is gone: admitted, unretired, and still
        needing the child (M4). Under `--skip-embed` a parked file needs
        nothing further — its paid answer folds and retires it as `Retired` on
        its own — so it is excluded; in every other mode its next step is an
        `EmbedRenderTask` a dead child cannot serve."""
        return [(i, f) for i, f in drv.admitted_files.items()
                if i not in done.retired_ids
                and not (cfg.skip_embed and i in poser.parked)]

    def child_owed() -> list[tuple[int, Path]]:
        """What counts as evidence of child liveness (N1): admitted, unretired,
        minus **all** parked files in every mode. A wedged child is always
        holding a task so a wedge never empties this; a healthy arbiter tail,
        where the child is idle by construction (I1), always does — which is
        what keeps the stall clock from running against arbiter latency
        (24 s mean, 45 s p95)."""
        return [(i, f) for i, f in drv.admitted_files.items()
                if i not in done.retired_ids and i not in poser.parked]

    def fail_outstanding(wedged: bool) -> None:
        """A wedged child is killed first — the drain path's join is untimed
        and must never meet a live wedge (M3) — then every file it still owes
        becomes a `Failure` row. Retirement is idempotent (J2) and the
        `Release`s go to a dead child harmlessly.

        It never calls `poser.drop`: a parked file's in-flight answer is
        already paid for and must still fold before `flush` (N3, C-R2-2) —
        `drop` would cancel exactly the future the quiescence loop is waiting
        to record."""
        if wedged:
            child.kill()
        drv.child_failed = True
        why = ("stopped responding" if wedged
               else f"exited with code {child.exitcode}")
        for index, f in outstanding():    # a snapshot: done.on mutates
            done.on(Failure(f, index, f"render child {why}"))

    # --- the walk ------------------------------------------------------------

    with _first_ctrl_c(stopping):
        try:
            for index, f in enumerate(cfg.walker):
                while (not drv.child_failed and not stopping.is_set()
                       and drv.admission.in_flight() >= cfg.window):
                    drain(block=True)     # liveness lives inside drain (M1),
                                          # so this gate — where the child
                                          # spends the whole run — is covered
                if drv.child_failed or stopping.is_set():
                    break                 # AFTER the gate (O3): fail_outstanding
                                          # fires inside it and releases it by
                                          # retiring, so a break checked only at
                                          # the loop top would admit one more
                                          # file to a known-dead child
                drv.admission.admitted += 1
                drv.admitted_files[index] = f
                try:
                    dispatch(route(f, index, cfg.ctx))
                except Exception as e:    # the warm path's error boundary (J3):
                    done.on(Failure(f, index, str(e)))   # route stats a file
                drain(block=False)        # the cache may list but that vanished,
                                          # and done.on(hit) loads the .npy
            while not stopping.is_set() and (drv.admission.in_flight() > 0
                                             or poser.parked):
                drain(block=True)         # quiescence FIRST (I1): the arbiter
                                          # tail resolves files long after the
                                          # walker runs dry. The parked clause
                                          # only differs after fail_outstanding
                                          # (N3) — retirement emptied in_flight,
                                          # but each fold still record_pose()s,
                                          # so the paid answers land before
                                          # flush. Bounded by the arbiter's own
                                          # 300 s transport deadline (O4), not
                                          # by anything here: five quiet minutes
                                          # is a tail, not a hang.
            if stopping.is_set():
                _abort(cfg)
                return
            tasks.send(EndOfInput())      # every Release precedes it (FIFO)
            child.join()                  # UNTIMED: quiescence means idle
        finally:
            done.flush()                  # main thread, both paths — and in a
                                          # finally, so a bug above still leaves
                                          # the pose cache and the partial CSV
    if child.exitcode != 0:               # AFTER flush, to stderr, never a
        print(f"child exit {child.exitcode}", file=sys.stderr)   # raise (M5,
        # L2): on a clean walk the run is complete and correct here and the
        # exitcode only says HOW the child ended; on the dead-child path this
        # line plus the truncated CSV IS the crash report — the lost work is
        # already Failure rows from drain's check.


def _abort(cfg: DriverConfig) -> None:
    """Ctrl-C, in the order §Shutdown fixes. Nothing here is optional and
    nothing commutes: the queue is dropped before the fold so no idle worker
    starts new work mid-abort (I7); the pose cache — the only artifact whose
    loss costs money — is durable *before* the one long wait, which is what
    bounds a second Ctrl-C's damage to the calls it interrupts; and the second
    flush picks up whatever `settle` recovered."""
    poser, done = cfg.poser, cfg.done
    cfg.arbiter.shutdown()
    poser.fold_done()
    done.flush()
    parked = len(poser.parked)        # what fold_done left is the wait's size
    if parked:
        # The abort's one long silent window is `settle`, a quiet process reads
        # as a hung one, and the reflex answer to a hang is the second Ctrl-C.
        # This line prices that keystroke honestly: the run is durable, these
        # are not. A narration that could only say "please wait" would not be
        # worth the code.
        print(f"saved {len(done.rows)} rows + pose cache; folding {parked} "
              f"in-flight VLM calls already paid for (up to {FOLD_S:g}s) — "
              f"Ctrl-C again forfeits only those {parked}", file=sys.stderr)
    abandoned = poser.settle(FOLD_S)
    done.flush()                      # AGAIN: idempotent, picks up the folds
    if abandoned:
        print(f"{abandoned} calls did not answer within {FOLD_S:g}s; those "
              f"files keep their ensemble pose", file=sys.stderr)
    cfg.tasks.close()                 # cancel_join_thread: unflushed pickles
    cfg.child.join(JOIN_S)            # must not hold the process open (I6)
