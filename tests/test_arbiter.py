"""src/arbiter.py: the windowed pool (interfaces.md §Arbiter). Windowing is
max_workers; shutdown draws the queued-vs-running distinction (wait=False,
cancel_futures=True) and stops paced calls that have not started yet;
pacing is a start-to-start interval tested through the injected clock/sleep
seams; `wrap` is the instrumentation seam, exercised with a counter in place
of the driver's `instrument.arbiter_call`. No network anywhere — every
callable is a fake."""
import threading
import time

from src.arbiter import Arbiter


def test_instant_calls_resolve_through_futures():
    a = Arbiter(workers=2)
    futs = [a.submit(lambda i=i: i * i) for i in range(5)]
    assert [f.result(timeout=5) for f in futs] == [0, 1, 4, 9, 16]
    a.shutdown()


def test_windowing_caps_concurrent_calls():
    """workers=2: with both workers held by slow calls, a third submitted
    call must not start until one of them finishes."""
    a = Arbiter(workers=2)
    started = [threading.Event() for _ in range(3)]
    release = threading.Event()

    def slow(i):
        started[i].set()
        assert release.wait(5)
        return i

    futs = [a.submit(lambda i=i: slow(i)) for i in range(3)]
    assert started[0].wait(5) and started[1].wait(5)
    assert not started[2].wait(0.2)      # windowed out while 0 and 1 run
    release.set()
    assert sorted(f.result(timeout=5) for f in futs) == [0, 1, 2]
    assert started[2].is_set()
    a.shutdown()


def test_shutdown_cancels_queued_but_not_running():
    """Today's comment's distinction (classify_stls.py:1238-1241): queued
    futures die, the in-flight call is not cancellable and its (billed)
    answer still lands. wait=False: shutdown returns without joining it."""
    a = Arbiter(workers=1)
    running = threading.Event()
    release = threading.Event()

    def slow():
        running.set()
        assert release.wait(5)
        return "answer"

    in_flight = a.submit(slow)
    assert running.wait(5)
    queued = a.submit(lambda: "never runs")
    t0 = time.monotonic()
    a.shutdown()
    assert time.monotonic() - t0 < 1.0   # did not wait out the in-flight call
    assert queued.cancelled()
    release.set()
    assert in_flight.result(timeout=5) == "answer"


def test_min_interval_paces_call_starts():
    """Clock frozen at 0: reserved start slots advance by min_interval per
    call, so the workers request sleeps of 0 (elided), i, 2i. workers=1
    keeps the reservation order deterministic."""
    sleeps = []
    a = Arbiter(workers=1, min_interval=7.0, clock=lambda: 0.0,
                sleep=sleeps.append)
    futs = [a.submit(lambda i=i: i) for i in range(3)]
    assert [f.result(timeout=5) for f in futs] == [0, 1, 2]
    assert sleeps == [7.0, 14.0]
    a.shutdown()


def test_zero_interval_never_sleeps():
    """min_interval=0 is the extracted default — today's pool has no pacing,
    so the sleep seam must never be touched."""
    def forbidden(_):
        raise AssertionError("pacing sleep on min_interval=0")

    a = Arbiter(workers=1, sleep=forbidden)
    assert a.submit(lambda: 42).result(timeout=5) == 42
    a.shutdown()


def test_min_interval_paces_across_concurrent_workers():
    """Pacing is global, not per-worker: with workers=2 the slot reservation
    happens under the lock, so two calls that could run simultaneously still
    take start slots 0 and min_interval apart. The lock is held only for the
    reservation — the sleep is outside it, so the workers wait in parallel."""
    sleeps = []
    a = Arbiter(workers=2, min_interval=5.0, clock=lambda: 0.0,
                sleep=sleeps.append)
    futs = [a.submit(lambda i=i: i) for i in range(4)]
    assert sorted(f.result(timeout=5) for f in futs) == [0, 1, 2, 3]
    assert sorted(sleeps) == [5.0, 10.0, 15.0]   # the first slot is now: no wait
    a.shutdown()


def test_submit_does_not_block_when_the_pool_is_saturated():
    """Invariant 5: no module blocks on the Arbiter. Every worker busy means
    the next submit queues — it must return its Future immediately, not wait
    for a slot the way a bounded put() would."""
    a = Arbiter(workers=1)
    running = threading.Event()
    release = threading.Event()

    def slow():
        running.set()
        assert release.wait(5)
        return "answer"

    a.submit(slow)
    assert running.wait(5)                # the one worker is now occupied
    t0 = time.monotonic()
    queued = [a.submit(lambda i=i: i) for i in range(20)]
    assert time.monotonic() - t0 < 1.0    # 20 submits against a full pool
    assert not any(f.done() for f in queued)
    release.set()
    assert [f.result(timeout=5) for f in queued] == list(range(20))
    a.shutdown()


# --- the instrumentation seam (C-R1-2) ---------------------------------------

def counting_wrap(log):
    """Stands in for the driver's instrument.arbiter_call: a decorator that
    records entry/exit around the call itself."""
    def wrap(call):
        def wrapped():
            log.append("enter")
            try:
                return call()
            finally:
                log.append("exit")
        return wrapped
    return wrap


def test_wrap_surrounds_every_unpaced_call():
    log = []
    a = Arbiter(workers=1, wrap=counting_wrap(log))
    assert [a.submit(lambda i=i: i).result(timeout=5) for i in range(3)] == \
        [0, 1, 2]
    assert log == ["enter", "exit"] * 3
    a.shutdown()


def test_wrap_surrounds_paced_calls_inside_the_sleep():
    """The pacing wait is not call latency: the wrapper must open after the
    sleep, so a timer measures the VLM call and not the rate limiter."""
    log = []
    a = Arbiter(workers=1, min_interval=7.0, clock=lambda: 0.0,
                sleep=lambda s: log.append(f"sleep {s}"),
                wrap=counting_wrap(log))
    assert [a.submit(lambda i=i: i).result(timeout=5) for i in range(2)] == [0, 1]
    assert log == ["enter", "exit", "sleep 7.0", "enter", "exit"]
    a.shutdown()


def test_no_wrap_is_the_default():
    """The Arbiter never imports instrument; with no wrap it calls straight
    through, which is what keeps a bare test (and the REPL) free of metrics."""
    a = Arbiter(workers=1)
    assert a.submit(lambda: 42).result(timeout=5) == 42
    a.shutdown()


# --- shutdown vs. a call still sleeping out its slot (C-R1-6) ----------------

def test_shutdown_stops_a_paced_call_that_has_not_started():
    """A paced call holds a worker while it sleeps, so cancel_futures cannot
    reach it — the flag must. Ctrl-C during a rate-limit wait buys nothing."""
    sleeping = threading.Event()
    release = threading.Event()
    called = []

    def sleep(_s):
        sleeping.set()
        assert release.wait(5)

    a = Arbiter(workers=1, min_interval=60.0, clock=lambda: 0.0, sleep=sleep)
    a.submit(lambda: called.append("first") or 1).result(timeout=5)  # no wait
    fut = a.submit(lambda: called.append("second") or 2)
    assert sleeping.wait(5)               # parked in the pacing sleep
    a.shutdown()
    release.set()
    assert fut.result(timeout=5) is None   # skipped, not billed
    assert called == ["first"]
