"""The Arbiter — a windowed pool, not an actor (docs/actor-refactor/
interfaces.md §Arbiter, data_structures.md Q1/D6).

The `Future` submit returns IS the Arbiter → Poser transport: the Poser
parks a file on it and `poll`/`fold_done`/`settle` fold the answer back in.
Extraction of today's `arbiter_pool` (classify_stls.py:1079-1080): windowing
is the pool's `max_workers` — at most `workers` calls in flight, queued
submissions start as workers free up. Rate limiting is a minimum
start-to-start spacing between calls, enforced on the worker thread so
`submit` never blocks (interfaces invariant 5: no module blocks on the
Arbiter). Today's pool has no spacing, so `min_interval=0.0` is the
extracted default.

Instrumentation is *injected*, never imported: `wrap` is applied around every
call on the worker path and the driver passes `instrument.arbiter_call` there
at wiring (C-R1-2), so this module keeps its stdlib-only import list and the
metrics stay the driver's choice.

`shutdown` draws the queued-vs-running distinction today's comment does
(classify_stls.py:1238-1241): queued futures are cancelled (unbilled work,
and a free worker must not pick up a new call mid-abort); calls already
running are not cancellable, and their non-daemon threads are joined by
`concurrent.futures`' atexit hook regardless (I7) — which is exactly why the
abort path folds them (`poser.settle`) instead of abandoning them.
"""
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable


class Arbiter:
    def __init__(self, workers: int = 8, min_interval: float = 0.0,
                 wrap: "Callable[[Callable], Callable] | None" = None,
                 clock=time.monotonic, sleep=time.sleep):
        """workers: today's --arbiter-workers (default 8). min_interval:
        seconds between call *starts*; 0 = no pacing (today's behaviour).
        wrap: a decorator applied to every call on the worker path — the
        driver passes `instrument.arbiter_call` at wiring, so the timer and
        the in-flight gauge measure the call itself rather than its queue
        wait. Injected rather than imported (C-R1-2): the Arbiter must stay
        importable without instrumentation, and a module that owns no metrics
        cannot pick the wrong ones. clock/sleep are injection seams for
        tests."""
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._min_interval = min_interval
        self._wrap = wrap
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_start = 0.0
        self._stopping = threading.Event()

    def submit(self, call: Callable[[], "int | None"]) -> Future:
        """Queue one VLM call; never blocks. Windowing and pacing both live
        behind this signature: the pool holds the window, and the wrapper
        reserves a start slot under the lock, then sleeps outside it so
        concurrent workers pace against each other without serialising.

        `wrap` goes around the call on both paths, inside the pacing sleep —
        the wait for a rate-limit slot is not call latency."""
        fn = self._wrap(call) if self._wrap is not None else call
        if self._min_interval <= 0:
            return self._pool.submit(fn)

        def paced():
            with self._lock:
                now = self._clock()
                wait = self._next_start - now
                self._next_start = max(now, self._next_start) + self._min_interval
            if wait > 0:
                self._sleep(wait)
            if self._stopping.is_set():
                return None            # see shutdown: a paced call that was
            return fn()                # still sleeping is unbilled work

        return self._pool.submit(paced)

    def shutdown(self) -> None:
        """Drop the queue, keep the in-flight calls: wait=False hands the
        timeout to the caller (the driver's FOLD_S, via poser.settle) instead
        of surrendering it to a pool with no notion of the 300 s transport
        deadline; cancel_futures=True cancels queued-never-started calls so
        no idle worker starts new work mid-abort (I7).

        The flag closes pacing's hole in that guarantee (C-R1-6): a paced call
        holds a worker while it sleeps out its rate-limit slot, so it is
        already *running* as far as `cancel_futures` is concerned — without
        the check, a min_interval large enough to matter is exactly a window
        in which the abort buys ~$0.30 answers it will never read, and each
        one extends the wait `settle` is spending. Checked after the sleep,
        not before: the point is that the call must not START after shutdown,
        and the whole sleep is time in which shutdown can arrive."""
        self._stopping.set()
        self._pool.shutdown(wait=False, cancel_futures=True)
