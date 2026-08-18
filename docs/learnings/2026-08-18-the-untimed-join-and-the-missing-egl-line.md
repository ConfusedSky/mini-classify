# The untimed join, and why a missing EGL line is not evidence (2026-08-18)

The wave-3 parity run hit one stall in ~17 pipeline runs: the progress bar
reached `classifying: 100%`, only `cache-meta.json` was written, the log never
contained `[Open3D INFO] EGL headless mode enabled.`, and the process sat for
9.5 minutes before being killed — 2.4× the driver's own `STALL_S = 240`, which
exists precisely to catch a wedged child.

**Not reproduced in 92 watched runs** (cold, warm, 3-model, `--instrument`,
kill-then-rerun, two runs sharing the iGPU, spaced). Normal run 12.4–19.2 s
throughout. The diagnosis below is by elimination, at roughly 85% confidence,
and the fix makes the symptom impossible whether or not the diagnosis is right.

## Two things the evidence does not mean

Both of these read as damning and are worthless. Anyone investigating a stall
here should know them before drawing conclusions.

**`classifying: 100%` proves only that the parent reached quiescence.** tqdm
updates after the loop body returns, and the bar advances on *admission*, so it
hits 100% instantly — before the child has done anything. It says nothing about
the child's progress.

**A missing EGL line is a fingerprint of SIGKILL, not of a pre-EGL stall.**
Open3D writes that line into libc's block-buffered stdout during
`OffscreenRenderer` construction, and on a pipe it only reaches the file at
exit. Measured:

| child | EGL line in log |
|---|---|
| renderer up, normal `os._exit(0)` | yes |
| renderer up, then SIGKILL | **no** |

So a child whose renderer was fully up and was *later killed* produces exactly
the observed log. The absent line is evidence that `fail_outstanding` already
fired — the opposite of what it appears to say.

## The negative control: the stall detector works

A child that stalls forever before the renderer but stays killable, run against
the unmodified driver at `STALL_S = 12`, exits cleanly at the deadline and
writes `results.csv` **and** `pose-cache.json`, with one
`RENDER_ERROR: render child stopped responding` row. That is not the observed
signature. A merely-stalled-but-killable child cannot produce this hang, and
the liveness check is not the broken part.

## The diagnosis

Only `cache-meta.json` existed. It is written by `cache_root()` before the
model load, and `done.flush()` lives in `run`'s `finally` — so the parent was
inside the `try`, past the walk, and never reached the flush. Between the
(proven-working) stall check and the (proven-unreached) flush there was exactly
one unbounded wait: **`child.join()`, untimed, on a child that `SIGKILL` could
not reap.**

The sequence: stall fires at 240 s → `fail_outstanding(wedged=True)` →
`child.kill()` → files retire → quiescence exits → `EndOfInput` →
`child.join()` blocks forever. A process wedged inside Filament's EGL/amdgpu
DRM ioctls sits in uninterruptible `D` state, where it neither dies on SIGKILL
nor reaps — which explains both halves at once: why it stopped responding, and
why the kill did not take.

**This was an invariant violation, not merely a missing timeout.** The code's
own comment licensed the untimed join — *"UNTIMED: quiescence means idle"* —
and that is true on every path but one: `fail_outstanding` reaches
`in_flight() == 0` by *killing* the child, not by it going idle. The abort path
already bounded its join; the drain path's precondition simply did not hold
after a kill.

Two further untimed joins in the stdlib would have relocated the hang to
interpreter exit if only the driver's join were bounded:
`multiprocessing.util._exit_function` joins every process in `_children` with
no timeout, and `Queue._finalize_join` joins the feeder thread. The fix
discards the child from `_children` and closes `tasks` with
`cancel_join_thread`, which is why it escapes both.

## What was rejected

A **Ready handshake** (child announces its renderer is up; parent times out
waiting) would not have prevented this. The parent already detects "child
produced nothing" via `STALL_S`; a handshake only moves detection from 240 s to
~60 s and then arrives at the same untimed join. It adds a message type, a
parent-side wait, and a new false-positive mode (a slow-but-healthy cold start
killed by a tight deadline). Worth filing separately for faster failure; not
worth coupling to this fix.

## Still open

`MpQueueTransport.recv(timeout)` is not truly bounded once `_poll` succeeds and
a partial message is in the pipe — the read of the remainder has no deadline.
It requires a completed render, so it cannot explain this hang, but it is
reachable on a long run. Fixing it properly means a length-prefixed read with
its own deadline.

Blast radius, unchanged: `fail_outstanding` sets `child_failed`, which stops
the walk. One wedged child at model 500 of 1000 still ends the run — but now in
~250 s with durable caches and honest `Failure` rows, so a warm rerun resumes,
instead of hanging forever with nothing flushed. Respawning the child would
remove even that; out of scope here.
