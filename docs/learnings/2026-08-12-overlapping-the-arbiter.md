## Overlapping the arbiter: measured, and what measuring caught (2026-08-12)

74 real models (Loot Studios; 1292 STLs on disk, 74 survive the
presupported/base/hollow filter), identical input, warm page cache, one flag
group changing at a time.

| condition | total | main pass | arbiter drain |
|---|---|---|---|
| inline arbiter, no prefetch | 631 s | 10:17 | — |
| **deferred arbiter + prefetch** | **456 s** | 6:39 | 0:41 |
| deferred + prefetch, no `--save-renders` | 459 s | 6:42 | 0:41 |

**28% faster**, and `--save-renders` is free — 456 s against 459 s is inside
the run-to-run noise (two identical control arms differed by 5%). That matches
the 0.13 s/model JPEG encode measured earlier, and confirms the render
decoupling removed the real cost.

**The first version of this measured *slower* — 661 s against 600 s — and the
bug it exposed is the point of the whole exercise.** Deferral moved the file's
arbiter call off the critical path correctly (main pass 9:47 → 6:14, removing
213 s of inline waiting) and then gave it all back in a 4:33 drain. Two causes,
both invisible to the test suite:

- The revisit re-entered the per-file function from the top: reload the mesh,
  re-render 24 candidate tiles, re-embed them — to reach a decision the
  ensemble had already made and stored.
- The deferral hook was keyed on *"is this the first visit"*, so on the revisit
  it was `None` and the arbiter was asked **a second time** — a duplicate billed
  call per escalated file, whose answer then overwrote the deferred one.

Every unit test passed either way. Both paths produced a valid pose. The
duplicate call was visible only as wall-clock and as a bill. It also explains
three models where the two arms disagreed on the up axis: not a race, just two
independent VLM calls on the same sheet answering differently.

**Generalised: an optimisation that adds a second code path for the same work
needs an A/B against the path it replaces, not a correctness test.** A 3-file
smoke test exercised the deferred path and could not have caught this, because
with 3 files there is nothing to overlap with.

Two methodology traps worth keeping:

- **Page cache.** The first timing (664 s) read 1.2 GB of STLs cold from the
  USB drive; every later run read them warm. Comparing against it would have
  credited the code with the page cache. Both arms must be warm, or both cold.
- **Sampling a moving system.** Watching GPU/CPU/disk mid-run and narrating
  each snapshot produced three wrong conclusions in an hour — "CPU-bound
  parsing" (the drive does 139 MB/s and CPU sat at half a core), "the arbiter
  is overlapping, 19 sockets" (system-wide count; the process held 0), and
  "the run has stalled" (it had already exited). Sample to completion, then
  interpret.

**The single-threaded prefetcher is not a bottleneck here, and was nearly
"fixed" on a hunch.** Measured on this input: mesh load 0.67 s against 1.81 s
of GPU work per model, a ratio of **0.37**. One loader thread stays ahead
comfortably; it only becomes the pacer above 1.0, which needs the heavy tail
(the p99 mesh loads in 15.4 s).

