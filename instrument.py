"""Stage timing and device sampling for the pipeline baseline (spike 1).

Answers one question: where does the wall clock go, and what is each device
doing while it goes there. The actor proposal is justified by ~70% idle on the
RTX 4060; this is what attributes that idle to stages instead of guessing.

Two measurements, deliberately separate:

* **Stage timing** is exact and *exclusive* — entering a nested stage pauses its
  parent, so the totals sum to the instrumented run rather than double-counting.
  Stages are tracked per thread, so the main thread's stages are the critical
  path and the arbiter pool's are overlapped work that costs no wall clock.
* **Device sampling** is statistical: a background thread records CPU, NVIDIA and
  amdgpu utilization against whatever stage the main thread is in. Individual
  samples are noisy; across a few hundred models the per-stage means are not.

Sampling is driven by `nvidia-smi -lms`, one persistent process whose output
lines set the cadence. That avoids both a spawn per sample (~40 ms) and clock
drift, and pynvml is not installed here. The amdgpu side is a sysfs read and
effectively free.

Everything is off unless enable() is called, and stage() is a cheap flag check
when it is off.

**Two processes, one table** (F-7). The render child times its own stages —
mesh-load, pose-render, view-render, save-renders — and it must not sample:
one nvidia-smi per run is the cadence, and a second sampler would double the
cost and interleave two clocks in one file. So the child calls
`enable(path, sample=False)`, ships `stage_totals()` back on `EndOfInput`
(src/render_child.py), and the parent folds them in with `merge()` under the
`child` role. Nothing here imports torch — which is what lets the child import
this module at all (interfaces.md's import table).
"""
import atexit
import json
import os
import subprocess
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

_enabled = False
_lock = threading.Lock()
_totals = defaultdict(float)      # (role, stage) -> seconds, exclusive
_counts = defaultdict(int)
_samples = []
_local = threading.local()
_current_main = "startup"         # what the sampler labels its samples with
_arbiter_inflight = 0
_started = None
_out_path = None
_sampler = None
_nv_proc = None
_stop = threading.Event()


def _role():
    """main thread = the critical path; everything else is overlapped."""
    return "main" if threading.current_thread() is threading.main_thread() else "async"


class _Frame:
    __slots__ = ("name", "since", "elapsed")

    def __init__(self, name, now):
        self.name, self.since, self.elapsed = name, now, 0.0

    def pause(self, now):
        self.elapsed += now - self.since
        self.since = None

    def resume(self, now):
        self.since = now

    def stop(self, now):
        if self.since is not None:
            self.elapsed += now - self.since
        return self.elapsed


@contextmanager
def stage(name):
    """Time a stage, exclusive of any nested stage inside it."""
    if not _enabled:
        yield
        return
    global _current_main
    stack = getattr(_local, "stack", None)
    if stack is None:
        stack = _local.stack = []
    now = time.perf_counter()
    if stack:
        stack[-1].pause(now)
    frame = _Frame(name, now)
    stack.append(frame)
    is_main = _role() == "main"
    if is_main:
        _current_main = name
    try:
        yield
    finally:
        now = time.perf_counter()
        elapsed = frame.stop(now)
        stack.pop()
        if stack:
            stack[-1].resume(now)
        if is_main:
            _current_main = stack[-1].name if stack else "other"
        with _lock:
            _totals[(_role(), name)] += elapsed
            _counts[(_role(), name)] += 1


@contextmanager
def arbiter_call():
    """An in-flight arbiter request. Counted so the tail can be read against how
    many calls were actually outstanding, not just how long the tail took."""
    if not _enabled:
        yield
        return
    global _arbiter_inflight
    with _lock:
        _arbiter_inflight += 1
    try:
        with stage("arbiter-call"):
            yield
    finally:
        with _lock:
            _arbiter_inflight -= 1


def _amd_card():
    """The amdgpu render device, or None. This is the iGPU that Filament ended
    up on — its 'vram' is carved out of system RAM, so its memory number is not
    comparable to the 4060's."""
    for dev in sorted(Path("/sys/class/drm").glob("card*/device")):
        try:
            driver = os.path.basename(os.path.realpath(dev / "driver"))
        except OSError:
            continue
        if driver == "amdgpu" and (dev / "gpu_busy_percent").exists():
            return dev
    return None


def _read_int(path, default=None):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return default


def _sample_loop(interval_ms):
    import psutil

    amd = _amd_card()
    proc = psutil.Process()
    psutil.cpu_percent(interval=None)      # prime; first call is meaningless
    proc.cpu_percent(interval=None)

    global _nv_proc
    _nv_proc = nv = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
         "--format=csv,noheader,nounits", "-lms", str(interval_ms)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        for line in nv.stdout:            # one line per interval sets the cadence
            if _stop.is_set():
                break
            try:
                nv_util, nv_mem = (int(v) for v in line.split(","))
            except ValueError:
                continue
            _samples.append({
                "t": round(time.perf_counter() - _started, 3),
                "stage": _current_main,
                "inflight": _arbiter_inflight,
                "cpu": psutil.cpu_percent(interval=None),
                "proc_cpu": proc.cpu_percent(interval=None),
                "rss_mb": proc.memory_info().rss >> 20,
                "nv_util": nv_util,
                "nv_mem_mb": nv_mem,
                "amd_util": _read_int(amd / "gpu_busy_percent", -1) if amd else -1,
                "amd_vram_mb": (_read_int(amd / "mem_info_vram_used", 0) >> 20) if amd else -1,
                # GTT, not vram, is where an iGPU's buffers actually land: it is
                # system RAM mapped for the GPU. Sizing the device tier off the
                # vram carve-out alone would miss the geometry entirely.
                "amd_gtt_mb": (_read_int(amd / "mem_info_gtt_used", 0) >> 20) if amd else -1,
            })
    finally:
        nv.terminate()


def _shutdown():
    """Stop sampling and reap the nvidia-smi child.

    The sampler is a daemon thread blocked on that child's stdout, so its own
    `finally` is not guaranteed to run — an interpreter exit kills the thread
    where it stands. In practice nvidia-smi dies anyway once its stdout closes,
    but 'in practice' is not a lifecycle, and a ~24 s-per-call arbiter pool
    already taught this codebase what an unowned child costs on Ctrl-C.

    Registered with atexit rather than left to report(), because report() runs
    after the CSV write and never executes on an interrupt at all."""
    _stop.set()
    proc = _nv_proc
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def enable(path, interval_ms=100, sample=True):
    """Start timing stages. path is where the raw samples and summary are
    written by `report()`.

    `sample=False` times stages without starting the device sampler: what the
    render child wants (module docstring). It writes nothing and never calls
    `report()` — its totals travel back to the parent through `stage_totals`."""
    global _enabled, _started, _out_path, _sampler
    if _enabled:
        return
    _enabled, _started, _out_path = True, time.perf_counter(), Path(path)
    if not sample:
        return
    _sampler = threading.Thread(target=_sample_loop, args=(interval_ms,), daemon=True)
    _sampler.start()
    atexit.register(_shutdown)


def enabled():
    """Whether this process is timing — the child's `enable` gate and the
    parent's "is there a stage table to wait for" check."""
    return _enabled


def stage_totals():
    """This process's stage totals as plain tuples: (role, stage, seconds,
    count). Picklable by construction — it crosses the child->parent queue."""
    with _lock:
        return tuple((r, s, t, _counts[(r, s)]) for (r, s), t in _totals.items())


def merge(rows, role="child"):
    """Fold another process's `stage_totals()` in under one role.

    Its own role split (main/async) is dropped: a separate process consumes no
    parent wall clock at all, so the distinction the parent's table draws —
    critical path against overlapped — does not carry across the boundary. The
    child's stages are all `child`, reported as their own table."""
    if not _enabled:
        return
    with _lock:
        for _role, name, seconds, count in rows:
            _totals[(role, name)] += seconds
            _counts[(role, name)] += count


def _table(rows, headers):
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = "  ".join("-" * w for w in widths)
    out = ["  ".join(str(h).ljust(w) for h, w in zip(headers, widths)), line]
    out += ["  ".join(str(c).ljust(w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join(out)


def report():
    """Print the breakdown and write raw samples next to it. Never raises — a
    measurement harness must not be able to fail the run it is measuring."""
    if not _enabled:
        return
    try:
        _shutdown()
        wall = time.perf_counter() - _started

        def stage_rows(role):
            rows = sorted(((s, _totals[(r, s)], _counts[(r, s)])
                           for (r, s) in _totals if r == role),
                          key=lambda x: -x[1])
            return [[s, c, f"{t:.1f}", f"{100 * t / wall:4.1f}%",
                     f"{1000 * t / c:.0f}" if c else "-"] for s, t, c in rows]

        print(f"\n=== stage breakdown ({wall:.1f}s wall) ===")
        print("\ncritical path (main thread):")
        print(_table(stage_rows("main"),
                     ["stage", "n", "total s", "% wall", "ms/call"]))
        async_rows = stage_rows("async")
        if async_rows:
            print("\noverlapped (other threads) — does not consume wall clock:")
            print(_table(async_rows, ["stage", "n", "total s", "% wall", "ms/call"]))
        child_rows = stage_rows("child")
        if child_rows:
            print("\nrender child (separate process) — overlapped, and the "
                  "reason the parent's\ncritical path is mostly waiting:")
            print(_table(child_rows, ["stage", "n", "total s", "% wall", "ms/call"]))

        if _samples:
            by_stage = defaultdict(list)
            for s in _samples:
                by_stage[s["stage"]].append(s)

            def mean(rows, key):
                vals = [r[key] for r in rows if r[key] >= 0]
                return sum(vals) / len(vals) if vals else 0.0

            rows = sorted(by_stage.items(), key=lambda kv: -len(kv[1]))
            print("\ndevice utilization by stage (mean %, "
                  f"{len(_samples)} samples):")
            print(_table([[st, len(rs), f"{mean(rs, 'cpu'):.0f}",
                           f"{mean(rs, 'nv_util'):.0f}", f"{mean(rs, 'amd_util'):.0f}",
                           f"{mean(rs, 'nv_mem_mb'):.0f}", f"{mean(rs, 'amd_gtt_mb'):.0f}"]
                          for st, rs in rows],
                         ["stage", "samples", "cpu%", "nvidia%", "amd%",
                          "nv MiB", "amd GTT MiB"]))
            print(f"\noverall: cpu {mean(_samples, 'cpu'):.0f}%  "
                  f"nvidia {mean(_samples, 'nv_util'):.0f}%  "
                  f"amd {mean(_samples, 'amd_util'):.0f}%  "
                  f"peak rss {max(s['rss_mb'] for s in _samples)} MiB")

        _out_path.parent.mkdir(parents=True, exist_ok=True)
        _out_path.write_text(json.dumps({
            "wall": wall,
            "stages": [{"role": r, "stage": s, "seconds": t, "count": _counts[(r, s)]}
                       for (r, s), t in _totals.items()],
            "samples": _samples,
        }, indent=1))
        print(f"\nwrote {_out_path} ({len(_samples)} samples)")
    except Exception as e:
        print(f"instrumentation report failed: {e}")
