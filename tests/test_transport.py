"""src/transport.py: the boundary protocol. Bounded/unbounded send semantics,
recv/recv_nowait on empty, the I6 close contract — an aborting parent with
unflushed pickles must not be held open by the queue's feeder thread — and its
opposite, `flush`, which a process about to `os._exit` needs so its last
message is not lost in that same feeder (F-7)."""
import subprocess
import sys
import threading
import time
from pathlib import Path

from src.transport import MpQueueTransport

REPO = Path(__file__).resolve().parent.parent


def test_unbounded_send_recv_preserves_order():
    t = MpQueueTransport()
    for i in range(5):
        t.send(("msg", i))
    got = [t.recv(timeout=5) for _ in range(5)]
    assert got == [("msg", i) for i in range(5)]


def test_recv_nowait_on_empty_returns_none():
    t = MpQueueTransport()
    assert t.recv_nowait() is None


def test_recv_nowait_returns_message_when_present():
    t = MpQueueTransport()
    t.send("hello")
    deadline = time.monotonic() + 5      # the feeder thread delivers
    while time.monotonic() < deadline:   # asynchronously; poll briefly
        m = t.recv_nowait()
        if m is not None:
            assert m == "hello"
            return
        time.sleep(0.01)
    raise AssertionError("message never arrived via recv_nowait")


def test_recv_timeout_on_empty_returns_none_promptly():
    t = MpQueueTransport()
    t0 = time.monotonic()
    assert t.recv(timeout=0.1) is None
    assert time.monotonic() - t0 < 5


def test_bounded_send_blocks_only_when_full():
    t = MpQueueTransport(maxsize=1)
    t.send("first")                      # fits: must not block
    started = threading.Event()
    done = threading.Event()

    def second_send():
        started.set()
        t.send("second")                 # full: blocks until a recv
        done.set()

    th = threading.Thread(target=second_send, daemon=True)
    th.start()
    assert started.wait(5)
    assert not done.wait(0.3), "send on a full bounded queue did not block"
    assert t.recv(timeout=5) == "first"  # frees the slot
    assert done.wait(5), "blocked send never completed after recv"
    assert t.recv(timeout=5) == "second"


def test_close_with_unflushed_items_does_not_hang():
    """I6: cancel_join_thread + close. The payload exceeds the OS pipe buffer
    and nobody ever reads it, so without cancel_join_thread the feeder thread
    would block and the interpreter's exit join would hang forever."""
    code = ("from src.transport import MpQueueTransport\n"
            "t = MpQueueTransport()\n"
            "t.send(b'x' * (1 << 20))\n"
            "t.close()\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


BIG = b"x" * (1 << 20)     # exceeds the pipe buffer, so the feeder cannot
                           # finish the write in one go — which is what makes
                           # the unflushed arm below a fact and not a race


def _exiting_child(results, flush):
    """Spawn target: send, optionally flush, then `os._exit` — the render
    child's EndOfInput arm, which is the only caller of `flush`."""
    import os
    results.send(("stages", BIG))
    if flush:
        results.flush()
    os._exit(0)


def test_flush_is_what_survives_an_os_exit():
    """F-7: `put` hands the pickle to a feeder thread, and `os._exit` does not
    wait for it. Without `flush` the child's last message never arrives — the
    unflushed arm is here so nobody 'simplifies' the call away."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    got = {}
    for flush in (True, False):
        results = MpQueueTransport(maxsize=3, ctx=ctx)
        p = ctx.Process(target=_exiting_child, args=(results, flush), daemon=True)
        p.start()
        got[flush] = results.recv(timeout=10)
        p.join(timeout=30)
        assert p.exitcode == 0
    assert got[True] == ("stages", BIG)
    assert got[False] is None, "unflushed send survived os._exit — flush is dead code"


def _echo_child(tasks, results):
    """Spawn target: must be module-level so the child can unpickle it."""
    while True:
        m = tasks.recv(timeout=30)
        if m == "stop":
            return
        results.send(("echo", m))


def test_round_trip_through_a_real_child_process():
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    tasks = MpQueueTransport(ctx=ctx)
    results = MpQueueTransport(maxsize=4, ctx=ctx)   # bounded, like v1's
    p = ctx.Process(target=_echo_child, args=(tasks, results), daemon=True)
    p.start()
    try:
        tasks.send({"file": "a.stl", "index": 0})
        assert results.recv(timeout=60) == ("echo", {"file": "a.stl", "index": 0})
        tasks.send("stop")
        p.join(timeout=60)
        assert p.exitcode == 0
    finally:
        if p.is_alive():
            p.kill()
            p.join(timeout=10)
