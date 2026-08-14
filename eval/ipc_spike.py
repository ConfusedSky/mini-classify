"""How much of the render-child queue cost is transport, and does shared
memory beat pickling?

The overlap spike's parent waited on queue.get() 6-8 s in a ~2-minute run.
That number conflates two things: genuine waiting (the child hasn't finished
rendering — no transport can remove it) and transport (pickle, pipe write,
read, unpickle — shared memory can). This harness isolates transport: the
child pre-builds one model's payload (24 tiles + 16 views, uint8, the overlap
spike's exact shape — ~17.7 MB at 384 px) and blasts N models through with
zero render time, so every microsecond on either side is transport.

Modes:
  queue      mp.Queue exactly as overlap_spike uses it — pickle in the
             feeder thread, pipe, unpickle in get()
  shm-touch  a pool of SharedMemory blocks; the child memcpys the payload in,
             a tiny (index, block) tuple crosses the queue, the parent
             consumes the arrays *in place* (strided touch) and returns the
             block. This is the real pipeline's shape: SigLIP preprocessing
             copies anyway, so the view never needs to be materialized.
  shm-copy   same, but the parent np.copy()s out before returning the block —
             the worst case, if in-place consumption turned out impossible
  pickle     pickle.dumps/loads in-process, no pipe — the serialization floor

Usage:
  .venv/bin/python eval/ipc_spike.py [--models 120] [--size 384]
"""
import argparse
import json
import multiprocessing as mp
import pickle
import time
from multiprocessing import shared_memory

import numpy as np

from common import OUT

TILES, VIEWS = 24, 16          # per-model render counts at production config


def payload(size):
    """One model's arrays, deterministic content so the parent can verify."""
    tiles = (np.arange(TILES * size * size * 3, dtype=np.uint32) % 251).astype(np.uint8)
    views = (np.arange(VIEWS * size * size * 3, dtype=np.uint32) % 241).astype(np.uint8)
    return (tiles.reshape(TILES, size, size, 3), views.reshape(VIEWS, size, size, 3))


def queue_child(n, size, q):
    tiles, views = payload(size)
    put_s = 0.0
    for i in range(n):
        t0 = time.perf_counter()
        q.put((i, tiles, views))
        put_s += time.perf_counter() - t0
    q.put(("stats", put_s))


def shm_child(n, size, names, free_q, out_q):
    tiles, views = payload(size)
    blocks = [shared_memory.SharedMemory(name=nm) for nm in names]
    split = tiles.nbytes
    copy_s = wait_s = 0.0
    for i in range(n):
        t0 = time.perf_counter()
        b = free_q.get()                       # blocks until the parent returns one
        wait_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        buf = blocks[b].buf
        np.frombuffer(buf, np.uint8, tiles.nbytes)[:] = tiles.reshape(-1)
        np.frombuffer(buf, np.uint8, views.nbytes, offset=split)[:] = views.reshape(-1)
        copy_s += time.perf_counter() - t0
        out_q.put((i, b))
    out_q.put(("stats", copy_s, wait_s))
    for blk in blocks:
        blk.close()


def consume(tiles, views):
    """Stand-in for handing the arrays onward: touch one byte per page, the
    cheapest read that defeats lazy tricks without becoming a second copy."""
    return int(tiles.reshape(-1)[::4096].sum()) + int(views.reshape(-1)[::4096].sum())


def run_queue(n, size):
    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=4)                   # overlap_spike's depth
    child = ctx.Process(target=queue_child, daemon=True, args=(n, size, q))
    wall0 = time.perf_counter()
    child.start()
    get_s = touch_s = 0.0
    stats = None
    while stats is None:
        t0 = time.perf_counter()
        msg = q.get()
        get_s += time.perf_counter() - t0
        if msg[0] == "stats":
            stats = msg
            continue
        _, tiles, views = msg
        t0 = time.perf_counter()
        consume(tiles, views)
        touch_s += time.perf_counter() - t0
    child.join()
    return {"wall_s": time.perf_counter() - wall0, "parent_get_s": get_s,
            "parent_consume_s": touch_s, "child_put_s": stats[1]}


def run_shm(n, size, copy_out):
    ctx = mp.get_context("spawn")
    t_shape, v_shape = (TILES, size, size, 3), (VIEWS, size, size, 3)
    split = int(np.prod(t_shape))
    total = split + int(np.prod(v_shape))
    blocks = [shared_memory.SharedMemory(create=True, size=total) for _ in range(6)]
    free_q, out_q = ctx.Queue(), ctx.Queue()
    for b in range(len(blocks)):
        free_q.put(b)
    child = ctx.Process(target=shm_child, daemon=True,
                        args=(n, size, [b.name for b in blocks], free_q, out_q))
    wall0 = time.perf_counter()
    child.start()
    get_s = touch_s = 0.0
    stats = None
    try:
        while stats is None:
            t0 = time.perf_counter()
            msg = out_q.get()
            get_s += time.perf_counter() - t0
            if msg[0] == "stats":
                stats = msg
                continue
            i, b = msg
            t0 = time.perf_counter()
            buf = blocks[b].buf
            tiles = np.frombuffer(buf, np.uint8, split).reshape(t_shape)
            views = np.frombuffer(buf, np.uint8, total - split, offset=split).reshape(v_shape)
            if copy_out:
                tiles, views = tiles.copy(), views.copy()
                free_q.put(b)                  # copied: block is free immediately
                consume(tiles, views)
            else:
                consume(tiles, views)          # in place: block held while consuming
                del tiles, views               # views must die before the buf can
                free_q.put(b)
            touch_s += time.perf_counter() - t0
        child.join()
    finally:
        for blk in blocks:
            blk.close()
            blk.unlink()
    return {"wall_s": time.perf_counter() - wall0, "parent_get_s": get_s,
            "parent_consume_s": touch_s, "child_copy_s": stats[1],
            "child_freeq_wait_s": stats[2]}


def run_pickle(n, size):
    tiles, views = payload(size)
    dump_s = load_s = 0.0
    for i in range(n):
        t0 = time.perf_counter()
        blob = pickle.dumps((i, tiles, views), protocol=5)
        dump_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        pickle.loads(blob)
        load_s += time.perf_counter() - t0
    return {"wall_s": dump_s + load_s, "dumps_s": dump_s, "loads_s": load_s}


def verify(size):
    """One round through each transport with content checked, so a broken
    offset can't masquerade as a fast one."""
    tiles, views = payload(size)
    want = consume(tiles, views)
    blob = pickle.loads(pickle.dumps((0, tiles, views), protocol=5))
    assert consume(blob[1], blob[2]) == want, "pickle roundtrip corrupted"
    blk = shared_memory.SharedMemory(create=True, size=tiles.nbytes + views.nbytes)
    try:
        np.frombuffer(blk.buf, np.uint8, tiles.nbytes)[:] = tiles.reshape(-1)
        np.frombuffer(blk.buf, np.uint8, views.nbytes, offset=tiles.nbytes)[:] = views.reshape(-1)
        got = consume(
            np.frombuffer(blk.buf, np.uint8, tiles.nbytes).reshape(tiles.shape),
            np.frombuffer(blk.buf, np.uint8, views.nbytes, offset=tiles.nbytes).reshape(views.shape))
        assert got == want, "shm layout corrupted"
    finally:
        blk.close()
        blk.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=int, default=120)
    parser.add_argument("--size", type=int, default=384)
    args = parser.parse_args()
    n = args.models
    mb = (TILES + VIEWS) * args.size * args.size * 3 / 1e6
    print(f"{n} models, {TILES}+{VIEWS} images at {args.size}px = {mb:.1f} MB/model")
    verify(args.size)

    results = {}
    for mode, run in [("queue", lambda: run_queue(n, args.size)),
                      ("shm-touch", lambda: run_shm(n, args.size, copy_out=False)),
                      ("shm-copy", lambda: run_shm(n, args.size, copy_out=True)),
                      ("pickle", lambda: run_pickle(n, args.size))]:
        r = run()
        r = {k: round(v, 4) for k, v in r.items()}
        r["per_model_ms"] = round(r["wall_s"] / n * 1e3, 2)
        r["gb_per_s"] = round(mb / 1e3 * n / r["wall_s"], 2)
        results[mode] = r
        parent = r.get("parent_get_s", 0) + r.get("parent_consume_s", 0)
        print(f"{mode:10s} {r['wall_s']:6.2f}s wall  {r['per_model_ms']:6.2f} ms/model  "
              f"{r['gb_per_s']:5.2f} GB/s  parent-side {parent / n * 1e3:5.2f} ms/model")

    out = OUT / "ipc_spike.json"
    out.write_text(json.dumps({"models": n, "size": args.size,
                               "mb_per_model": round(mb, 2),
                               "results": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
