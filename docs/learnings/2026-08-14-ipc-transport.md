## Data structures, and the queue's transport tax (2026-08-14)

The actor refactor's design session. Two products: the shapes are now fixed in
`docs/actor-refactor/data_structures.md` (frozen per-edge message types, `Pose`
as a dataclass in `pose.py`, a byte-budgeted resident-mesh dict in the child,
one admission counter), and one number in the overlap write-up got the
follow-up it was owed — whether the render-child's queue cost is worth
replacing with shared memory. Harness: `eval/ipc_spike.py`; raw results in
`eval/out/ipc_spike.json`, which is gitignored — the figures here are the
record.

### The challenge: 6–8 s of parent wait extrapolates to ~5% of a long run

The overlap spike's parent spent 6–8 s of a ~2-minute run waiting on
`queue.get()`, and the write-up waved it off as "nearly free". Extrapolated,
5% of a multi-hour cold run is real minutes, which is worth a measurement —
but that wait conflates two different things: genuine waiting (the child
hasn't finished rendering; no transport removes it) and transport (pickle,
pipe, unpickle; shared memory removes most of it). The spike isolates
transport by blasting the overlap spike's exact per-model payload — 24 tiles
+ 16 views, uint8 at 384 px, 17.7 MB/model — through each transport with
**zero render time**, so every microsecond on either side is transport.

### The pipe is the tax, not pickle

120- and 240-model runs, stable across both:

| transport | per model | throughput |
|---|---|---|
| `mp.Queue` (overlap_spike's transport) | 13.5–14.3 ms | 1.3 GB/s |
| shm block pool, consumed in place | 2.8–3.5 ms | 5.1–6.3 GB/s |
| shm block pool, copy-out worst case | 4.7–5.5 ms | 3.2–3.8 GB/s |
| raw `pickle.dumps/loads`, no pipe | 4.1 ms | 4.3 GB/s |

Raw pickle moves the payload at 4.3 GB/s; `mp.Queue` drops to 1.3 GB/s because
the bytes take two extra trips through an OS pipe. Shared memory wins 4–5× not
by skipping serialization but because one memcpy into the block replaces three
copies through the kernel. In-place beats copy-out, and is the honest mode:
SigLIP preprocessing copies anyway, so the parent can consume the view and
return the block.

### But the decomposition kills the 5% estimate

At ~14 ms/model of measured transport, the overlap spike's 60 models account
for **~0.85 s of the 6–8 s wait**. The other ~5–7 s was the parent genuinely
waiting for renders. The addressable saving is queue-minus-shm ≈ 10–11 ms per
model on the side that matters (the parent is the SigLIP feeder, the
bottleneck), against ~2 s/model of wall: **~0.5% of a cold run**, roughly a
minute per three hours — not 5%. The intuition treated the whole wait as
addressable when ~85% of it was render-wait.

### Decision: `mp.Queue` v1 behind a transport interface

Too small to complicate the first build; too cheap and too proven to discard.
The child↔parent edge goes behind a minimal `send`/`recv` interface; the shm
pool (block pool + free-list back-edge + `unlink` on abort, ~40 lines in the
spike) drops in if instrumenting the built pipeline shows the parent's `get()`
starving the 4060 — and becomes clearly worth it if renders ever go to
2048 px, where the same 4–5× applies to 29× the bytes.

### Housekeeping

`docs/masa/` is now `docs/actor-refactor/` (the proposal, the renderer
research note, and the new data-structures note); the 2026-08-13 week review
moved to `docs/reviews/`. References across `eval/`, `OPEN_QUESTIONS.md`, and
`classify_stls.py` updated.
