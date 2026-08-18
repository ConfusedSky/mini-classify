"""How much faster can the SigLIP side get without changing its outputs?

The pipeline embeds 24 pose tiles + 16 classification views per cold model
(`src.embedder.Embedder.embed_images`, which this loads through `eval/rig.py`
and times directly), so the only knobs that matter are batch shape,
where preprocessing runs, and whether the image forward can be compiled. This
harness measures all three on the real renders rather than noise, and never
touches a knob that would move the embeddings.

Phases (each selectable: `siglip_bench.py sweep compile prep`):
  sweep    img/s vs batch size, full path (AutoProcessor + forward) and
           forward-only on pre-preprocessed tensors. The gap is CPU preprocessing.
  compile  torch.compile on the image forward: compile cost, steady-state gain,
           and max abs diff on the normalised embeddings vs eager. Any drift
           above ~1e-3 disqualifies it — the embeddings are cached on disk.
  prep     AutoProcessor alone, per batch, on the CPU: is it big enough to be
           worth running a batch ahead of the GPU on a thread?

Raw numbers land in eval/out/siglip_bench.json; the projection at the end turns
them into whole-collection seconds for the 2284-model run.

Thermal hygiene is not optional on this card. A first run that swept the batch
sizes in ascending order read as "throughput falls with batch size" — the card
(a 4060 Laptop, 80 W cap) was in SW thermal slowdown from the warmup onwards and
its SM clock decayed 2250 -> 1320 MHz across the sweep, exactly tracking the
batch order. So: soak to a steady clock first, then measure the batch sizes in
shuffled interleaved rounds, and record the clock beside every number.
"""
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from common import OUT  # puts REPO on sys.path

import rig
from src.embedder import as_tensor, embed_texts

# The production `Embedder`, built in main(). Module-level because every phase
# below takes (model, processor, device) from the days when the forward was a
# free function; those are `EMB.model` / `EMB.processor` / `EMB.device` now,
# and the forward itself is `EMB.embed_images` — the same call the pipeline's
# Embedder makes, which is the point of benchmarking it rather than a copy.
EMB = None

RENDERS = Path(__file__).resolve().parent.parent / "embed-cache2" / "renders" / "384px-8v-e20,-20"
MODEL = "google/siglip2-so400m-patch14-384"
RESULTS = OUT / "siglip_bench.json"

BATCHES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
REPEATS = 7          # median of 7, per batch size, per path
ROUNDS = 5           # interleaved rounds for the sweep (>=5 medians per point)
WARMUP = 2
N_IMAGES = 128       # distinct renders held in memory; batches slice into these
SOAK_S = int(os.environ.get("SOAK_S", 150))   # sustained load before timing (steady clocks)
SEED = 0

# What the production run actually does per cold model, from src/embedder.py:
# `embed_tiles` sends the up-candidate grid in one un-capped call (no batch=
# argument, so --embed-batch never reaches it), then `embed_views` sends the
# 16-view list with batch=self.embed_batch.
POSE_TILES = 24
VIEW_TILES = 16
N_MODELS = 2284


def sm_clock():
    """Current SM clock in MHz, or None. Sampled beside every timing so a
    throttled point can be spotted (and normalised) instead of believed."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def gpu_state():
    """Clocks/temp/throttle reasons, so a drifting card shows up in the JSON."""
    import subprocess
    q = ("clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,"
         "clocks_throttle_reasons.active")
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"


def load_images(n=N_IMAGES):
    files = sorted(RENDERS.glob("*.jpg"))[:n]
    if len(files) < n:
        sys.exit(f"only {len(files)} renders under {RENDERS}")
    imgs = []
    for f in files:
        im = Image.open(f)
        im.load()                      # decode now: embed_images is handed in-memory renders
        imgs.append(im.convert("RGB"))
    print(f"{len(imgs)} renders loaded from {RENDERS.name}, size {imgs[0].size}")
    return imgs


def timed(fn, repeats=REPEATS, warmup=WARMUP):
    """Median/min/all wall times for fn, synchronising the device around each."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return {"median": statistics.median(ts), "min": min(ts), "max": max(ts),
            "all": [round(t, 5) for t in ts]}


def batch_slice(imgs, n):
    """n images, cycling the pool if n exceeds it (keeps pixels real either way)."""
    return [imgs[i % len(imgs)] for i in range(n)]


def soak(model, processor, imgs, device, seconds=SOAK_S):
    """Hold the card under load until its clock stops falling.

    Cold-boost numbers are not what a 2284-model run gets: this card sits in SW
    thermal slowdown within seconds of the first batch. Timing only after the
    soak means every batch size is compared at the same (sustained) clock.
    """
    print(f"\n-- soak: {seconds}s of sustained load to reach a steady clock --")
    batch = batch_slice(imgs, 32)
    t0 = time.perf_counter()
    trace = []
    nxt = 0
    while time.perf_counter() - t0 < seconds:
        EMB.embed_images(batch)
        el = time.perf_counter() - t0
        if el >= nxt:
            trace.append({"t": round(el, 1), "state": gpu_state()})
            print(f"  {el:5.1f}s  {trace[-1]['state']}")
            nxt = el + 15
    torch.cuda.synchronize()
    return trace


def duty(model, processor, imgs, device, results, gap=8.0, cycles=8, cool=60):
    """Throughput at the pipeline's duty cycle, not back-to-back.

    Back-to-back timing is the wrong regime for the current pipeline: per model
    it renders for seconds (other device) and only then embeds 24+16 images, so
    the 4060 idles and cools between bursts and runs at a much higher clock than
    a soaked card. This measures a burst of exactly the production shape with a
    render-shaped idle gap in between, which is the rate a projection should use
    while the stages stay sequential. If rendering is ever overlapped with
    embedding, the sweep's soaked numbers become the right ones instead.
    """
    print(f"\n-- duty cycle: {POSE_TILES}+{VIEW_TILES} images per burst, {gap}s idle gap "
          f"(cooling {cool}s first) --")
    time.sleep(cool)
    print(f"  after cooldown: {gpu_state()}")
    tiles, views = batch_slice(imgs, POSE_TILES), batch_slice(imgs, VIEW_TILES)
    rows = []
    for i in range(cycles):
        clk = sm_clock()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        EMB.embed_images(tiles)      # pose grid: un-capped
        EMB.embed_images(views)      # views: --embed-batch
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        rows.append({"cycle": i, "s": dt, "img_s": (POSE_TILES + VIEW_TILES) / dt,
                     "clock_mhz": clk, "state": gpu_state()})
        print(f"  cycle {i}  {dt:5.2f} s  {rows[-1]['img_s']:5.1f} img/s  clk {clk}")
        time.sleep(gap)
    med = statistics.median(r["s"] for r in rows)
    out = {"gap_s": gap, "cycles": cycles, "cool_s": cool, "per_model_images":
           POSE_TILES + VIEW_TILES, "median_s_per_model": med,
           "img_s": (POSE_TILES + VIEW_TILES) / med, "rows": rows}
    print(f"  median {med:.2f} s per model -> {out['img_s']:.1f} img/s, "
          f"{med * N_MODELS:.0f} s over {N_MODELS} models")
    results["duty"] = out
    return out


def sweep(model, processor, imgs, device, results):
    """img/s vs batch size through both paths, interleaved across rounds.

    One rep per (batch, path) per round, batch order reshuffled each round, so
    residual thermal drift spreads evenly over the batch sizes instead of
    masquerading as a batch-size effect.
    """
    print(f"\n-- sweep: {ROUNDS} interleaved rounds, full path vs forward-only --")
    rng = random.Random(SEED)
    live = list(BATCHES)
    acc = {n: {"full": [], "fwd": [], "clock": [], "peak": 0.0} for n in live}

    for rnd in range(ROUNDS + 1):     # round 0 is warmup, not recorded
        order = live[:]
        rng.shuffle(order)
        for n in order:
            batch = batch_slice(imgs, n)
            pv = None
            try:
                # preprocessed outside every timer: the forward-only path must not
                # pay for it, and holding one set per batch size would eat ~750 MB
                # of the card that the batch-128 forward needs
                pv = processor(images=batch, return_tensors="pt").to(device)["pixel_values"]

                @torch.no_grad()
                def fwd():
                    feat = as_tensor(model.get_image_features(pixel_values=pv))
                    return torch.nn.functional.normalize(feat, dim=-1)

                clk = sm_clock()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                EMB.embed_images(batch)
                torch.cuda.synchronize()
                t_full = time.perf_counter() - t0

                t0 = time.perf_counter()
                fwd()
                torch.cuda.synchronize()
                t_fwd = time.perf_counter() - t0
            except torch.cuda.OutOfMemoryError:
                print(f"  batch {n:4d}  OOM -> dropped")
                live.remove(n)
                acc[n]["oom"] = True
                del pv
                torch.cuda.empty_cache()
                continue
            del pv
            acc[n]["peak"] = max(acc[n]["peak"], torch.cuda.max_memory_allocated() / 2**30)
            torch.cuda.reset_peak_memory_stats()
            if rnd:
                acc[n]["full"].append(t_full)
                acc[n]["fwd"].append(t_fwd)
                acc[n]["clock"].append(clk)
        print(f"  round {rnd} done ({'warmup' if not rnd else 'recorded'})  {gpu_state()}")

    rows = []
    for n in BATCHES:
        a = acc[n]
        row = {"batch": n, "peak_mem_gib": a["peak"], "clocks_mhz": a["clock"]}
        if a.get("oom") or not a["full"]:
            row["oom"] = True
            rows.append(row)
            continue
        mf, mw = statistics.median(a["full"]), statistics.median(a["fwd"])
        row.update({
            "full": {"median": mf, "min": min(a["full"]), "max": max(a["full"]),
                     "all": [round(t, 5) for t in a["full"]]},
            "fwd": {"median": mw, "min": min(a["fwd"]), "max": max(a["fwd"]),
                    "all": [round(t, 5) for t in a["fwd"]]},
            "full_img_s": n / mf, "fwd_img_s": n / mw,
            "prep_share": 1 - mw / mf,
            "prep_ms": (mf - mw) * 1000,
            "median_clock_mhz": statistics.median([c for c in a["clock"] if c]) or None,
        })
        print(f"  batch {n:4d}  full {row['full_img_s']:6.1f} img/s  "
              f"fwd-only {row['fwd_img_s']:6.1f} img/s  prep {row['prep_ms']:6.1f} ms "
              f"({100 * row['prep_share']:4.1f}%)  peak {row['peak_mem_gib']:.2f} GiB  "
              f"clk {row['median_clock_mhz']:.0f}")
        rows.append(row)
    torch.cuda.empty_cache()
    results["sweep"] = rows
    return rows


def prep_only(processor, imgs, results):
    """AutoProcessor on the CPU alone — the overlap budget for a prefetch thread."""
    print("\n-- preprocessing: AutoProcessor on CPU, no GPU work --")
    rows = []
    for n in [16, 24, 32, 48, 64]:
        batch = batch_slice(imgs, n)
        ts = []
        for _ in range(WARMUP):
            processor(images=batch, return_tensors="pt")
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            processor(images=batch, return_tensors="pt")
            ts.append(time.perf_counter() - t0)
        med = statistics.median(ts)
        rows.append({"batch": n, "median": med, "per_image": med / n,
                     "all": [round(t, 5) for t in ts]})
        print(f"  batch {n:4d}  {med * 1000:7.1f} ms  ({med / n * 1000:5.2f} ms/img)")
    results["prep"] = rows
    return rows


def overlap_test(model, processor, imgs, device, results, batch=VIEW_TILES, calls=8):
    """Does preprocessing on a thread ahead of the GPU actually buy the gap?

    embed_images preprocesses batch i, waits for the GPU on batch i, then
    preprocesses batch i+1 — so the CPU cost is serial. This runs the same work
    both ways, alternating, and reports the measured saving rather than assuming
    the gap closes perfectly.
    """
    from concurrent.futures import ThreadPoolExecutor
    print(f"\n-- preprocessing overlap: {calls} batches of {batch}, inline vs prefetched --")
    lists = [batch_slice(imgs, batch) for _ in range(calls)]

    @torch.no_grad()
    def inline():
        for b in lists:
            pv = processor(images=b, return_tensors="pt").to(device)["pixel_values"]
            torch.nn.functional.normalize(as_tensor(
                model.get_image_features(pixel_values=pv)), dim=-1)

    @torch.no_grad()
    def prefetched():
        with ThreadPoolExecutor(1) as ex:
            fut = ex.submit(lambda b=lists[0]:
                            processor(images=b, return_tensors="pt")["pixel_values"])
            for i, b in enumerate(lists):
                pv = fut.result().to(device)
                if i + 1 < len(lists):
                    nxt = lists[i + 1]
                    fut = ex.submit(lambda b=nxt:
                                    processor(images=b, return_tensors="pt")["pixel_values"])
                torch.nn.functional.normalize(as_tensor(
                    model.get_image_features(pixel_values=pv)), dim=-1)

    inline(), prefetched()
    a, b_, clocks = [], [], []
    for _ in range(REPEATS):
        clocks.append(sm_clock())
        for fn_, sink in ((prefetched, b_), (inline, a)):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn_()
            torch.cuda.synchronize()
            sink.append(time.perf_counter() - t0)
    mi, mp = statistics.median(a), statistics.median(b_)
    row = {"batch": batch, "calls": calls, "clocks_mhz": clocks,
           "inline": {"median": mi, "all": [round(t, 4) for t in a]},
           "prefetched": {"median": mp, "all": [round(t, 4) for t in b_]},
           "inline_img_s": batch * calls / mi, "prefetched_img_s": batch * calls / mp,
           "saved_per_call_ms": (mi - mp) / calls * 1000,
           "speedup": mi / mp}
    print(f"  inline     {row['inline_img_s']:6.1f} img/s")
    print(f"  prefetched {row['prefetched_img_s']:6.1f} img/s  ({row['speedup']:.3f}x, "
          f"{row['saved_per_call_ms']:.1f} ms saved per call)")
    results["overlap"] = row
    return row


def drift_test(model, processor, imgs, device, results, batch=VIEW_TILES):
    """Does the compiled forward's drift reach the decisions?

    max abs diff is a bound, not an answer: what matters is whether a category
    argmax or a pose upright score flips. So score the real renders against the
    real categories.txt text embeddings both ways and count disagreements.
    """
    print("\n-- compile drift against real category decisions --")
    cats = [l.strip() for l in open(Path(__file__).resolve().parent.parent
                                    / "categories.txt") if l.strip()]
    text = embed_texts(model, processor, cats, device).float()

    @torch.no_grad()
    def embed_all(fn):
        out = []
        for i in range(0, len(imgs), batch):
            chunk = imgs[i:i + batch]
            pv = processor(images=chunk, return_tensors="pt").to(device)["pixel_values"]
            out.append(torch.nn.functional.normalize(
                as_tensor(fn(pixel_values=pv)), dim=-1).float())
        return torch.cat(out)

    a = embed_all(model.get_image_features)
    b = embed_all(torch.compile(model.get_image_features))
    sa, sb = a @ text.T, b @ text.T
    flips = int((sa.argmax(-1) != sb.argmax(-1)).sum())
    marg_a = (sa.topk(2, -1).values[:, 0] - sa.topk(2, -1).values[:, 1])
    row = {"n_images": len(imgs), "n_categories": len(cats), "argmax_flips": flips,
           "max_score_delta": (sa - sb).abs().max().item(),
           "median_top1_margin": marg_a.median().item(),
           "min_top1_margin": marg_a.min().item(),
           "images_with_margin_below_delta":
               int((marg_a < (sa - sb).abs().max()).sum())}
    print(f"  argmax flips {flips}/{len(imgs)}; max score delta "
          f"{row['max_score_delta']:.2e}; median top1 margin "
          f"{row['median_top1_margin']:.4f} (min {row['min_top1_margin']:.4f}); "
          f"{row['images_with_margin_below_delta']} images have a margin under that delta")
    results["compile_drift_decisions"] = row
    return row


def prefetch_duty(model, processor, imgs, device, results, chunk=8, cycles=8,
                  gap=6.0, cool=45):
    """The production call shape, with and without a preprocessing prefetch.

    The overlap phase found nothing because a loop of 8 back-to-back calls lets
    CUDA's async queue hide the CPU work by itself. Production does not look like
    that: one embed_images call per list, then .float().cpu() forces a sync, so
    the first chunk's preprocessing is unhidden dead GPU time. This A/Bs the real
    shape (24 tiles then 16 views, each followed by the sync) against a chunked
    version that preprocesses chunk i+1 on a thread while the GPU runs chunk i.
    Same pixels, same maths, same order: only the overlap differs.
    """
    from concurrent.futures import ThreadPoolExecutor
    print(f"\n-- prefetch at the duty cycle: chunk {chunk}, {cycles} A/B cycles --")
    tiles, views = batch_slice(imgs, POSE_TILES), batch_slice(imgs, VIEW_TILES)

    @torch.no_grad()
    def chunked(images, ex):
        fut = ex.submit(lambda b=images[:chunk]:
                        processor(images=b, return_tensors="pt")["pixel_values"])
        out = []
        for i in range(0, len(images), chunk):
            pv = fut.result().to(device, non_blocking=True)
            nxt = images[i + chunk:i + 2 * chunk]
            if nxt:
                fut = ex.submit(lambda b=nxt:
                                processor(images=b, return_tensors="pt")["pixel_values"])
            out.append(as_tensor(model.get_image_features(pixel_values=pv)))
        feat = out[0] if len(out) == 1 else torch.cat(out)
        return torch.nn.functional.normalize(feat, dim=-1)

    def current():
        for lst in (tiles, views):
            EMB.embed_images(lst).float().cpu().numpy()

    def prefetched(ex):
        for lst in (tiles, views):
            chunked(lst, ex).float().cpu().numpy()

    # equality check first: chunking must not move the embeddings
    with ThreadPoolExecutor(1) as ex:
        ref = EMB.embed_images(tiles).float()
        got = chunked(tiles, ex).float()
        eq = (ref - got).abs().max().item()
        print(f"  chunked vs current, max abs diff: {eq:.2e}")

        time.sleep(cool)
        print(f"  after cooldown: {gpu_state()}")
        a, b_ = [], []
        for i in range(cycles):
            for fn_, sink in ((current, a), (lambda: prefetched(ex), b_)):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn_()
                torch.cuda.synchronize()
                sink.append(time.perf_counter() - t0)
                time.sleep(gap)
            print(f"  cycle {i}  current {a[-1]:.3f} s  prefetched {b_[-1]:.3f} s  "
                  f"clk {sm_clock()}")
    mc, mp = statistics.median(a), statistics.median(b_)
    row = {"chunk": chunk, "cycles": cycles, "gap_s": gap, "max_abs_diff": eq,
           "current": {"median": mc, "all": [round(t, 4) for t in a]},
           "prefetched": {"median": mp, "all": [round(t, 4) for t in b_]},
           "saved_s_per_model": mc - mp, "speedup": mc / mp,
           "saved_s_collection": (mc - mp) * N_MODELS}
    print(f"  current {mc:.3f} s/model, prefetched {mp:.3f} s/model "
          f"({row['speedup']:.3f}x, {row['saved_s_collection']:.0f} s over {N_MODELS})")
    results["prefetch_duty"] = row
    return row


def compile_test(model, processor, imgs, device, results, batch=None,
                 mode="default"):
    """Compile the image forward at one fixed shape and check it against eager.

    Fixed shape on purpose: the production call sites hand it 24 tiles and then
    16 views, so a compiled forward would face two shapes and pay two compiles
    unless the batch is padded.

    mode is torch.compile's: "default", "max-autotune", "reduce-overhead".
    Worth knowing before reading a max-autotune number on this card: inductor
    gates its Triton GEMM autotuning behind is_big_gpu (>=68 SMs) and the 4060
    Laptop is far under it, so "max-autotune" here autotunes pointwise/
    reduction kernels only — the row records whether the gate was open.
    """
    batch = batch or VIEW_TILES
    torch._dynamo.reset()          # a fresh compile per mode, no cache reuse
    try:
        from torch._inductor.utils import is_big_gpu
        big_gpu = bool(is_big_gpu(0) if is_big_gpu.__code__.co_argcount else is_big_gpu())
    except Exception:
        big_gpu = None
    print(f"\n-- torch.compile mode={mode}: image forward at batch {batch} "
          f"(max_autotune_gemm gate open: {big_gpu}) --")
    imgs_b = batch_slice(imgs, batch)
    inputs = processor(images=imgs_b, return_tensors="pt").to(device)
    pv = inputs["pixel_values"]

    @torch.no_grad()
    def eager():
        feat = as_tensor(model.get_image_features(pixel_values=pv))
        return torch.nn.functional.normalize(feat, dim=-1)

    ref = eager().float().clone()
    base = timed(eager)

    fn = torch.compile(model.get_image_features,
                       mode=None if mode == "default" else mode)

    @torch.no_grad()
    def compiled():
        feat = as_tensor(fn(pixel_values=pv))
        return torch.nn.functional.normalize(feat, dim=-1)

    t0 = time.perf_counter()
    got = compiled()
    torch.cuda.synchronize()
    cold = time.perf_counter() - t0
    d = (got.float() - ref).abs()
    drift = d.max().item()
    # per-row angle is what actually decides a category: the embeddings are
    # row-normalised and then dotted with the text embeddings, so cosine against
    # the eager row bounds how far any similarity score can move
    cos = (got.float() * ref).sum(-1).clamp(-1, 1)
    row_extra = {"mean_abs_diff": d.mean().item(),
                 "min_cosine_vs_eager": cos.min().item(),
                 "max_1_minus_cosine": (1 - cos).max().item(),
                 "ref_component_rms": ref.pow(2).mean().sqrt().item()}
    print(f"  1-cos vs eager (max over rows): {row_extra['max_1_minus_cosine']:.2e}, "
          f"component rms {row_extra['ref_component_rms']:.4f}")

    # A/B interleaved: a compiled-then-eager pair per round, so a clock that keeps
    # sliding penalises both equally instead of inventing (or hiding) a speedup
    compiled(), eager()
    et, ct, clocks = [], [], []
    for _ in range(REPEATS):
        clocks.append(sm_clock())
        for fn_, sink in ((compiled, ct), (eager, et)):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn_()
            torch.cuda.synchronize()
            sink.append(time.perf_counter() - t0)
    hot = {"median": statistics.median(ct), "min": min(ct), "max": max(ct),
           "all": [round(t, 5) for t in ct]}
    ab_eager = {"median": statistics.median(et), "min": min(et), "max": max(et),
                "all": [round(t, 5) for t in et]}

    row = {"batch": batch, "mode": mode, "big_gpu_gate": big_gpu,
           "eager_pre_compile": base, "eager": ab_eager, "compiled": hot,
           "cold_wall_s": cold, "clocks_mhz": clocks,
           "eager_img_s": batch / ab_eager["median"],
           "compiled_img_s": batch / hot["median"],
           "speedup": ab_eager["median"] / hot["median"], "max_abs_diff": drift,
           "gpu_after": gpu_state(), **row_extra}
    print(f"  compile+first call {cold:6.1f} s")
    print(f"  eager    {row['eager_img_s']:6.1f} img/s")
    print(f"  compiled {row['compiled_img_s']:6.1f} img/s  ({row['speedup']:.3f}x)")
    print(f"  max abs diff on normalised embeddings: {drift:.2e}")
    key = str(batch) if mode == "default" else f"{batch}@{mode}"
    results.setdefault("compile", {})[key] = row
    return row


def project(results):
    """Whole-collection seconds for the pose and view embeds, current vs best.

    "Best" is only meaningful if it clears the measurement noise, so the spread
    of the per-batch medians is reported next to it — on a card that is power/
    thermally clamped the curve comes out flat and the winner is a coin toss.
    """
    rows = results.get("sweep") or []
    ok = [r for r in rows if "full_img_s" in r]
    if not ok:
        return
    by = {r["batch"]: r for r in ok}
    best = max(ok, key=lambda r: r["full_img_s"])
    rates = [r["full_img_s"] for r in ok]
    # within-point spread: worst rep vs best rep at each batch size, as a fraction
    jitter = statistics.median([(r["full"]["max"] - r["full"]["min"]) / r["full"]["median"]
                               for r in ok])
    spread = (max(rates) - min(rates)) / statistics.median(rates)
    print(f"\n  batch-size spread across the sweep: {100 * spread:.1f}% of median; "
          f"within-point rep jitter: {100 * jitter:.1f}%")
    if spread <= 2 * jitter:
        print("  -> flat within noise: batch size is not a throughput knob here")

    def secs(per_model_images, batch_row):
        return per_model_images * N_MODELS / batch_row["full_img_s"]

    cur_pose = by.get(POSE_TILES, best)      # un-capped: one call of 24
    cur_view = by.get(VIEW_TILES, best)      # --embed-batch 0: one call of 16
    proj = {
        "n_models": N_MODELS,
        "current": {
            "pose_batch": cur_pose["batch"], "view_batch": cur_view["batch"],
            "pose_s": secs(POSE_TILES, cur_pose), "view_s": secs(VIEW_TILES, cur_view),
        },
        "best": {
            "batch": best["batch"], "img_s": best["full_img_s"],
            "pose_s": secs(POSE_TILES, best), "view_s": secs(VIEW_TILES, best),
        },
    }
    for k in ("current", "best"):
        proj[k]["total_s"] = proj[k]["pose_s"] + proj[k]["view_s"]
    proj["saving_s"] = proj["current"]["total_s"] - proj["best"]["total_s"]
    fo = {r["batch"]: r["fwd_img_s"] for r in ok}
    proj["forward_only_floor_s"] = (POSE_TILES / fo[cur_pose["batch"]]
                                    + VIEW_TILES / fo[cur_view["batch"]]) * N_MODELS
    # The sweep runs the card soaked, which is the right regime only for an
    # embed-only pass (re-embedding cached renders). While rendering and embedding
    # stay sequential the card idles between bursts and holds 2250 MHz, so the
    # duty-cycle measurements are what a cold production run actually gets.
    for key, label in (("duty", "median_s_per_model"), ):
        d = results.get(key)
        if d:
            proj["duty_cycled"] = {"s_per_model": d[label], "img_s": d["img_s"],
                                   "total_s": d[label] * N_MODELS}
    pf = results.get("prefetch_duty")
    if pf:
        proj["production_shape"] = {
            "current_s_per_model": pf["current"]["median"],
            "current_total_s": pf["current"]["median"] * N_MODELS,
            "prefetched_s_per_model": pf["prefetched"]["median"],
            "prefetched_total_s": pf["prefetched"]["median"] * N_MODELS,
            "saving_s": pf["saved_s_collection"]}
    proj["batch_spread_frac"] = spread
    proj["rep_jitter_frac"] = jitter
    proj["flat_within_noise"] = spread <= 2 * jitter
    results["projection"] = proj
    print("\n-- projection over 2284 models --")
    print(f"  current  pose(b{proj['current']['pose_batch']}) {proj['current']['pose_s']:7.0f} s"
          f" + views(b{proj['current']['view_batch']}) {proj['current']['view_s']:7.0f} s"
          f" = {proj['current']['total_s']:7.0f} s")
    print(f"  best b{proj['best']['batch']:<3d} {proj['best']['total_s']:7.0f} s"
          f"  (saving {proj['saving_s']:.0f} s)")
    print(f"  floor if preprocessing were fully overlapped: "
          f"{proj['forward_only_floor_s']:.0f} s")
    if "duty_cycled" in proj:
        print(f"  duty-cycled (card idle between models, 2250 MHz): "
              f"{proj['duty_cycled']['img_s']:.1f} img/s -> "
              f"{proj['duty_cycled']['total_s']:.0f} s")
    if "production_shape" in proj:
        p = proj["production_shape"]
        print(f"  production shape (24+16 with the .cpu() sync): "
              f"{p['current_total_s']:.0f} s now, {p['prefetched_total_s']:.0f} s "
              f"with a prep prefetch (saves {p['saving_s']:.0f} s)")


def main():
    phases = sys.argv[1:] or ["sweep", "prep", "overlap", "compile", "drift"]
    if phases == ["project"]:
        # re-project from a saved sweep: no model, no GPU
        prev = json.loads(RESULTS.read_text())
        project(prev)
        RESULTS.write_text(json.dumps(prev, indent=1))
        print(f"\nwrote {RESULTS}")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {"model": MODEL, "device": device, "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
               "renders": str(RENDERS), "repeats": REPEATS, "warmup": WARMUP,
               "phases": phases, "gpu_at_start": gpu_state()}
    print(f"gpu at start: {results['gpu_at_start']}")

    imgs = load_images()
    global EMB
    print(f"loading {MODEL} fp16 on {device} ...")
    t0 = time.perf_counter()
    # model_load_s now covers the prompt-bank text pass too — the Embedder does
    # both in __init__, and it is a startup number, not one of the timed phases
    EMB = rig.embedder(MODEL, device=device)
    model, processor = EMB.model, EMB.processor
    results["model_load_s"] = time.perf_counter() - t0
    results["weights_gib"] = torch.cuda.memory_allocated() / 2**30
    print(f"  {results['model_load_s']:.1f} s, weights {results['weights_gib']:.2f} GiB")

    # Real work until the clock stops falling, so no timed batch is paying for
    # clock ramp-down. See the module docstring: skipping this inverts the sweep.
    results["soak"] = soak(model, processor, imgs, device)
    torch.cuda.reset_peak_memory_stats()
    results["gpu_after_soak"] = gpu_state()

    if "sweep" in phases:
        sweep(model, processor, imgs, device, results)
    if "duty" in phases:
        duty(model, processor, imgs, device, results)
    if "prep" in phases:
        prep_only(processor, imgs, results)
    if "overlap" in phases:
        overlap_test(model, processor, imgs, device, results)
    if "prefetch" in phases:
        prefetch_duty(model, processor, imgs, device, results,
                      chunk=int(os.environ.get("PREFETCH_CHUNK", 8)))
    if "drift" in phases:
        drift_test(model, processor, imgs, device, results)
    if "compile" in phases:
        # both production shapes: 24 pose tiles, then the 16-view list. A compiled
        # forward meets both, so both compiles get paid unless the batch is padded.
        for b in [int(x) for x in os.environ.get("COMPILE_BATCH", str(VIEW_TILES)).split(",")]:
            for m in os.environ.get("COMPILE_MODES", "default").split(","):
                compile_test(model, processor, imgs, device, results, batch=b,
                             mode=m.strip())
    if "sweep" in phases:
        project(results)

    results["gpu_at_end"] = gpu_state()
    prev = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    prev.update({k: v for k, v in results.items() if v is not None})
    RESULTS.write_text(json.dumps(prev, indent=1))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
