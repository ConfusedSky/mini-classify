"""Does overlapping the iGPU renderer with SigLIP on the 4060 saturate the card?

The cold pipeline is strictly sequential per model: render 24 pose tiles and 16
classification views (Filament, iGPU), then embed all 40 (SigLIP, RTX 4060).
Different devices, program-order serial — the 4060 idles while Filament draws
and vice versa. Spike 4 showed threads cannot fix this (render_to_image holds
the GIL ~85-92%), so this harness measures the process-boundary version:

  baseline   one process, render then embed, model after model
  overlap    a child process renders and streams uint8 arrays through a
             bounded queue; the parent embeds as batches arrive

Identical GPU work in both modes — same tiles, same views, same batches. No
caches read or written, no arbiter, no CSV: this measures the overlap, not the
pipeline. The Amdahl prediction from the instrumented run is ~1.45x; the other
number that matters is embed-busy (SigLIP seconds / wall), which is what "GPU
utilization" actually means here. Page cache is warmed before either mode so
the USB drive's first-read cost doesn't land on whichever runs first.

Usage:
  .venv/bin/python eval/overlap_spike.py [--models 60] [--modes baseline,overlap]
  .venv/bin/python eval/overlap_spike.py --fake-embed --models 3   # plumbing only
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from common import OUT  # puts REPO on sys.path

import pose
from classify_stls import (add_cache_args, apply_run_params, cache_root,
                           load_file_list, load_mesh, make_renderer,
                           render_up_candidate_grid, render_views,
                           rotation_to_z_up, view_angles)


def pick_models(args, n):
    """Every k-th pose-cached model, deterministic. Returns [(path, up)]."""
    root = cache_root(Path(args.input), args.cache_dir, confirm=False)
    files = load_file_list(Path(args.input), args.cache_dir)
    poses = pose.load_pose_cache(args.cache_dir)
    posed = [(f, poses[k]["up"]) for f in files
             if (k := pose.file_identity(f, root)) in poses]
    step = max(1, len(posed) // n)
    return posed[::step][:n]


def render_one(renderer, f, up, angles):
    """The cold render work for one model: 24 tiles then 16 rotated views,
    in production order. Returns two uint8 stacks."""
    mesh = load_mesh(f)
    grid = render_up_candidate_grid(renderer, mesh)
    tiles = np.stack([np.asarray(im) for row in grid for im in row])
    mesh.rotate(rotation_to_z_up(np.array(up, dtype=float)), center=(0, 0, 0))
    views = np.stack([np.asarray(im) for im in render_views(renderer, mesh, angles)])
    return tiles, views


def render_child(work, size, views, elevations, queue):
    """Child process: owns Filament (and its GIL) on the iGPU. Streams
    (index, name, tiles, views, render_seconds) and a None sentinel."""
    renderer = make_renderer(size)
    angles = view_angles(views, elevations)
    for i, (f, up) in enumerate(work):
        t0 = time.perf_counter()
        try:
            tiles, view_ims = render_one(renderer, Path(f), up, angles)
        except Exception as e:
            queue.put(("error", i, Path(f).name, str(e)))
            continue
        queue.put((i, Path(f).name, tiles, view_ims, time.perf_counter() - t0))
    queue.put(None)


def make_embedder(args, fake):
    if fake:
        return lambda images: time.sleep(0.01)
    import torch
    from transformers import AutoModel, AutoProcessor
    from classify_stls import embed_images
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    def embed(images):
        embed_images(model, processor, list(images), device, batch=args.embed_batch)
    return embed


def run_baseline(work, args, embed):
    """Today's order: render, embed, next model. One process."""
    renderer = make_renderer(args.render_size)
    angles = view_angles(args.views, args.elevations)
    render_s = embed_s = 0.0
    wall0 = time.perf_counter()
    for f, up in work:
        t0 = time.perf_counter()
        tiles, views = render_one(renderer, f, up, angles)
        render_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        embed(tiles)
        embed(views)
        embed_s += time.perf_counter() - t0
    return time.perf_counter() - wall0, render_s, embed_s, 0.0


def run_overlap(work, args, embed):
    """Renderer in a child process, embeds here as batches arrive. The queue
    bound is the admission control: a few models of backlog keeps the 4060 fed
    without buffering the whole run in RAM (~18 MB per queued model)."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=args.queue_depth)
    child = ctx.Process(target=render_child, daemon=True,
                        args=([(str(f), up) for f, up in work], args.render_size,
                              args.views, args.elevations, queue))
    wall0 = time.perf_counter()
    child.start()
    render_s = embed_s = wait_s = 0.0
    done = 0
    while True:
        t0 = time.perf_counter()
        msg = queue.get()
        wait_s += time.perf_counter() - t0
        if msg is None:
            break
        if msg[0] == "error":
            print(f"  child: {msg[2]}: {msg[3]}", file=sys.stderr)
            continue
        _, _, tiles, views, r = msg
        render_s += r
        t0 = time.perf_counter()
        embed(tiles)
        embed(views)
        embed_s += time.perf_counter() - t0
        done += 1
    child.join()
    return time.perf_counter() - wall0, render_s, embed_s, wait_s


def main():
    parser = argparse.ArgumentParser()
    add_cache_args(parser, "STL directory (defaults to the last classify_stls.py run)")
    parser.add_argument("--models", type=int, default=60)
    parser.add_argument("--modes", default="baseline,overlap")
    parser.add_argument("--queue-depth", type=int, default=4)
    parser.add_argument("--embed-batch", type=int, default=None)
    parser.add_argument("--fake-embed", action="store_true",
                        help="skip SigLIP entirely — checks the process plumbing "
                             "without touching CUDA")
    args = apply_run_params(parser)
    if not args.input:
        sys.exit("no input recorded — run classify_stls.py once or pass the STL dir")

    work = pick_models(args, args.models)
    print(f"{len(work)} pose-cached models, {args.render_size}px, "
          f"{args.views}x{len(args.elevations)} views + 24 tiles each")

    # warm the page cache so neither mode pays the USB first-read
    t0 = time.perf_counter()
    total = sum(len(f.read_bytes()) for f, _ in work)
    print(f"page cache warmed: {total / 1e9:.2f} GB in {time.perf_counter() - t0:.1f}s")

    embed = make_embedder(args, args.fake_embed)
    if not args.fake_embed:
        embed(np.zeros((2, args.render_size, args.render_size, 3), dtype=np.uint8))  # CUDA warmup

    results = {}
    for mode in args.modes.split(","):
        run = {"baseline": run_baseline, "overlap": run_overlap}[mode]
        wall, render_s, embed_s, wait_s = run(work, args, embed)
        per = wall / len(work)
        results[mode] = {"wall_s": round(wall, 2), "per_model_s": round(per, 3),
                         "render_s": round(render_s, 2), "embed_s": round(embed_s, 2),
                         "embed_busy": round(embed_s / wall, 3),
                         "parent_wait_s": round(wait_s, 2)}
        print(f"\n{mode}: {wall:.1f}s wall ({per:.2f}s/model)  "
              f"render {render_s:.1f}s  embed {embed_s:.1f}s  "
              f"4060 busy {embed_s / wall:.0%}"
              + (f"  parent idle waiting {wait_s:.1f}s" if mode == "overlap" else ""))

    if {"baseline", "overlap"} <= results.keys():
        b, o = results["baseline"], results["overlap"]
        print(f"\nspeedup {b['wall_s'] / o['wall_s']:.2f}x  "
              f"(embed busy {b['embed_busy']:.0%} -> {o['embed_busy']:.0%})")

    out = OUT / "overlap_spike.json"
    out.write_text(json.dumps({"models": len(work), "config": {
        "render_size": args.render_size, "views": args.views,
        "elevations": args.elevations, "queue_depth": args.queue_depth,
        "embed_batch": args.embed_batch}, "results": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
