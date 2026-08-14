"""What does torch.compile flip on the POSE side?

compile_flips.py measured category flips (1 of 341, a coin toss); review R1
pointed out the pose path was not covered: under --compile the ensemble's
tile embeddings carry the same drift into combine_up, which decides both the
up argmax and the margin — and the margin gates a *paid* arbiter call at
MARGIN_THRESHOLD. This harness measures both exposures directly.

Per model: load the mesh, run the production ensemble path once — geometry
scores, render_up_candidate_grid, one preprocess of the flat tiles — then
push the identical tensors through eager and compiled SigLIP, score both with
upright_scores -> combine_up, and compare:

  up flip     eager and compiled disagree on the winning up candidate
  gate flip   needs_arbiter_margin answers differently — one tower would
              have bought an arbiter call the other would not

Targeting from the cached margins (diluted by tile-count and state noise
since the cache was resolved, so read results against the *eager* margin):
the tightest |margin| (up-flip risk), the nearest |margin - threshold|
(gate-flip risk), and an every-k-th control. Rendering is live — one
OffscreenRenderer, kept for the process lifetime.

No cache is written.

Usage:
  .venv/bin/python eval/compile_pose_flips.py [--cache-dir embed-cache2]
      [--group-size 60]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from common import OUT  # puts REPO on sys.path

import torch
import pose
from classify_stls import (add_cache_args, apply_run_params, as_tensor,
                           cache_root, load_file_list, load_mesh,
                           make_renderer, render_up_candidate_grid)


def main():
    parser = argparse.ArgumentParser()
    add_cache_args(parser, "unused — reads the pose cache, renders live")
    parser.add_argument("--group-size", type=int, default=60)
    args = apply_run_params(parser)
    root = cache_root(Path(args.input), args.cache_dir, confirm=False)
    files = load_file_list(Path(args.input), args.cache_dir)
    poses = pose.load_pose_cache(args.cache_dir)

    with_margin = [(v["margin"], f) for f in files
                   if (v := poses.get(pose.file_identity(f, root)))
                   and v.get("margin") is not None]
    n = args.group_size
    tight = sorted(with_margin, key=lambda t: t[0])[:n]
    gate = sorted(with_margin, key=lambda t: abs(t[0] - pose.MARGIN_THRESHOLD))[:n]
    chosen = {f: ("tight", m) for m, f in tight}
    for m, f in gate:
        chosen.setdefault(f, ("gate", m))
    rest = [t for t in with_margin if t[1] not in chosen]
    for m, f in rest[:: max(1, len(rest) // n)][:n]:
        chosen[f] = ("control", m)
    print(f"{len(chosen)} models: {sum(1 for g, _ in chosen.values() if g == 'tight')} "
          f"tight, {sum(1 for g, _ in chosen.values() if g == 'gate')} gate, "
          f"{sum(1 for g, _ in chosen.values() if g == 'control')} control")

    from transformers import AutoModel, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    from classify_stls import embed_raw
    with torch.no_grad():
        up_T = embed_raw(model, processor, pose.UPRIGHT_PROMPTS, device).float().cpu().numpy()
        down_T = embed_raw(model, processor, pose.TOPPLED_PROMPTS, device).float().cpu().numpy()
    compiled_feat = torch.compile(model.get_image_features)   # the bound method

    renderer = make_renderer(args.render_size)

    @torch.no_grad()
    def sig_scores(inputs, fn, n_candidates):
        feat = as_tensor(fn(**inputs))
        feat = torch.nn.functional.normalize(feat, dim=-1).float().cpu().numpy()
        return pose.upright_scores(feat, up_T, down_T).reshape(n_candidates, -1).mean(axis=1)

    results, done = [], 0
    t0 = time.perf_counter()
    for f, (group, cached_margin) in sorted(chosen.items(), key=lambda kv: kv[1][1]):
        try:
            mesh = load_mesh(f)
            geo = pose.up_axis_scores(mesh)
            grid = render_up_candidate_grid(renderer, mesh)
        except Exception as e:
            print(f"  skip {f.stem}: {e}", file=sys.stderr)
            continue
        flat = [im for row in grid for im in row]
        inputs = processor(images=flat, return_tensors="pt").to(device)
        idx_e, margin_e = pose.combine_up(geo, sig_scores(inputs, model.get_image_features, len(grid)))
        idx_c, margin_c = pose.combine_up(geo, sig_scores(inputs, compiled_feat, len(grid)))
        row = {"group": group, "file": f.stem, "cached_margin": cached_margin,
               "eager_margin": float(margin_e), "compiled_margin": float(margin_c),
               "up_flip": bool(idx_e != idx_c),
               "gate_flip": bool(pose.needs_arbiter_margin(margin_e)
                                 != pose.needs_arbiter_margin(margin_c))}
        results.append(row)
        if row["up_flip"] or row["gate_flip"]:
            kind = "UP" if row["up_flip"] else "GATE"
            print(f"  {kind} FLIP [{group}] {f.stem}: eager "
                  f"{pose.up_str(pose.UP_CANDIDATES[idx_e])}/{margin_e:.4f} vs "
                  f"compiled {pose.up_str(pose.UP_CANDIDATES[idx_c])}/{margin_c:.4f}")
        done += 1
        if done % 30 == 0:
            print(f"  {done}/{len(chosen)}  ({time.perf_counter() - t0:.0f}s)")

    dm = np.array([abs(r["eager_margin"] - r["compiled_margin"]) for r in results])
    print(f"\n{len(results)} models measured")
    for group in ("tight", "gate", "control"):
        rs = [r for r in results if r["group"] == group]
        print(f"{group:8s} {sum(r['up_flip'] for r in rs)} up flips, "
              f"{sum(r['gate_flip'] for r in rs)} gate flips of {len(rs)}")
    print(f"margin |delta| eager vs compiled: max {dm.max():.2e} "
          f"median {np.median(dm):.2e}")
    out = OUT / "compile_pose_flips.json"
    out.write_text(json.dumps({"threshold": pose.MARGIN_THRESHOLD,
                               "results": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
