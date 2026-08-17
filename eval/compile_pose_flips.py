"""What does torch.compile flip on the POSE side — bounded on the population
that is actually exposed?

compile_flips.py measured category flips (1 of 341, a coin toss); review R1
pointed out the pose path was not covered: under --compile the ensemble's
tile embeddings carry the drift into combine_up, which decides both the up
argmax and the margin — and the margin gates a *paid* arbiter call at
MARGIN_THRESHOLD. The first version of this harness targeted by **cached**
margin and review T2 showed why that under-samples: cached margins mix
n_az=4- and n_az=2-era resolutions and are not keyed on render size, so 54
cached-in-band models collapsed to 4 live-in-band — the gate result rested
on n=4. (That 54 -> 4 collapse is itself the measurement that turned the
render-size gap from hypothesis into observation.)

So: two phases, selection on the margins the towers actually contest.

  census   one eager-only pass over every model with a pose entry: live
           margin and live up argmax, saved to out/pose_live_margins.json
           and reused (--refresh-census to redo). Also reports
           cached-vs-live margin drift, the T2 observation.
  compare  compile-vs-eager on exactly the live-exposed set: models whose
           live margin sits within --band of 0 (up-flip exposure) or of
           MARGIN_THRESHOLD (gate-flip exposure). Tiles are rendered once
           per model and preprocessed once, so both towers see identical
           tensors; the bound method is compiled (the null-canary lesson).

No cache is written.

Usage:
  .venv/bin/python eval/compile_pose_flips.py [--cache-dir embed-cache2]
      [--band 0.022] [--refresh-census]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from common import OUT  # puts REPO on sys.path

import torch
from src import pose
from classify_stls import (add_cache_args, apply_run_params, as_tensor,
                           cache_root, embed_raw, load_file_list, load_mesh,
                           make_renderer, render_up_candidate_grid)

CENSUS = OUT / "pose_live_margins.json"


def main():
    parser = argparse.ArgumentParser()
    add_cache_args(parser, "unused — reads the pose cache, renders live")
    parser.add_argument("--band", type=float, default=0.022,
                        help="2x the max margin delta the first run observed")
    parser.add_argument("--refresh-census", action="store_true")
    args = apply_run_params(parser)
    root = cache_root(Path(args.input), args.cache_dir, confirm=False)
    files = load_file_list(Path(args.input), args.cache_dir)
    poses = pose.load_pose_cache(args.cache_dir)
    work = [(f, e) for f in files
            if (e := poses.get(pose.file_identity(f, root)))
            and e.get("margin") is not None]

    from transformers import AutoModel, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)
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

    def ensemble(f, fns):
        """geometry + tiles once, then one (idx, margin) per tower in fns."""
        mesh = load_mesh(f)
        geo = pose.up_axis_scores(mesh)
        grid = render_up_candidate_grid(renderer, mesh)
        flat = [im for row in grid for im in row]
        inputs = processor(images=flat, return_tensors="pt").to(device)
        return [pose.combine_up(geo, sig_scores(inputs, fn, len(grid))) for fn in fns]

    # --- phase 1: the live-margin census ------------------------------------
    if CENSUS.exists() and not args.refresh_census:
        census = json.loads(CENSUS.read_text())
        print(f"census: {len(census)} live margins from {CENSUS} (--refresh-census to redo)")
    else:
        print(f"census: eager pass over {len(work)} models (~{len(work) * 1.3 / 60:.0f} min)")
        census, t0 = {}, time.perf_counter()
        for i, (f, entry) in enumerate(work):
            try:
                (idx, margin), = ensemble(f, [model.get_image_features])
            except Exception as e:
                print(f"  skip {f.stem}: {e}", file=sys.stderr)
                continue
            census[str(f)] = {"cached_margin": entry["margin"],
                              "live_margin": float(margin), "live_idx": int(idx)}
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(work)}  ({time.perf_counter() - t0:.0f}s)")
        CENSUS.write_text(json.dumps(census))
        print(f"census: wrote {len(census)} live margins to {CENSUS}")

    lm = np.array([c["live_margin"] for c in census.values()])
    drift = np.array([abs(c["live_margin"] - c["cached_margin"]) for c in census.values()])
    thr = pose.MARGIN_THRESHOLD
    print(f"\ncached-vs-live margin drift (T2's observation): "
          f"median {np.median(drift):.3f}  p90 {np.percentile(drift, 90):.3f}  "
          f"max {drift.max():.3f}")
    print(f"live margins: {(np.abs(lm - thr) <= args.band).sum()} within "
          f"{args.band:g} of the {thr} gate, {(lm <= args.band).sum()} within "
          f"{args.band:g} of 0")

    # --- phase 2: compile-compare the live-exposed set ----------------------
    exposed = [(p, c) for p, c in census.items()
               if abs(c["live_margin"] - thr) <= args.band
               or c["live_margin"] <= args.band]
    print(f"\ncompare: {len(exposed)} live-exposed models")
    results, t0 = [], time.perf_counter()
    for i, (p, c) in enumerate(sorted(exposed, key=lambda t: t[1]["live_margin"])):
        try:
            (ie, me), (ic, mc) = ensemble(Path(p), [model.get_image_features, compiled_feat])
        except Exception as e:
            print(f"  skip {Path(p).stem}: {e}", file=sys.stderr)
            continue
        row = {"file": Path(p).stem, "census_margin": c["live_margin"],
               "eager_margin": float(me), "compiled_margin": float(mc),
               "up_flip": bool(ie != ic),
               "gate_flip": bool(pose.needs_arbiter_margin(me)
                                 != pose.needs_arbiter_margin(mc))}
        results.append(row)
        if row["up_flip"] or row["gate_flip"]:
            kind = "UP" if row["up_flip"] else "GATE"
            print(f"  {kind} FLIP {row['file']}: eager {me:.4f} vs compiled {mc:.4f}")
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(exposed)}  ({time.perf_counter() - t0:.0f}s)")

    em = np.array([r["eager_margin"] for r in results])
    dm = np.array([abs(r["eager_margin"] - r["compiled_margin"]) for r in results])
    in_gate = np.abs(em - thr) <= args.band
    in_up = em <= args.band
    print(f"\n{len(results)} compared: "
          f"{sum(r['gate_flip'] for r in results)} gate flips "
          f"({int(in_gate.sum())} still in gate band at compare time), "
          f"{sum(r['up_flip'] for r in results)} up flips "
          f"({int(in_up.sum())} in up band)")
    print(f"margin |delta| eager vs compiled: median {np.median(dm):.2e} "
          f"max {dm.max():.2e}")
    out = OUT / "compile_pose_flips.json"
    out.write_text(json.dumps({"threshold": thr, "band": args.band,
                               "results": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
