"""Does the fp16 scoring matmul actually flip any classification?

The production scorer casts cached fp32 embeddings *down* to fp16 to multiply
against the text embeddings on the GPU (`classify_stls.py:1078,1104`), while
the pose ensemble runs the same kind of matmul in fp32 numpy. torch.compile
was disqualified because its embedding drift (max 7.3-9.8e-04) matched the
closest observed top-1 margin (9e-04, 128 renders, 0 flips) — but that was a
sample, and the fp16 cast is a perturbation of the same order applied to every
model on every run. This harness scores every cached embedding both ways and
counts what actually moves:

  fp16-gpu   the production path, byte for byte: .to(device, fp16), matmul,
             .float().cpu().numpy(), pool
  fp32-cpu   the same embeddings and text embeddings in fp32 numpy
  fp64-cpu   fp32 inputs, fp64 accumulation — shows whether fp32 is converged

Reported per pooling mode (all three, since the pool is applied after the
matmul and costs nothing extra): top-1 flips, top-3 set and order changes, the
margin (top1-top2) distribution, and how many models sit with a margin inside
the drift band — the population a compile-scale perturbation could flip.

No cache is written. Needs the GPU for the text embeddings and the production
path.

Usage:
  .venv/bin/python eval/score_precision.py [--cache-dir embed-cache2]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from common import OUT  # puts REPO on sys.path

import torch
from src.cachedir import add_cache_args, apply_run_params, embeds_dir
from src.embedder import embed_texts
from src.query import pool_sims

DRIFT = 9.8e-4          # torch.compile's max embedding drift, LEARNINGS


def main():
    parser = argparse.ArgumentParser()
    add_cache_args(parser, "unused — this reads only the embedding cache")
    parser.add_argument("--categories", default="categories.txt")
    parser.add_argument("--pool", choices=["mean", "max", "softmax"],
                        default="softmax")   # the production default
    args = apply_run_params(parser)
    edir = embeds_dir(args.cache_dir)
    files = sorted(edir.glob("*.npy"))
    if not files:
        sys.exit(f"no embeddings under {edir}")
    categories = [l.strip() for l in open(args.categories) if l.strip()]
    print(f"{len(files)} cached models, {len(categories)} categories, "
          f"run-params pool={args.pool}")

    from transformers import AutoModel, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    with torch.no_grad():
        text_embeds = embed_texts(model, processor, categories, device)  # fp16, device
    t32 = text_embeds.float().cpu().numpy()

    modes = ["mean", "max", "softmax"]
    stats = {m: {"top1_flips": [], "top3_set": 0, "top3_order": 0,
                 "margins32": [], "fp64_top1_flips": 0} for m in modes}
    max_dsim = 0.0
    for f in files:
        e32 = np.load(f)
        # production path, byte for byte
        e16 = torch.from_numpy(e32).to(device, dtype=text_embeds.dtype)
        vs16 = (e16 @ text_embeds.T).float().cpu().numpy()
        vs32 = e32 @ t32.T
        vs64 = e32.astype(np.float64) @ t32.astype(np.float64).T
        max_dsim = max(max_dsim, float(np.abs(vs16 - vs32).max()))
        for m in modes:
            s16, s32, s64 = (pool_sims(v, m) for v in (vs16, vs32, vs64))
            o16, o32, o64 = s16.argsort()[::-1], s32.argsort()[::-1], s64.argsort()[::-1]
            st = stats[m]
            if o16[0] != o32[0]:
                st["top1_flips"].append(
                    {"file": f.stem, "fp16": categories[o16[0]],
                     "fp32": categories[o32[0]],
                     "margin32": float(s32[o32[0]] - s32[o32[1]])})
            st["top3_set"] += int(set(o16[:3].tolist()) != set(o32[:3].tolist()))
            st["top3_order"] += int(o16[:3].tolist() != o32[:3].tolist())
            st["fp64_top1_flips"] += int(o32[0] != o64[0])
            st["margins32"].append(float(s32[o32[0]] - s32[o32[1]]))

    report = {"models": len(files), "categories": len(categories),
              "max_view_sim_delta_fp16_vs_fp32": max_dsim, "modes": {}}
    for m in modes:
        st = stats[m]
        marg = np.array(st["margins32"])
        report["modes"][m] = {
            "top1_flips": len(st["top1_flips"]),
            "top3_set_changes": st["top3_set"],
            "top3_order_changes": st["top3_order"],
            "fp32_vs_fp64_top1_flips": st["fp64_top1_flips"],
            "min_margin": float(marg.min()),
            "margin_p1": float(np.percentile(marg, 1)),
            "margin_median": float(np.median(marg)),
            "models_with_margin_below_drift": int((marg < DRIFT).sum()),
            "models_with_margin_below_2x_drift": int((marg < 2 * DRIFT).sum()),
            "flips": st["top1_flips"][:20],
        }
        r = report["modes"][m]
        tag = " <- production" if m == args.pool else ""
        print(f"\n{m}{tag}: top1 flips {r['top1_flips']}/{len(files)}  "
              f"top3 set {r['top3_set_changes']}  order {r['top3_order_changes']}  "
              f"fp32-vs-fp64 flips {r['fp32_vs_fp64_top1_flips']}")
        print(f"  fp32 margin: min {r['min_margin']:.2e}  p1 {r['margin_p1']:.2e}  "
              f"median {r['margin_median']:.3f}  "
              f"< drift({DRIFT:.1e}): {r['models_with_margin_below_drift']}  "
              f"< 2x: {r['models_with_margin_below_2x_drift']}")
        for fl in st["top1_flips"][:5]:
            print(f"    flip {fl['file'][:12]}: {fl['fp16']!r} <-> {fl['fp32']!r} "
                  f"(fp32 margin {fl['margin32']:.2e})")
    print(f"\nmax per-view sim delta fp16 vs fp32: {max_dsim:.2e}")

    out = OUT / "score_precision.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
