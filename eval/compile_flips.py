"""What does torch.compile actually flip?

score_precision.py bounded the question: 124 of 2943 models hold production
top-1 margins below torch.compile's measured embedding drift (max 9.8e-04),
249 below twice it. This harness answers it directly for the population that
matters: for each at-risk model (fp32 margin < 2x drift) plus a random
control, load the 16 saved production renders, preprocess ONCE, and push the
identical tensors through eager SigLIP and torch.compile'd SigLIP. Same
pixels, same preprocessing, one variable. Score both with the production
softmax pool and report the flips by name.

The at-risk set is scored worst-margin-first so the likeliest flips print
early. Selection margins come from the cached fp32 embeddings (fp32 = fp64
on every model, so they are the converged reference); the eager-vs-compiled
comparison is then internally consistent on the saved jpgs regardless of jpg
loss or render nondeterminism.

No cache is written. ~350 models x 16 views x 2 towers at ~18 img/s is
~10 minutes plus ~1 minute of compile.

Usage:
  .venv/bin/python eval/compile_flips.py [--cache-dir embed-cache2]
      [--risk-margin 19.6e-4] [--control 100]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from common import OUT  # puts REPO on sys.path

import torch
from src import pose
from classify_stls import (add_cache_args, apply_run_params, as_tensor,
                           cache_key, cache_root, embed_texts, embeds_dir,
                           load_file_list, pool_sims, render_key, renders_dir)

DRIFT = 9.8e-4          # torch.compile's max embedding drift, LEARNINGS


def main():
    parser = argparse.ArgumentParser()
    add_cache_args(parser, "unused — reads the caches and saved renders")
    parser.add_argument("--categories", default="categories.txt")
    parser.add_argument("--pool", choices=["mean", "max", "softmax"],
                        default="softmax")   # the production default
    parser.add_argument("--risk-margin", type=float, default=2 * DRIFT)
    parser.add_argument("--control", type=int, default=100)
    args = apply_run_params(parser)
    root = cache_root(Path(args.input), args.cache_dir, confirm=False)
    files = load_file_list(Path(args.input), args.cache_dir)
    poses = pose.load_pose_cache(args.cache_dir)
    edir = embeds_dir(args.cache_dir)
    rdir = renders_dir(args.cache_dir, args)
    categories = [l.strip() for l in open(args.categories) if l.strip()]

    # a file qualifies when its cached embedding and all 16 saved views exist
    work = []
    for f in files:
        entry = poses.get(pose.file_identity(f, root))
        if entry is None and args.up_axis == "auto":
            continue
        token = pose.embed_cache_token(entry or {}, args.up_axis)
        npy = edir / f"{cache_key(f, args, token, root)}.npy"
        if not npy.exists():
            continue
        rkey = render_key(f, root)
        views = sorted(rdir.glob(f"{rkey}_view*.*"))
        if len(views) != args.views * len(args.elevations):
            continue
        work.append((f, npy, views))
    print(f"{len(work)} models with cached embeddings and full render sets")

    from transformers import AutoModel, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    with torch.no_grad():
        t32 = embed_texts(model, processor, categories, device).float().cpu().numpy()

    # margins from the cached fp32 embeddings — the converged reference
    def margin_of(npy):
        s = pool_sims(np.load(npy) @ t32.T, args.pool)
        o = s.argsort()[::-1]
        return float(s[o[0]] - s[o[1]])

    margins = [(margin_of(npy), f, views) for f, npy, views in work]
    at_risk = sorted(m for m in margins if m[0] < args.risk_margin)
    rest = [m for m in margins if m[0] >= args.risk_margin]
    control = rest[:: max(1, len(rest) // args.control)][:args.control]
    print(f"{len(at_risk)} at-risk (margin < {args.risk_margin:.1e}), "
          f"{len(control)} control")

    # compile the bound method, as siglip_bench did — wrapping the whole model
    # only intercepts forward(), and get_image_features would silently stay
    # eager (first run of this harness: 0.0 drift, 1 s "compile")
    compiled_feat = torch.compile(model.get_image_features)

    @torch.no_grad()
    def both(views):
        """Identical preprocessed tensors through both towers."""
        imgs = [Image.open(v).convert("RGB") for v in views]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        out = []
        for fn in (model.get_image_features, compiled_feat):
            feat = as_tensor(fn(**inputs))
            feat = torch.nn.functional.normalize(feat, dim=-1)
            out.append(feat.float().cpu().numpy())
        return out

    t0 = time.perf_counter()
    both(at_risk[0][2])   # compile warmup outside the loop
    print(f"compile warmup {time.perf_counter() - t0:.0f}s")

    results = []
    for group, batch in (("at-risk", at_risk), ("control", control)):
        for i, (marg, f, views) in enumerate(batch):
            eager, compiled = both(views)
            drift = float(np.abs(eager - compiled).max())
            se = pool_sims(eager @ t32.T, args.pool)
            sc = pool_sims(compiled @ t32.T, args.pool)
            oe, oc = se.argsort()[::-1], sc.argsort()[::-1]
            flip = bool(oe[0] != oc[0])
            results.append({"group": group, "file": f.stem, "cached_margin": marg,
                            "drift": drift, "flip": flip,
                            "eager_margin": float(se[oe[0]] - se[oe[1]]),
                            "eager": categories[oe[0]], "compiled": categories[oc[0]]})
            if flip:
                print(f"  FLIP [{group}] {f.stem}: {categories[oe[0]]!r} -> "
                      f"{categories[oc[0]]!r} (eager margin "
                      f"{se[oe[0]] - se[oe[1]]:.2e}, drift {drift:.1e})")
            if (i + 1) % 50 == 0:
                print(f"  {group}: {i + 1}/{len(batch)}")

    for group in ("at-risk", "control"):
        rs = [r for r in results if r["group"] == group]
        flips = [r for r in rs if r["flip"]]
        drifts = np.array([r["drift"] for r in rs])
        print(f"\n{group}: {len(flips)}/{len(rs)} top-1 flips  "
              f"drift max {drifts.max():.1e} median {np.median(drifts):.1e}")
    out = OUT / "compile_flips.json"
    out.write_text(json.dumps({"pool": args.pool, "risk_margin": args.risk_margin,
                               "results": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
