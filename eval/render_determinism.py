"""Which stage makes pose resolution nondeterministic — the renderer or the
tower?

Review U2: two identical eager passes move a model's margin by a median of
2.66e-02 and up to 2.69e-01 — gate-crossing scale, mechanism unidentified.
The candidates separate cheaply in one process:

  pixels   render the same loaded mesh's candidate grid R times and hash the
           tile bytes. Different hashes => Filament is nondeterministic.
  tower    embed one repeat's tiles twice from the SAME preprocessed tensors.
           Nonzero delta => fp16 kernel variance contributes too.
  margin   the full ensemble per repeat (geometry computed once — it is
           seeded) => how far the margin and the up pick move on identical
           input, per stage.

Models: the tightest live margins from the census (out/pose_live_margins.json,
written by compile_pose_flips.py) plus mid-distribution controls.

Usage:
  .venv/bin/python eval/render_determinism.py [--cache-dir embed-cache2]
      [--models 6] [--repeats 3]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from common import OUT  # puts REPO on sys.path

import torch
import pose
from classify_stls import (add_cache_args, apply_run_params, as_tensor,
                           embed_raw, load_mesh, make_renderer,
                           render_up_candidate_grid)


def main():
    parser = argparse.ArgumentParser()
    add_cache_args(parser, "unused — models come from the live-margin census")
    parser.add_argument("--models", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    args = apply_run_params(parser)
    census_path = OUT / "pose_live_margins.json"
    if not census_path.exists():
        sys.exit("no census — run eval/compile_pose_flips.py first")
    census = json.loads(census_path.read_text())
    ranked = sorted(census.items(), key=lambda kv: kv[1]["live_margin"])
    half = args.models // 2
    picks = ranked[:half] + ranked[len(ranked) // 2::len(ranked) // (args.models - half)][:args.models - half]

    from transformers import AutoModel, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    with torch.no_grad():
        up_T = embed_raw(model, processor, pose.UPRIGHT_PROMPTS, device).float().cpu().numpy()
        down_T = embed_raw(model, processor, pose.TOPPLED_PROMPTS, device).float().cpu().numpy()
    renderer = make_renderer(args.render_size)

    @torch.no_grad()
    def embed(inputs):
        feat = as_tensor(model.get_image_features(**inputs))
        return torch.nn.functional.normalize(feat, dim=-1).float().cpu().numpy()

    rows = []
    for p, c in picks:
        mesh = load_mesh(Path(p))
        geo = pose.up_axis_scores(mesh)          # seeded — computed once
        stacks, margins, idxs = [], [], []
        for _ in range(args.repeats):
            grid = render_up_candidate_grid(renderer, mesh)
            flat = [im for row in grid for im in row]
            arr = np.stack([np.asarray(im) for im in flat])
            stacks.append(arr)
            inputs = processor(images=flat, return_tensors="pt").to(device)
            sig = pose.upright_scores(embed(inputs), up_T, down_T) \
                      .reshape(len(grid), -1).mean(axis=1)
            idx, margin = pose.combine_up(geo, sig)
            idxs.append(int(idx)); margins.append(float(margin))
        hashes = [hashlib.sha1(s.tobytes()).hexdigest()[:12] for s in stacks]
        pix_delta = max(int(np.abs(stacks[0].astype(int) - s.astype(int)).max())
                        for s in stacks[1:])
        pix_frac = max(float((stacks[0] != s).mean()) for s in stacks[1:])
        # the tower on byte-identical tensors, twice
        inputs = processor(images=[im for im in stacks[0]], return_tensors="pt").to(device)
        tower_delta = float(np.abs(embed(inputs) - embed(inputs)).max())
        rows.append({"file": Path(p).stem, "live_margin": c["live_margin"],
                     "pixels_identical": len(set(hashes)) == 1,
                     "pixel_max_delta": pix_delta, "pixel_frac_diff": pix_frac,
                     "tower_max_delta": tower_delta,
                     "margin_spread": max(margins) - min(margins),
                     "picks": idxs, "pick_stable": len(set(idxs)) == 1})
        r = rows[-1]
        print(f"{r['file'][:36]:36s} pixels {'identical' if r['pixels_identical'] else f'DIFFER (max {pix_delta}/255, {pix_frac:.1%} of px)':32s} "
              f"tower {tower_delta:.1e}  margin spread {r['margin_spread']:.2e}  "
              f"picks {idxs}{'' if r['pick_stable'] else '  <- UNSTABLE'}")

    print(f"\npixels identical on {sum(r['pixels_identical'] for r in rows)}/{len(rows)} "
          f"models; tower repeat delta max {max(r['tower_max_delta'] for r in rows):.1e}; "
          f"margin spread max {max(r['margin_spread'] for r in rows):.2e}")
    out = OUT / "render_determinism.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
