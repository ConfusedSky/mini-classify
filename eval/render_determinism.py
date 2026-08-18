"""Which stage makes pose resolution nondeterministic — and does the
post-processing toggle fix it?

Review U2: two identical eager passes move a model's margin by a median of
2.66e-02 and up to 2.69e-01 — gate-crossing scale. Phase one of this harness
isolated the mechanism: Filament differs on every repeat (~43% of pixels by
2-28/255, the fingerprint of temporal dithering in the post-processing
chain), while the tower on byte-identical tensors is bit-deterministic.

Review V1 then measured `scene.view.set_post_processing(False)` making a
synthetic render byte-stable after one warm-up frame. This version runs the
candidate fixes as arms, per real model, in one process:

  default   production config — the baseline nondeterminism
  nopost    set_post_processing(False), one throwaway frame after the toggle
  noaa      nopost + set_antialiasing(False), for the outlier whose 28/255
            delta looks like a silhouette AA resolve, not dithering

Per arm: pixel hashes across repeats (identical or not), ensemble margin per
repeat, up pick per repeat, and the tower repeated on byte-identical tensors.
Across arms: the margin shift default -> fixed, which is the one-time cost
V1 warns about (every pixel changes, so every embedding and margin does — a
POSE_CACHE_VERSION and EMBED_CACHE_VERSION bump if adopted).

Models: the tightest live margins from the census (out/pose_live_margins.json,
written by compile_pose_flips.py) plus mid-distribution controls, with
32mm_Pipe5 (the AA-outlier and the model that flipped its up pick between
identical eager passes) pinned into the set.

Usage:
  .venv/bin/python eval/render_determinism.py [--cache-dir embed-cache2]
      [--models 6] [--repeats 3] [--arms default,nopost,noaa]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from common import OUT  # puts REPO on sys.path

import torch
import rig
from src import pose
from src.cachedir import add_cache_args, apply_run_params
from src.embedder import as_tensor

PINNED = ("32mm_Pipe5",)


def configure(r, arm):
    """Filament view flags, reached through the production Renderer's own
    OffscreenRenderer — the arms are exactly the toggles review V1 tested."""
    view = r._renderer.scene.view
    if arm in ("nopost", "noaa"):
        view.set_post_processing(False)
    else:
        view.set_post_processing(True)
    view.set_antialiasing(arm != "noaa")


def main():
    parser = argparse.ArgumentParser()
    add_cache_args(parser, "unused — models come from the live-margin census")
    parser.add_argument("--models", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--arms", default="default,nopost,noaa")
    args = apply_run_params(parser)
    census_path = OUT / "pose_live_margins.json"
    if not census_path.exists():
        sys.exit("no census — run eval/compile_pose_flips.py first")
    census = json.loads(census_path.read_text())
    ranked = sorted(census.items(), key=lambda kv: kv[1]["live_margin"])
    half = args.models // 2
    picks = dict(ranked[:half])
    for p, c in census.items():
        if Path(p).stem in PINNED:
            picks[p] = c
    step = max(1, len(ranked) // args.models)
    for p, c in ranked[len(ranked) // 2::step]:
        if len(picks) >= args.models + len(PINNED):
            break
        picks.setdefault(p, c)

    e = rig.embedder(args.model)
    model, processor, device = e.model, e.processor, e.device
    up_T, down_T = e.up_T, e.down_T
    renderer = rig.rig(args.render_size)

    @torch.no_grad()
    def embed(inputs):
        # the raw forward, not Embedder.embed_images: this phase re-embeds the
        # *same* preprocessed tensor to separate the tower's determinism from
        # the renderer's, so preprocessing has to stay outside the timed call
        feat = as_tensor(model.get_image_features(**inputs))
        return torch.nn.functional.normalize(feat, dim=-1).float().cpu().numpy()

    def one_pass(lm, geo):
        grid = rig.pose_tiles(renderer, lm)
        flat = [im for row in grid for im in row]
        arr = np.stack(flat)
        inputs = processor(images=flat, return_tensors="pt").to(device)
        sig = pose.upright_scores(embed(inputs), up_T, down_T) \
                  .reshape(len(grid), -1).mean(axis=1)
        idx, margin = pose.combine_up(geo, sig)
        return arr, int(idx), float(margin)

    arms = [a.strip() for a in args.arms.split(",")]
    rows = []
    for p, c in picks.items():
        lm = rig.load(Path(p))
        geo = pose.up_axis_scores(lm.mesh)       # seeded — computed once
        per_arm = {}
        for arm in arms:
            configure(renderer, arm)
            one_pass(lm, geo)                    # throwaway: the warm-up frame
            stacks, idxs, margins = [], [], []
            for _ in range(args.repeats):
                arr, idx, margin = one_pass(lm, geo)
                stacks.append(arr); idxs.append(idx); margins.append(margin)
            hashes = {hashlib.sha1(s.tobytes()).hexdigest() for s in stacks}
            pix_delta = max((int(np.abs(stacks[0].astype(int) - s.astype(int)).max())
                             for s in stacks[1:]), default=0)
            per_arm[arm] = {"identical": len(hashes) == 1,
                            "pixel_max_delta": pix_delta,
                            "margins": margins, "picks": idxs,
                            "margin_spread": max(margins) - min(margins),
                            "pick_stable": len(set(idxs)) == 1}
        configure(renderer, "default")           # leave the scene as found
        shift = abs(np.mean(per_arm[arms[-1]]["margins"])
                    - np.mean(per_arm["default"]["margins"])) if "default" in per_arm else None
        rows.append({"file": Path(p).stem, "live_margin": c["live_margin"],
                     "arms": per_arm, "margin_shift_default_to_fixed": shift})
        bits = "  ".join(
            f"{a}: {'stable' if r['identical'] else f'DIFFERS(max {r['pixel_max_delta']})'}"
            f"/spread {r['margin_spread']:.1e}{'' if r['pick_stable'] else '/PICK UNSTABLE'}"
            for a, r in per_arm.items())
        print(f"{Path(p).stem[:32]:32s} {bits}  shift {shift:.2e}")

    for arm in arms:
        n = sum(r["arms"][arm]["identical"] for r in rows)
        print(f"\n{arm}: pixels byte-identical on {n}/{len(rows)} models, "
              f"max margin spread {max(r['arms'][arm]['margin_spread'] for r in rows):.2e}")
    if "default" in arms:
        shifts = [r["margin_shift_default_to_fixed"] for r in rows]
        print(f"one-time margin shift default -> {arms[-1]}: "
              f"median {np.median(shifts):.2e} max {max(shifts):.2e}")
    out = OUT / "render_determinism.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")
    rig.exit_without_teardown()   # the live OffscreenRenderer must not be destroyed


if __name__ == "__main__":
    main()
