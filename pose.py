"""Pose resolution for STL renders: up-axis detection with confidence,
front-view scoring, a per-file pose cache, and a VLM arbiter for
ambiguous cases. classify_stls.py orchestrates; this module never
imports classify_stls (no rendering / model code here)."""
import json
from pathlib import Path

import numpy as np

UP_CANDIDATES = [np.array(u, dtype=float) for u in
                 [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]]

ABS_SCORE_FLOOR = 0.02  # best flat-base score below this = "no print base found"


def detect_up_axis(mesh, n_samples=4000):
    """Score every candidate up by flat print-base evidence: how much
    down-facing flat surface sits in the bottom 2% height slab.

    Returns (up, ratio, best_score) — ratio = runner_up/best, lower is
    more confident (symmetric meshes like a barrel come out ~1.0)."""
    pcd = mesh.sample_points_uniformly(n_samples, use_triangle_normal=True)
    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    scores = []
    for up in UP_CANDIDATES:
        h = pts @ up
        extent = h.max() - h.min()
        if extent <= 0:
            scores.append(0.0)
            continue
        in_bottom_slab = h < h.min() + 0.02 * extent
        facing_down = normals @ up < -0.9
        scores.append(float(np.mean(in_bottom_slab & facing_down)))
    order = np.argsort(scores)[::-1]
    best, runner = scores[order[0]], scores[order[1]]
    ratio = runner / best if best > 0 else 1.0
    return UP_CANDIDATES[order[0]], ratio, best


def needs_arbiter(ratio, best_score, threshold=0.6):
    return ratio > threshold or best_score < ABS_SCORE_FLOOR
