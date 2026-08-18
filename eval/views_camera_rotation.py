"""I11: how must `renderer.views` carry the up-rotation — in the camera, or by
rotating the mesh? Measured 2026-08-17; the answer changed the interfaces note.

The draft rule was "rotate into the camera, never `mesh.rotate`", on the theory
that residency only pays if the resident geometry is reusable as-is. `R.T` was
proven pixel-identical only for the *pose tile* grid at one elevation
(render_up_candidate_grid), while the classification views span 8 azimuths x 2
elevations, and the roundtrip spike that produced the residency numbers
*rotated held meshes* (eval/overlap_spike.py:101-103). This script tested the
draft rule and it failed: the ambient fill is a world-fixed environment map, so
rotating the rig lights the geometry differently, and every cached embedding
was computed with the mesh rotated. `views` now rotates a **copy** of the
resident mesh (interfaces.md §Render child), and the third arm below is the
proof that the reworked `views` reproduces the reference path byte for byte.

Method (harness pattern from eval/render_determinism.py): Filament's default
post-processing is temporally dithered — repeats of the *same* render differ
on ~43% of pixels — so byte-identity is only meaningful with
`set_post_processing(False)` (byte-stable after one warm-up frame; review V1).
Arms:

  nopost   post-processing off, production lighting — the real I11 test
  noibl    nopost + indirect light off (sun only) — attribution: the ambient
           fill is a world-fixed environment map (classify_stls.py:61-69), so
           if the camera trick differs only through it, this arm is identical
  default  production config, one model — places both deltas against the
           repeat noise production already carries

Three render paths per model x candidate up:

  camera   the rejected rule, kept so the finding stays reproducible:
           `_show` + `_shoot_rotated`, i.e. what `views` used to be
  views    production today — `Renderer.views`, a rotated copy per visit,
           run twice (fresh `LoadedMesh`, then residency hit). The repeat is
           doing two jobs: a mutated resident original would come back
           double-rotated, and a case that is not byte-stable against *itself*
           is the renderer's own floor rather than a path difference (the
           add/remove churn here is heavier than render_determinism.py's
           steady state; expect a stray 1/255 pixel on a case or two)
  mesh     the reference: `mesh.rotate` on a copy + `render_views`' framing,
           uploaded on its own; repeated so repeat-stability is established
           before any comparison

Usage:
  .venv/bin/python eval/views_camera_rotation.py [--render-size 384]
      [--views 8] [--elevations 20,-20] [--stl-dir test-stls]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from common import OUT, REPO

from src import loader
from src import pose
from src.messages import RenderConfig
from src.renderer import Renderer, orbit_camera, rotation_to_z_up, view_angles


def configure(r, arm):
    view = r._renderer.scene.view
    view.set_post_processing(arm == "default")
    r._renderer.scene.scene.enable_indirect_light(arm != "noibl")


def render_camera_path(r, lm, index, up, angles):
    """The rejected rule, reconstructed: the resident geometry as loaded, with
    the up-rotation carried in the camera (`R.T`). This is what `views` was
    before I11 — `_show` + `_shoot_rotated`, which `pose_tiles` still uses."""
    center, radius = r._show(lm, index, pin=False)
    R = rotation_to_z_up(np.asarray(up, dtype=float))
    return np.stack(r._shoot_rotated(R, center, radius, angles))


def render_views_path(r, lm, index, up):
    """Production: `Renderer.views` — a rotated copy of the resident mesh."""
    return np.stack(r.views(lm, index, up))


def render_mesh_path(r, mesh, up, angles):
    """The reference: mesh.rotate + render_views' framing + plain cams."""
    rm = o3d.geometry.TriangleMesh(mesh)                  # copy; keep original
    rm.rotate(rotation_to_z_up(np.asarray(up, dtype=float)), center=(0, 0, 0))
    scene = r._renderer.scene
    if r._visible is not None:                            # hide the resident
        scene.show_geometry(r._visible, False)            # geometry so only
        r._visible = None                                 # the copy renders
    scene.add_geometry("__ref__", rm, r._material)
    bounds = rm.get_axis_aligned_bounding_box()
    center = np.asarray(bounds.get_center(), dtype=float)
    radius = float(np.linalg.norm(bounds.get_extent()) * 1.4)
    cams = [(center, *orbit_camera(center, radius, az, elev))
            for az, elev in angles]
    images = np.stack(r._shoot(cams))
    scene.remove_geometry("__ref__")
    return images


def sha(a):
    return hashlib.sha1(a.tobytes()).hexdigest()


def delta(a, b):
    d = np.abs(a.astype(int) - b.astype(int))
    return int(d.max()), float((d > 0).mean()), int((d > 0).sum())


def one_case(r, lm, index, up, angles):
    cam1 = render_camera_path(r, lm, index, up, angles)   # fresh upload
    cam2 = render_camera_path(r, None, index, up, angles)  # resident re-show
    cam3 = render_camera_path(r, None, index, up, angles)  # re-show repeat
    new1 = render_views_path(r, lm, index, up)            # production, cold
    new2 = render_views_path(r, None, index, up)          # residency hit
    ref1 = render_mesh_path(r, lm.mesh, up, angles)
    ref2 = render_mesh_path(r, lm.mesh, up, angles)
    cam_max, cam_frac, _ = delta(cam2, ref1)
    new_max, new_frac, new_px = delta(new1, ref1)
    rep_max, _, rep_px = delta(new1, new2)
    return {"up": [float(x) for x in up],
            "cam_repeat_stable": sha(cam2) == sha(cam3),
            "upload_vs_reshow_max_delta": delta(cam1, cam2)[0],
            "ref_repeat_stable": sha(ref1) == sha(ref2),
            # the rejected rule, against the reference
            "camera_identical": sha(cam2) == sha(ref1),
            "camera_max_delta": cam_max, "camera_frac_differing": cam_frac,
            # production, against the same reference: must be byte-identical
            "views_identical": sha(new1) == sha(ref1),
            "views_max_delta": new_max, "views_frac_differing": new_frac,
            "views_diff_pixels": new_px,
            # the same call, twice: the residency hit must not change a pixel
            # (a mutated resident original would come back double-rotated), and
            # a case that is not stable against *itself* cannot be read as a
            # path difference — that is the renderer's own floor
            "views_repeat_stable": sha(new1) == sha(new2),
            "views_repeat_max_delta": rep_max,
            "views_repeat_diff_pixels": rep_px}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-size", type=int, default=384)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--elevations", default="20,-20")
    ap.add_argument("--stl-dir", default=str(REPO / "test-stls"))
    args = ap.parse_args()
    elevations = tuple(float(e) for e in args.elevations.split(","))
    stls = sorted(Path(args.stl_dir).glob("*.stl"))
    if not stls:
        raise SystemExit(f"no STLs in {args.stl_dir}")

    cfg = RenderConfig(render_size=args.render_size, views=args.views,
                       elevations=elevations, save_renders_dir=None,
                       render_format="png", budget_bytes=2 << 30,
                       collection_root=Path(args.stl_dir))
    r = Renderer(cfg)
    angles = view_angles(args.views, list(elevations))
    n_views = len(angles)
    meshes = [loader.get(f) for f in stls]

    results, warm_idx = {}, 10_000
    for arm in ("nopost", "noibl"):
        configure(r, arm)
        render_views_path(r, meshes[0], warm_idx, pose.UP_CANDIDATES[0])
        r.release(warm_idx)
        warm_idx += 1                                     # warm-up frame (V1)
        rows = []
        print(f"\n--- arm: {arm} ---")
        for mi, (f, lm) in enumerate(zip(stls, meshes)):
            for ci, up in enumerate(pose.UP_CANDIDATES):
                row = one_case(r, lm, 100 * warm_idx + mi * 10 + ci, up, angles)
                row["file"] = f.stem
                rows.append(row)
                print(f"{f.stem:20s} up={np.array(up, int)}  "
                      f"stable cam={row['cam_repeat_stable']} "
                      f"ref={row['ref_repeat_stable']}  "
                      f"views={'IDENTICAL' if row['views_identical'] else 'DIFFERS'}"
                      f" (repeat "
                      f"{'==' if row['views_repeat_stable'] else '!='})"
                      "  camera=" + ("IDENTICAL" if row["camera_identical"] else
                                     f"DIFFERS max {row['camera_max_delta']} on "
                                     f"{row['camera_frac_differing']:.1%} px"))
        results[arm] = rows

    # --- default arm: production repeat noise for scale -------------------
    configure(r, "default")
    lm, up = meshes[0], pose.UP_CANDIDATES[2]
    render_views_path(r, lm, 30_000, pose.UP_CANDIDATES[0])   # warm-up
    r.release(30_000)
    new1 = render_views_path(r, lm, 30_001, up)
    new2 = render_views_path(r, None, 30_001, up)
    cam1 = render_camera_path(r, None, 30_001, up, angles)
    ref1 = render_mesh_path(r, lm.mesh, up, angles)
    noise_max, noise_frac, _ = delta(new1, new2)
    cvm_max, cvm_frac, _ = delta(cam1, ref1)
    vvm_max, vvm_frac, _ = delta(new1, ref1)
    results["default"] = {"file": stls[0].stem,
                          "repeat_noise_max": noise_max,
                          "repeat_noise_frac": noise_frac,
                          "cam_vs_mesh_max": cvm_max,
                          "cam_vs_mesh_frac": cvm_frac,
                          "views_vs_mesh_max": vvm_max,
                          "views_vs_mesh_frac": vvm_frac}
    print(f"\ndefault post-processing, {stls[0].stem}: repeat noise "
          f"max {noise_max} on {noise_frac:.1%} px; camera-vs-mesh "
          f"max {cvm_max} on {cvm_frac:.1%} px; views-vs-mesh "
          f"max {vvm_max} on {vvm_frac:.1%} px")

    n = len(results["nopost"])
    rows = results["nopost"]
    identical = sum(row["views_identical"] for row in rows)
    # a case whose own repeat is unstable is the renderer's floor, not a path
    # difference — read it out separately rather than counting it either way.
    # A case is excludable only while its path delta stays within its own
    # repeat floor (A-R1-5): a real difference riding on an unstable case
    # must still fail.
    def within_floor(row):
        return (not row["views_repeat_stable"]
                and row["views_max_delta"] <= row["views_repeat_max_delta"])
    unstable = [row for row in rows if within_floor(row)]
    fails = [row for row in rows
             if not row["views_identical"] and not within_floor(row)]
    verdict = "PASS" if not fails else f"FAIL ({len(fails)}/{n} differ)"
    cam_fails = sum(not row["camera_identical"] for row in rows)
    noibl_cam_fails = sum(not row["camera_identical"] for row in results["noibl"])
    print(f"\nviews (rotated copy) vs mesh.rotate, nopost, {n} model x up "
          f"cases, {n_views} views each: {verdict}")
    print(f"  byte-identical to the reference: {identical}/{n}; "
          f"repeat-stable against itself: {n - len(unstable)}/{n}")
    for row in unstable:
        print(f"  floor: {row['file']} up={np.array(row['up'], int)} "
              f"repeat differs max {row['views_repeat_max_delta']} on "
              f"{row['views_repeat_diff_pixels']} px, vs reference max "
              f"{row['views_max_delta']} on {row['views_diff_pixels']} px "
              f"of {n_views * args.render_size ** 2 * 3:,}")
    print(f"the rejected rule (rotation in the camera): {cam_fails}/{n} differ; "
          f"with indirect light off {noibl_cam_fails}/{n}")
    out = OUT / "views_camera_rotation.json"
    out.write_text(json.dumps({"verdict": verdict, "results": results},
                              indent=2))
    print(f"wrote {out}")
    # exit without interpreter teardown: teardown would destroy the live
    # OffscreenRenderer, the one hard-constraint abort (CLAUDE.md)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
