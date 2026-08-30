"""Arm: the production Open3D renderer, identical rig, but each view framed
tight to the rotated mesh's projected vertices (5% margin) instead of the
1.4x extent-norm orbit radius. Output renders/open3d_tight/<key>_v<i>.png"""
import json, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path.home() / "Documents/tests/mini-classify"
sys.path.insert(0, str(REPO))
import open3d as o3d
from src import loader, pose
from src.messages import RenderConfig
from src.renderer import ROTATED_NAME, Renderer, orbit_camera

HERE = Path(__file__).parent
OUT = HERE / "renders/open3d_tight"; OUT.mkdir(parents=True, exist_ok=True)
S = json.load(open(HERE / "sample.json"))
ONLY = sys.argv[1] if len(sys.argv) > 1 else None
if ONLY:
    S["sample"] = [m for m in S["sample"] if ONLY in m["key"]]
FOV, MARGIN = 45.0, 1.05
T = np.tan(np.radians(FOV / 2))


def tight_cams(verts, center, angles):
    cams = []
    rel = verts - center
    for az, el in angles:
        eye0, up, _ = orbit_camera(center, 1.0, az, el)
        d = eye0 - center                       # unit, center -> eye
        right = np.cross(-d, up); right /= np.linalg.norm(right)
        x = rel @ right; y = rel @ up; z = rel @ d
        dist = float(np.max(z + np.maximum(np.abs(x), np.abs(y)) / T)) * MARGIN
        eye, up, sun = orbit_camera(center, dist, az, el)
        cams.append((center, eye, up, sun))
    return cams


def main():
    cfg = RenderConfig(render_size=512, views=S["views"], elevations=tuple(S["elevations"]),
                       save_renders_dir=None, render_format="png", budget_bytes=1 << 30,
                       collection_root=Path(S["root"]))
    r = Renderer(cfg)
    t0 = time.time()
    for n, m in enumerate(S["sample"], 1):
        if all((OUT / f"{m['key']}_v{i}.png").exists() for i in range(len(S["angles"]))):
            continue
        lm = loader.get(m["path"])
        rm = r._admit(lm, n, pin=False)
        rot = o3d.geometry.TriangleMesh(rm.mesh)
        rot.rotate(pose.rotation_to_z_up(np.asarray(m["up"], dtype=float)), center=(0, 0, 0))
        verts = np.asarray(rot.vertices)
        if len(verts) > 200_000:
            verts = verts[np.random.default_rng(0).choice(len(verts), 200_000, replace=False)]
        center = np.asarray(rot.get_axis_aligned_bounding_box().get_center(), dtype=float)
        r._hide_visible()
        scene = r._renderer.scene
        scene.add_geometry(ROTATED_NAME, rot, r._material)
        try:
            imgs = r._shoot(tight_cams(verts, center, S["angles"]))
        finally:
            scene.remove_geometry(ROTATED_NAME)
        for i, im in enumerate(imgs):
            Image.fromarray(im).save(OUT / f"{m['key']}_v{i}.png", compress_level=1)
        r.resident.clear()
        if n % 20 == 0:
            print(f"[{n}/{len(S['sample'])}] {time.time()-t0:.0f}s", flush=True)
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
