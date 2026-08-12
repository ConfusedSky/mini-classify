"""Head-crop comparison of the top fill candidates."""
import sys
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from PIL import Image, ImageDraw

from classify_stls import load_mesh, rotation_to_z_up
import pose

from common import OUT, AX, IDX, load_labels, mark, score

OUT = str(OUT)  # from common
STL = "/run/media/masa/Files and S/STL/DM Stash/Crimson Masquerade/Arkham Ravenswood - Unsupported/32_Unsupported_Arkham_BodyMask.stl"
SIZE = 512
CROP = (170, 45, 330, 205)   # head + torso

mesh = load_mesh(STL)
up_axis, _, _ = pose.detect_up_axis(mesh)
mesh.rotate(rotation_to_z_up(np.array(up_axis)), center=(0, 0, 0))

r = rendering.OffscreenRenderer(SIZE, SIZE)
r.scene.set_background([1.0, 1.0, 1.0, 1.0])
r.scene.scene.enable_sun_light(True)
mat = rendering.MaterialRecord(); mat.shader = "defaultLit"; mat.base_color = [0.7, 0.7, 0.7, 1.0]
r.scene.add_geometry("mesh", mesh, mat)
b = mesh.get_axis_aligned_bounding_box()
center, radius = b.get_center(), np.linalg.norm(b.get_extent()) * 1.4
n = lambda v: np.asarray(v, dtype=np.float32) / np.linalg.norm(v)

az, e = np.deg2rad(270.0), np.deg2rad(20.0)
eye = center + radius * np.array([np.cos(az)*np.cos(e), np.sin(az)*np.cos(e), np.sin(e)])
up = np.array([-np.cos(az)*np.sin(e), -np.sin(az)*np.sin(e), np.cos(e)])
r.setup_camera(45.0, center, eye, up)
fwd = (center - eye) / np.linalg.norm(center - eye)
right = np.cross(fwd, up); right /= np.linalg.norm(right)
key = fwd + [0, 0, -0.6]


def render(sun_dir, intensity, ibl=None):
    r.scene.scene.enable_indirect_light(ibl is not None)
    if ibl is not None:
        r.scene.scene.set_indirect_light_intensity(ibl)
    r.scene.scene.set_sun_light(n(sun_dir), [1.0, 1.0, 1.0], intensity)
    return np.asarray(r.render_to_image()).astype(np.float32)


def multipass(weights):
    acc = np.zeros((SIZE, SIZE, 3), np.float32)
    for d, w in weights:
        acc += w * render(d, 90000)
    return np.clip(acc, 0, 255)


V = {
    "A  current": render(key, 90000),
    "B  IBL 10k fill": render(key, 90000, ibl=10000),
    "C  IBL 20k fill": render(key, 90000, ibl=20000),
    "D  3-pass sun (camera-relative)": multipass(
        [(key, 0.6), (fwd + up * 0.8, 0.25), (fwd - right * 1.0 + up * 0.2, 0.15)]),
}

S = 480
grid = Image.new("RGB", (S * 2, (S + 26) * 2), (255, 255, 255))
d = ImageDraw.Draw(grid)
for i, (k, img) in enumerate(V.items()):
    im = Image.fromarray(img.astype(np.uint8)).convert("RGB").crop(CROP).resize((S, S), Image.LANCZOS)
    x, y = S * (i % 2), (S + 26) * (i // 2)
    d.text((x + 6, y + 7), k, fill=(0, 0, 0))
    grid.paste(im, (x, y + 26))
grid.save(f"{OUT}/heads.png")
print("wrote heads.png")
