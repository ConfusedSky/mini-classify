"""Compare fill strategies on the real mesh: indirect (IBL) fill vs multi-pass sun."""
import sys
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from PIL import Image

from classify_stls import load_mesh, rotation_to_z_up
import pose

from common import OUT, AX, IDX, load_labels, mark, score

OUT = str(OUT)  # from common
STL = "/run/media/masa/Files and S/STL/DM Stash/Crimson Masquerade/Arkham Ravenswood - Unsupported/32_Unsupported_Arkham_BodyMask.stl"
SIZE = 512

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


def aim(az_deg, elev_deg=20.0):
    az, e = np.deg2rad(az_deg), np.deg2rad(elev_deg)
    eye = center + radius * np.array([np.cos(az)*np.cos(e), np.sin(az)*np.cos(e), np.sin(e)])
    up = np.array([-np.cos(az)*np.sin(e), -np.sin(az)*np.sin(e), np.cos(e)])
    r.setup_camera(45.0, center, eye, up)
    fwd = (center - eye) / np.linalg.norm(center - eye)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    return fwd, right, up


def render(sun_dir, intensity, ibl=None):
    r.scene.scene.enable_indirect_light(ibl is not None)
    if ibl is not None:
        r.scene.scene.set_indirect_light_intensity(ibl)
    r.scene.scene.set_sun_light(n(sun_dir), [1.0, 1.0, 1.0], intensity)
    return np.asarray(r.render_to_image()).astype(np.float32)


def multipass(fwd, right, up, weights):
    """Weighted sum of sun passes from camera-relative directions."""
    acc = np.zeros((SIZE, SIZE, 3), np.float32)
    for d, w in weights:
        acc += w * render(d, 90000, ibl=None)
    return np.clip(acc, 0, 255)


VARIANTS = {}
fwd, right, up = aim(270.0)
key = fwd + [0, 0, -0.6]

VARIANTS["A current"] = render(key, 90000, ibl=None)
for i in (10000, 20000, 30000, 45000):
    VARIANTS[f"IBL {i//1000}k"] = render(key, 90000, ibl=i)
VARIANTS["IBL 30k, sun 60k"] = render(key, 60000, ibl=30000)
VARIANTS["3-pass sun"] = multipass(fwd, right, up, [
    (key, 0.6), (fwd + up * 0.8, 0.25), (fwd - right * 1.0 + up * 0.2, 0.15)])
VARIANTS["2-pass sun"] = multipass(fwd, right, up, [(key, 0.7), (fwd + up * 0.9, 0.3)])

print(f"{'variant':20} {'mean':>7} {'p2':>6} {'frac<20':>8} {'frac<40':>8}")
for name, img in VARIANTS.items():
    g = np.asarray(Image.fromarray(img.astype(np.uint8)).convert("L"), float)
    o = g[g < 230]
    print(f"{name:20} {o.mean():7.1f} {np.percentile(o,2):6.1f} {(o<20).mean():8.3f} {(o<40).mean():8.3f}")

# full-frame sheet + head crops
names = list(VARIANTS)
sheet = Image.new("RGB", (SIZE * len(names), SIZE + 260), (255, 255, 255))
for i, k in enumerate(names):
    im = Image.fromarray(VARIANTS[k].astype(np.uint8)).convert("RGB")
    sheet.paste(im, (SIZE * i, 0))
    sheet.paste(im.crop((150, 40, 280, 170)).resize((260, 260), Image.NEAREST), (SIZE * i, SIZE))
sheet.save(f"{OUT}/fills.png")
print("wrote", f"{OUT}/fills.png", "order:", names)

# view-to-view bias check: does IBL make brightness swing with azimuth?
print("\nazimuth brightness swing (object-pixel mean):")
for label, ibl in [("no IBL", None), ("IBL 30k", 30000)]:
    ms = []
    for azd in (0, 90, 180, 270):
        f2, r2, u2 = aim(azd)
        g = np.asarray(Image.fromarray(render(f2 + [0, 0, -0.6], 90000, ibl=ibl).astype(np.uint8)).convert("L"), float)
        ms.append(g[g < 230].mean())
    print(f"  {label:8} " + " ".join(f"{m:6.1f}" for m in ms) + f"   spread {max(ms)-min(ms):5.1f}")
