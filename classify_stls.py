"""Zero-shot STL classification: multiview renders scored against text categories with SigLIP.

Usage:
  python classify_stls.py /path/to/stls --categories categories.txt --out results.csv
  python classify_stls.py model.stl --save-renders renders/   # single file, keep debug renders

Renders each mesh from several viewpoints (Open3D offscreen), embeds the views
with SigLIP, averages them, and ranks against text embeddings of the categories.
"""
import argparse
import csv
import os
import hashlib
import json
import time
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import torch
from PIL import Image
from tqdm import tqdm

def as_tensor(feat):
    return feat if isinstance(feat, torch.Tensor) else feat.pooler_output


PROMPT_TEMPLATES = [
    "a 3D render of a {} miniature",
    "a photo of a {} figurine",
    "a tabletop miniature of a {}",
]


def make_renderer(size):
    # Meshes are rotated into Z-up world space before rendering, so this rig
    # (key light from above + fills) is correct for any --up-axis choice. The
    # built-in indirect/environment light has a fixed Y-up orientation and
    # shades a Z-up world from the side, so it is disabled.
    renderer = rendering.OffscreenRenderer(size, size)
    scene = renderer.scene.scene
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    scene.enable_indirect_light(False)
    scene.enable_sun_light(True)  # direction is set per view in render_views
    return renderer


UP_CANDIDATES = [np.array(u, dtype=float) for u in
                 [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]]


def detect_up_axis(mesh, n_samples=4000):
    """Pick the up direction whose 'down' side has a flat, down-facing surface —
    the print base miniatures stand on. Returns a unit vector."""
    pcd = mesh.sample_points_uniformly(n_samples)
    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    best, best_score = UP_CANDIDATES[0], -1.0
    for up in UP_CANDIDATES:
        h = pts @ up
        extent = h.max() - h.min()
        if extent <= 0:
            continue
        in_bottom_slab = h < h.min() + 0.02 * extent
        facing_down = normals @ up < -0.9
        score = float(np.mean(in_bottom_slab & facing_down))
        if score > best_score:
            best, best_score = up, score
    return best


def rotation_to_z_up(up):
    z = np.array([0.0, 0.0, 1.0])
    if np.allclose(up, z):
        return np.eye(3)
    if np.allclose(up, -z):
        return o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    axis = np.cross(up, z)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(up @ z, -1, 1))
    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)


def render_views(renderer, mesh_path, n_views, elevation_deg=20, up_axis="auto"):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles():
        raise ValueError("no triangles")
    mesh.compute_vertex_normals()
    if up_axis == "auto":
        up = detect_up_axis(mesh)
    elif up_axis == "y":
        up = np.array([0.0, 1.0, 0.0])
    else:
        up = np.array([0.0, 0.0, 1.0])
    mesh.rotate(rotation_to_z_up(up), center=(0, 0, 0))

    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = [0.7, 0.7, 0.7, 1.0]

    renderer.scene.clear_geometry()
    renderer.scene.add_geometry("mesh", mesh, mat)

    bounds = mesh.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    radius = np.linalg.norm(bounds.get_extent()) * 1.4
    elev = np.deg2rad(elevation_deg)

    images = []
    for i in range(n_views):
        az = 2 * np.pi * i / n_views
        eye = center + radius * np.array(
            [np.cos(az) * np.cos(elev), np.sin(az) * np.cos(elev), np.sin(elev)]
        )
        renderer.setup_camera(45.0, center, eye, [0, 0, 1])
        # headlight: key light shines from the camera, tilted downward in world
        # space so shading is consistent with "up" from every orbit angle
        sun_dir = (center - eye) / np.linalg.norm(center - eye) + [0, 0, -0.6]
        renderer.scene.scene.set_sun_light(sun_dir / np.linalg.norm(sun_dir),
                                           [1.0, 1.0, 1.0], 90000)
        img = np.asarray(renderer.render_to_image())
        images.append(Image.fromarray(img))
    return images


@torch.no_grad()
def embed_texts(model, processor, categories, device):
    embeds = []
    for cat in categories:
        prompts = [t.format(cat) for t in PROMPT_TEMPLATES]
        inputs = processor(text=prompts, padding="max_length", return_tensors="pt").to(device)
        feat = as_tensor(model.get_text_features(**inputs))
        feat = torch.nn.functional.normalize(feat, dim=-1).mean(0)
        embeds.append(torch.nn.functional.normalize(feat, dim=-1))
    return torch.stack(embeds)  # (n_categories, dim)


@torch.no_grad()
def embed_images(model, processor, images, device):
    inputs = processor(images=images, return_tensors="pt").to(device)
    feat = as_tensor(model.get_image_features(**inputs))
    return torch.nn.functional.normalize(feat, dim=-1)  # (n_views, dim)


def pool_sims(view_sims, mode, axis=-2):
    """Pool per-view similarity scores (..., n_views, n_categories) over views.

    mean: robust whole-object consensus (a feature seen in 1 of 4 views keeps
    ~25% weight). max: "clearly visible from some angle" — lets single-view
    features decide. softmax: in between (sharpness set by BETA).
    """
    if mode == "mean":
        return view_sims.mean(axis)
    if mode == "max":
        return view_sims.max(axis)
    BETA = 50.0
    w = np.exp(BETA * (view_sims - view_sims.max(axis, keepdims=True)))
    return (w * view_sims).sum(axis) / w.sum(axis)


SKIP_TAGS = ("presupported", "pre-supported", "pre_supported", "supported",
             "base", "hollow", "75mm")


def skip(name):
    # "unsupported" means NO supports — don't let the "supported" tag match inside it
    low = name.lower().replace("unsupported", "")
    return any(t in low for t in SKIP_TAGS)


def find_stls(root):
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune skipped directories before descending — big win on slow drives
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and not skip(d)]
        for name in filenames:
            if (not name.startswith(".") and name.lower().endswith(".stl")
                    and not skip(name)):
                found.append(Path(dirpath) / name)
    return sorted(found)


def load_file_list(inp, cache_dir, rescan=False):
    """Directory walk with cached file list (see --rescan)."""
    walk_cache = None
    if cache_dir:
        walk_id = hashlib.sha1(f"{inp.resolve()}|{SKIP_TAGS}|unsupported-ok".encode()).hexdigest()
        walk_cache = Path(cache_dir) / f"walk-{walk_id}.json"
    if walk_cache and walk_cache.exists() and not rescan:
        saved = json.loads(walk_cache.read_text())
        files = [Path(p) for p in saved["files"]]
        gone = [f for f in files if not f.exists()]
        files = [f for f in files if f.exists()]
        age_days = (time.time() - saved["scanned"]) / 86400
        note = f", {len(gone)} vanished since scan" if gone else ""
        print(f"using cached file list: {len(files)} files, scanned "
              f"{age_days:.1f} days ago{note} (--rescan to refresh)")
        return files
    files = find_stls(inp)
    if walk_cache:
        walk_cache.parent.mkdir(parents=True, exist_ok=True)
        walk_cache.write_text(json.dumps(
            {"scanned": time.time(), "files": [str(f) for f in files]}))
    return files


def cache_key(f, args):
    stat = f.stat()
    # "pv" = per-view cache format: (n_views, dim) instead of one pooled vector
    raw = f"{f.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{args.views}|{args.render_size}|{args.up_axis}|{args.model}|pv"
    return hashlib.sha1(raw.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="STL file or directory of STL files")
    parser.add_argument("--categories", default="categories.txt")
    parser.add_argument("--out", default="results.csv")
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--model", default="google/siglip2-so400m-patch14-384")
    parser.add_argument("--save-renders", help="directory to save render images for debugging")
    parser.add_argument("--up-axis", choices=["auto", "z", "y"], default="auto",
                        help="up axis of source meshes; auto detects the flat print base (default)")
    parser.add_argument("--cache-dir", default="embed-cache",
                        help="directory of cached per-file image embeddings; reruns with new "
                             "categories skip rendering/embedding entirely (set '' to disable)")
    parser.add_argument("--rescan", action="store_true",
                        help="re-walk the input directory instead of using the cached file list")
    parser.add_argument("--pool", choices=["mean", "max", "softmax"], default="mean",
                        help="how per-view scores combine: mean = whole-object consensus, "
                             "max = single-view features decide, softmax = in between")
    args = parser.parse_args()

    inp = Path(args.input)
    files = load_file_list(inp, args.cache_dir, args.rescan) if inp.is_dir() else [inp]
    if not files:
        sys.exit(f"no STL files found under {inp}")
    categories = [l.strip() for l in open(args.categories) if l.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModel, AutoProcessor

    print(f"loading {args.model} on {device} ...")
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    text_embeds = embed_texts(model, processor, categories, device)
    renderer = None  # created lazily on first cache miss

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    hits = 0

    rdir = Path(args.save_renders) if args.save_renders else None

    rows = []
    for f in tqdm(files, desc="classifying"):
        cache_file = cache_dir / f"{cache_key(f, args)}.npy" if cache_dir else None
        # --save-renders only forces a re-render for files whose renders are missing
        renders_saved = rdir is None or all(
            (rdir / f"{f.stem}_view{i}.png").exists() for i in range(args.views))
        if cache_file and cache_file.exists() and renders_saved:
            img_embeds = torch.from_numpy(np.load(cache_file)).to(device, dtype=text_embeds.dtype)
            hits += 1
        else:
            if rdir and renders_saved:
                # embed straight from previously saved renders — no re-rendering
                images = [Image.open(rdir / f"{f.stem}_view{i}.png").convert("RGB")
                          for i in range(args.views)]
            else:
                if renderer is None:
                    renderer = make_renderer(args.render_size)
                try:
                    images = render_views(renderer, f, args.views, up_axis=args.up_axis)
                except Exception as e:
                    rows.append({"file": str(f), "top1": f"RENDER_ERROR: {e}"})
                    continue
                if rdir:
                    rdir.mkdir(parents=True, exist_ok=True)
                    for i, im in enumerate(images):
                        im.save(rdir / f"{f.stem}_view{i}.png")
            img_embeds = embed_images(model, processor, images, device)
            if cache_file:
                np.save(cache_file, img_embeds.float().cpu().numpy())
        view_sims = (img_embeds @ text_embeds.T).float().cpu().numpy()  # (n_views, n_cats)
        sims = torch.from_numpy(pool_sims(view_sims, args.pool))
        order = sims.argsort(descending=True)
        row = {"file": str(f)}
        for rank in range(min(3, len(categories))):
            idx = order[rank]
            row[f"top{rank + 1}"] = categories[idx]
            row[f"score{rank + 1}"] = round(sims[idx].item(), 4)
        rows.append(row)

    fields = ["file", "top1", "score1", "top2", "score2", "top3", "score3"]
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} models, {hits} from embedding cache)")


if __name__ == "__main__":
    main()
