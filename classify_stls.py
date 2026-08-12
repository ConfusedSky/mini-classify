"""Zero-shot STL classification: multiview renders scored against text categories with SigLIP.

Usage:
  python classify_stls.py /path/to/stls --categories categories.txt --out results.csv
  python classify_stls.py model.stl --save-renders renders/   # single file, keep debug renders

Renders each mesh from several viewpoints (Open3D offscreen), embeds the views
with SigLIP, averages them, and ranks against text embeddings of the categories.

Viewpoints are a turntable of --views azimuths at each --elevations pitch, so
--views 4 --elevations 20,-10 gives 8 renders per mesh. Every run records its
parameters in <cache-dir>/run-params.json; cluster_models.py and
test_categories.py default from that file, so cache-identity flags (and the
input directory) only have to be typed once, here.

Meshes are stood upright first, from three tiers of evidence: flat print-base
geometry with a confidence ratio, a SigLIP vote over the six up-candidate tiles
(the two averaged; --no-up-ensemble for geometry alone), and a local VLM
arbitrating low-confidence cases (--pose-vlm). The front-facing view index is recorded
per file (front_view column) so downstream tools can show the render that
actually faces the viewer, and resolved poses persist in
<cache-dir>/pose-cache.json.
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

import pose
from pose import detect_up_axis

def as_tensor(feat):
    return feat if isinstance(feat, torch.Tensor) else feat.pooler_output


PROMPT_TEMPLATES = [
    "a 3D render of a {} miniature",
    "a photo of a {} figurine",
    "a tabletop miniature of a {}",
]


SUN_INTENSITY = 90000.0
# Ambient fill. The sun is the only light Filament gives us here — add_directional_
# /point_/spot_light all return True and then render as a <0.1/255 no-op — so with
# indirect light off, every surface facing away from the key falls to pure black
# and swallows detail (an 11% of object pixels under 25/255 on a hat brim shading a
# face). The built-in environment map is world-fixed and does not orbit with the
# camera, so it is deliberately kept far below the key: at 10k the crushed-black
# fraction is 0, while the brightness it adds still swings ~30/255 across azimuths.
FILL_INTENSITY = 10000.0


def make_renderer(size):
    # Meshes are rotated into Z-up world space before rendering, so this rig
    # (key light from above + ambient fill) is correct for any --up-axis choice.
    renderer = rendering.OffscreenRenderer(size, size)
    scene = renderer.scene.scene
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    scene.enable_indirect_light(True)
    scene.set_indirect_light_intensity(FILL_INTENSITY)
    scene.enable_sun_light(True)  # direction is set per view in render_views
    return renderer


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


def load_mesh(mesh_path):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles():
        raise ValueError("no triangles")
    mesh.compute_vertex_normals()
    return mesh


DEFAULT_ELEVATIONS = [20.0]
UP_TILE_ELEVATION = 20.0  # pose contact sheet: fixed, independent of --elevations


def view_angles(n_views, elevations):
    """(azimuth, elevation) radian pairs: a full turntable ring per elevation.

    Elevation-major, so views 0..n_views-1 are the first ring — a run with one
    elevation lays out exactly as it did before elevations existed, and
    view0.png keeps meaning the same camera."""
    return [(2 * np.pi * i / n_views, np.deg2rad(e))
            for e in elevations for i in range(n_views)]


def render_views(renderer, mesh, angles):
    """Render one image per (azimuth, elevation) pair. The mesh must already be
    rotated into Z-up world space (the light rig and camera 'up' assume it)."""
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = [0.7, 0.7, 0.7, 1.0]

    renderer.scene.clear_geometry()
    renderer.scene.add_geometry("mesh", mesh, mat)

    bounds = mesh.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    radius = np.linalg.norm(bounds.get_extent()) * 1.4

    images = []
    for az, elev in angles:
        eye = center + radius * np.array(
            [np.cos(az) * np.cos(elev), np.sin(az) * np.cos(elev), np.sin(elev)]
        )
        # Camera 'up' is world +Z carried along the orbit, not +Z itself: the two
        # frame the image identically, but past |elev| 87.44 (|up . view| > 0.999)
        # Filament calls +Z degenerate and swaps in a fixed fallback up, which
        # freezes the image orientation so azimuth stops changing the render.
        up = [-np.cos(az) * np.sin(elev), -np.sin(az) * np.sin(elev), np.cos(elev)]
        renderer.setup_camera(45.0, center, eye, up)
        # headlight: key light shines from the camera, tilted downward in world
        # space so shading is consistent with "up" from every orbit angle
        sun_dir = (center - eye) / np.linalg.norm(center - eye) + [0, 0, -0.6]
        renderer.scene.scene.set_sun_light(sun_dir / np.linalg.norm(sun_dir),
                                           [1.0, 1.0, 1.0], SUN_INTENSITY)
        img = np.asarray(renderer.render_to_image())
        images.append(Image.fromarray(img))
    return images


def render_up_candidate_tiles(renderer, mesh):
    """One render per candidate up (fixed azimuth) for the VLM contact sheet."""
    tiles = []
    for up in pose.UP_CANDIDATES:
        m = o3d.geometry.TriangleMesh(mesh)
        m.rotate(rotation_to_z_up(up), center=(0, 0, 0))
        tiles.append(render_views(renderer, m, view_angles(1, [UP_TILE_ELEVATION]))[0])
    return tiles


# Encodings for --save-renders. These files are written and never read back —
# the classifier always embeds the in-memory render — so the only thing a lossy
# format costs is what a human eye needs, not embedding fidelity. Measured on 16
# views at 2048 px: jpg 0.13 s / 205 KB each, png 3.83 s / 3.9 MB, webp 1.02 s /
# 100 KB. compress_level=1 is byte-identical to PIL's default 6 and 6.1x faster.
RENDER_FORMATS = {
    "jpg": (".jpg", {"quality": 92}),
    "png": (".png", {"compress_level": 1}),
    "webp": (".webp", {"quality": 90}),
}


def render_subdir(args):
    """Renders live under the camera config that produced them.

    A filename carries only stem and view index, but cache_key covers render
    size, views and elevations — so without this a rerun at a different size
    leaves the previous config's images in place and the contact sheets stop
    describing what was actually classified. Same elevation formatting as
    cache_key, so the two never disagree about what one config is."""
    elev = ",".join(f"{e:g}" for e in args.elevations)
    return f"{args.render_size}px-{args.views}v-e{elev}"


def render_index(rdir):
    """Map '<stem>_view<i>' to the saved render, from one listing of the dir.

    Extension-agnostic on purpose: a directory may hold PNGs written before
    --render-format existed alongside new JPEGs, and switching format must
    neither re-render them nor hide them from the tools. Newest wins when a view
    exists in both. One listing rather than a glob per view — the lookup runs
    n_views times per model, and real stems contain '(' and '['."""
    if rdir is None or not Path(rdir).is_dir():
        return {}
    files = sorted((p for p in Path(rdir).iterdir() if p.is_file()),
                   key=lambda p: p.stat().st_mtime)
    return {p.stem: p for p in files}


def save_renders(rdir, stem, images, fmt):
    """Write the debug renders. Never fails the run — like the pose sheet, these
    exist for a human to look at, not for the classifier."""
    ext, opts = RENDER_FORMATS[fmt]
    try:
        rdir.mkdir(parents=True, exist_ok=True)
        for i, im in enumerate(images):
            im.save(rdir / f"{stem}_view{i}{ext}", **opts)
    except OSError as e:
        print(f"  could not save renders for {stem}: {e}")


def resolve_up(mesh, args, get_renderer, vlm_backend, score_upright=None, sheet_path=None):
    """Resolve the up axis for --up-axis auto, cheapest evidence first:
    geometry, then SigLIP over the up-candidate tiles, then the VLM.

    Returns (up, ratio, source). Sources that leave the geometry answer alone
    stay "heuristic" so the embedding-cache key is unchanged — only an actual
    override becomes "ensemble" or "vlm".

    score_upright(tiles) -> per-candidate SigLIP scores; None (--skip-embed)
    falls back to geometry alone. The ensemble runs on *every* model rather
    than only low-confidence ones: geometry can be confidently wrong with a
    real-looking base (32mm_Gate_L scores a 0.43 ratio on the wrong face), and
    those never reach the arbiter.

    sheet_path, when the arbiter runs at all, keeps that model's contact sheet
    beside its renders — the scratch copy in the cache dir is one fixed name
    every model overwrites, so it only ever shows the last file processed."""
    geo_scores = pose.up_axis_scores(mesh)
    geo_idx, ratio, best = pose.rank_up_scores(geo_scores)
    up, source = pose.UP_CANDIDATES[geo_idx], "heuristic"

    tiles = None
    if score_upright is not None:
        tiles = render_up_candidate_tiles(get_renderer(), mesh)
        idx = pose.combine_up_scores(geo_scores, score_upright(tiles))
        if idx != geo_idx:
            up, source = pose.UP_CANDIDATES[idx], "ensemble"

    # confidence is still geometry's: it is what says "no base to measure"
    if vlm_backend and pose.needs_arbiter(ratio, best, args.up_conf):
        if tiles is None:
            tiles = render_up_candidate_tiles(get_renderer(), mesh)
        idx = pose.ask_vlm_up(tiles, vlm_backend, args.cache_dir or ".",
                              args.pose_vlm_model, save_to=sheet_path)
        if idx is not None and not np.allclose(pose.UP_CANDIDATES[idx], up):
            return pose.UP_CANDIDATES[idx], ratio, "vlm"
    return up, ratio, source


@torch.no_grad()
def embed_raw(model, processor, texts, device):
    """Embed raw text strings (no category templates), row-normalized."""
    inputs = processor(text=texts, padding="max_length", return_tensors="pt").to(device)
    feat = as_tensor(model.get_text_features(**inputs))
    return torch.nn.functional.normalize(feat, dim=-1)  # (n_texts, dim)


@torch.no_grad()
def embed_texts(model, processor, categories, device):
    embeds = []
    for cat in categories:
        prompts = [t.format(cat) for t in PROMPT_TEMPLATES]
        feat = embed_raw(model, processor, prompts, device).mean(0)
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


def cache_key(f, args, up_token):
    stat = f.stat()
    # A single 20° ring appends nothing, so keys written before --elevations
    # existed stay byte-identical and those (expensive) caches survive.
    elev = "" if args.elevations == DEFAULT_ELEVATIONS else \
        "|e:" + ",".join(f"{e:g}" for e in args.elevations)
    # "pv" = per-view cache format: (n_views, dim) instead of one pooled vector.
    # up_token is "auto"/"z"/"y" for deterministic poses (legacy-compatible)
    # and "vlm:<x,y,z>" when a VLM override changed the render.
    raw = f"{f.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{args.views}|{args.render_size}|{up_token}|{args.model}|pv{elev}"
    return hashlib.sha1(raw.encode()).hexdigest()


def total_views(args):
    return args.views * len(args.elevations)


def parse_elevations(text):
    """Comma-separated camera elevations in degrees: '20' or '20,-10,55'."""
    if isinstance(text, list):  # already parsed (came from the run manifest)
        return text
    try:
        elevs = [float(v) for v in text.split(",") if v.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a list of numbers: {text!r}")
    if not elevs:
        raise argparse.ArgumentTypeError("need at least one elevation")
    if any(abs(e) > 90 for e in elevs):
        # ±90 is straight down / straight up; render_views carries 'up' around the
        # orbit so the poles are ordinary cameras, not a degenerate look-at
        raise argparse.ArgumentTypeError("elevations must be within ±90 degrees")
    return elevs


def add_cache_args(parser, input_help):
    """Args that identify an embedding cache. Every tool reading the cache must
    agree on these, which is what the run manifest automates — declared in one
    place so a new one can't be added to the classifier and forgotten in the
    tools that read what it wrote."""
    parser.add_argument("input", nargs="?", help=input_help)
    parser.add_argument("--views", type=int, default=4,
                        help="azimuths per elevation ring (default 4)")
    parser.add_argument("--elevations", type=parse_elevations, default=DEFAULT_ELEVATIONS,
                        help="comma-separated camera elevations in degrees; each gets a "
                             "full ring of --views azimuths, so total views is the "
                             "product (default 20)")
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--model", default="google/siglip2-so400m-patch14-384")
    parser.add_argument("--up-axis", choices=["auto", "z", "y"], default="auto",
                        help="up axis of source meshes; auto detects the flat print base (default)")
    parser.add_argument("--cache-dir", default="embed-cache",
                        help="directory of cached per-file image embeddings; reruns with new "
                             "categories skip rendering/embedding entirely (set '' to disable)")


RUN_PARAMS_FILE = "run-params.json"
# What a classify run records for the tools that read its cache. Keys are
# argparse dests; anything not declared by a given tool's parser is ignored.
RUN_PARAMS_KEYS = ("input", "views", "elevations", "render_size", "model",
                   "up_axis", "pool", "categories", "renders_dir", "render_format")


def load_run_params(cache_dir):
    if not cache_dir:
        return {}
    p = Path(cache_dir) / RUN_PARAMS_FILE
    return json.loads(p.read_text()) if p.exists() else {}


def save_run_params(args):
    """Record this run's parameters next to the cache it just wrote. Kept with
    the cache rather than in a committed config so the description can't drift
    from what the embeddings actually are."""
    if not args.cache_dir:
        return
    params = {k: getattr(args, k, None) for k in RUN_PARAMS_KEYS}
    # a single-file run describes no collection — leave the recorded root alone
    params["input"] = str(Path(args.input).resolve()) if Path(args.input).is_dir() else None
    params = load_run_params(args.cache_dir) | {
        k: v for k, v in params.items() if v is not None}
    p = Path(args.cache_dir) / RUN_PARAMS_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(params, indent=2))


def apply_run_params(parser):
    """parse_args(), with defaults filled in from the last classify run.
    Explicit command-line values still win — set_defaults only moves the
    fallback."""
    known, _ = parser.parse_known_args()
    params = load_run_params(getattr(known, "cache_dir", None))
    dests = {a.dest for a in parser._actions}
    applied = {k: v for k, v in params.items() if k in dests}
    parser.set_defaults(**applied)
    args = parser.parse_args()
    if applied:
        print(f"defaults from {Path(known.cache_dir) / RUN_PARAMS_FILE}: "
              + ", ".join(sorted(applied)) + " (command line overrides)")
    return args


def main():
    parser = argparse.ArgumentParser()
    add_cache_args(parser, "STL file or directory of STL files "
                           "(defaults to the last run's directory)")
    parser.add_argument("--categories", default="categories.txt")
    parser.add_argument("--out", default="results.csv")
    parser.add_argument("--save-renders", dest="renders_dir",
                        help="directory to save render images for debugging, as "
                             "<render config>/<stem>_view<i>.<ext> plus <stem>_pose.png "
                             "for each model whose up axis the VLM had to arbitrate")
    parser.add_argument("--render-format", choices=sorted(RENDER_FORMATS), default="jpg",
                        help="encoding for --save-renders images (default jpg). Nothing "
                             "reads these back — the classifier embeds the in-memory "
                             "render — so lossy is safe here, and jpg encodes ~180x "
                             "faster and ~16x smaller than png at 2048 px")
    parser.add_argument("--rescan", action="store_true",
                        help="re-walk the input directory instead of using the cached file list")
    parser.add_argument("--pool", choices=["mean", "max", "softmax"], default="mean",
                        help="how per-view scores combine: mean = whole-object consensus, "
                             "max = single-view features decide, softmax = in between")
    parser.add_argument("--pose-vlm", choices=["auto", "ollama", "claude", "off"],
                        default="auto",
                        help="arbiter for low-confidence up detection: local ollama "
                             "vision model, claude CLI, or off (auto = ollama if reachable)")
    parser.add_argument("--pose-vlm-model", default="gemma4:26b",
                        help="ollama model name used by --pose-vlm")
    parser.add_argument("--up-conf", type=float, default=0.6,
                        help="up-detection ambiguity threshold: runner-up/best flat-base "
                             "score ratio above this escalates to the pose VLM")
    parser.add_argument("--no-up-ensemble", dest="up_ensemble", action="store_false",
                        help="decide the up axis from flat-base geometry alone, without "
                             "the SigLIP vote over the up-candidate tiles")
    parser.add_argument("--skip-embed", action="store_true",
                        help="skip embedding the generated images")
    args = apply_run_params(parser)
    if not args.input:
        sys.exit("no input given, and no directory recorded in "
                 f"{Path(args.cache_dir or '.') / RUN_PARAMS_FILE}")

    inp = Path(args.input)
    files = load_file_list(inp, args.cache_dir, args.rescan) if inp.is_dir() else [inp]
    if not files:
        sys.exit(f"no STL files found under {inp}")
    n_views = total_views(args)
    print(f"{n_views} views per model: {args.views} azimuths at "
          f"{', '.join(f'{e:g}' for e in args.elevations)} degrees")
    categories = [l.strip() for l in open(args.categories) if l.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModel, AutoProcessor

    print(f"loading {args.model} on {device} ...")
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    text_embeds = embed_texts(model, processor, categories, device)
    renderer = None  # created lazily on first render

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    hits = 0

    rdir = Path(args.renders_dir) / render_subdir(args) if args.renders_dir else None
    saved_renders = render_index(rdir)
    redrawn = 0  # rendered only to refresh --save-renders, embedding already cached
    angles = view_angles(args.views, args.elevations)

    pose_cache = pose.load_pose_cache(args.cache_dir)
    vlm_backend = args.pose_vlm
    if vlm_backend == "auto":
        vlm_backend = "ollama" if pose.ollama_available() else None
        if vlm_backend is None:
            print("pose VLM: ollama not reachable — ambiguous poses keep the heuristic guess")
    elif vlm_backend == "off":
        vlm_backend = None

    front_T = embed_raw(model, processor, pose.FRONT_PROMPTS, device).float().cpu().numpy()
    back_T = embed_raw(model, processor, pose.BACK_PROMPTS, device).float().cpu().numpy()

    score_upright = None
    if args.up_ensemble and not args.skip_embed:
        up_T = embed_raw(model, processor, pose.UPRIGHT_PROMPTS, device).float().cpu().numpy()
        down_T = embed_raw(model, processor, pose.TOPPLED_PROMPTS, device).float().cpu().numpy()

        def score_upright(tiles):
            embeds = embed_images(model, processor, tiles, device).float().cpu().numpy()
            return pose.upright_scores(embeds, up_T, down_T)

    def get_renderer():
        nonlocal renderer
        if renderer is None:
            renderer = make_renderer(args.render_size)
        return renderer

    rows = []
    try:
        for f in tqdm(files, desc="classifying"):
            mesh = None
            pose_changed = False
            if args.up_axis in ("z", "y"):
                up = [0.0, 0.0, 1.0] if args.up_axis == "z" else [0.0, 1.0, 0.0]
                entry = {"up": up, "confidence": 0.0, "source": "forced"}
            else:
                entry = pose_cache.get(pose.file_identity(f))
                if entry is None:
                    try:
                        mesh = load_mesh(f)
                    except Exception as e:
                        rows.append({"file": str(f), "top1": f"RENDER_ERROR: {e}"})
                        continue
                    up, ratio, source = resolve_up(
                        mesh, args, get_renderer, vlm_backend, score_upright,
                        sheet_path=rdir / f"{f.stem}_pose.png" if rdir else None)
                    entry = {"up": [float(v) for v in up],
                             "confidence": round(ratio, 4), "source": source}
                    pose_cache[pose.file_identity(f)] = entry
                    # Saved renders predate a fresh override, so they show the old
                    # pose. Only the debug files are at stake now — the embedding
                    # re-keys on its own, because the override moves up_token.
                    pose_changed = source in ("vlm", "ensemble")

            token = pose.embed_cache_token(entry, args.up_axis)
            cache_file = cache_dir / f"{cache_key(f, args, token)}.npy" if cache_dir else None
            cached = cache_file is not None and cache_file.exists()
            # Two independent questions. Embeddings come from the .npy cache or a
            # fresh render, never from the files on disk: those are debug output,
            # and nothing in their names ties them to this run's cache key. That
            # decoupling is what makes a lossy --render-format safe.
            need_embeds = not cached and not args.skip_embed
            # --save-renders only forces a re-render for files whose renders are missing
            need_renders = rdir is not None and (pose_changed or not all(
                f"{f.stem}_view{i}" in saved_renders for i in range(n_views)))

            if need_embeds or need_renders:
                try:
                    if mesh is None:
                        mesh = load_mesh(f)
                    mesh.rotate(rotation_to_z_up(np.array(entry["up"])), center=(0, 0, 0))
                    images = render_views(get_renderer(), mesh, angles)
                except Exception as e:
                    rows.append({"file": str(f), "top1": f"RENDER_ERROR: {e}"})
                    continue
                # write whenever we rendered, not only when files were missing:
                # at 0.13 s it is cheaper than leaving a stale image on disk next
                # to a fresh embedding
                if rdir is not None:
                    save_renders(rdir, f.stem, images, args.render_format)
                    redrawn += not need_embeds

            if cached and not args.skip_embed:
                img_embeds = torch.from_numpy(np.load(cache_file)).to(device, dtype=text_embeds.dtype)
                hits += 1
            elif need_embeds:
                img_embeds = embed_images(model, processor, images, device)
                if cache_file:
                    np.save(cache_file, img_embeds.float().cpu().numpy())

            if not args.skip_embed:
                view_np = img_embeds.float().cpu().numpy()
                if "front_view" not in entry:
                    entry["front_view"] = pose.front_view_index(view_np, front_T, back_T)
                view_sims = (img_embeds @ text_embeds.T).float().cpu().numpy()  # (n_views, n_cats)
                sims = torch.from_numpy(pool_sims(view_sims, args.pool))
                order = sims.argsort(descending=True)
                row = {"file": str(f), "up": pose.up_str(entry["up"]),
                       "pose_conf": entry["confidence"], "pose_source": entry["source"],
                       "front_view": entry["front_view"]}
                for rank in range(min(3, len(categories))):
                    idx = order[rank]
                    row[f"top{rank + 1}"] = categories[idx]
                    row[f"score{rank + 1}"] = round(sims[idx].item(), 4)
                rows.append(row)
    finally:
        # interrupted cold passes keep their (expensive) pose resolutions, and
        # still describe the cache they partly filled
        if args.up_axis == "auto":
            pose.save_pose_cache(args.cache_dir, pose_cache)
        save_run_params(args)

    fields = ["file", "top1", "score1", "top2", "score2", "top3", "score3",
              "up", "pose_conf", "pose_source", "front_view"]
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} models, {hits} from embedding cache"
          + (f", {redrawn} re-rendered only to refresh --save-renders" if redrawn else "")
          + ")")


if __name__ == "__main__":
    main()
