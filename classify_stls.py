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
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import torch
from PIL import Image
from tqdm import tqdm

import instrument
import pose
from pose import detect_up_axis
from instrument import stage

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


# Binary STL: an 80-byte header, a uint32 triangle count, then this fixed
# 50-byte record per triangle. The fixed stride is the whole trick.
STL_RECORD = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])


def read_binary_stl(path):
    """A binary STL read straight into arrays, or None if it is not one.

    read_triangle_mesh dominates everything before the first pixel: ~3.9 s on an
    800k-triangle collection mesh against ~120 ms here, where the upload it
    feeds is 275 ms. Optimising the renderer was optimising the small half
    (eval/load_path.py, docs/masa/renderer_alternatives.md).

    The header cannot be trusted to say which format this is — plenty of binary
    STLs start with "solid" — so the test is arithmetic: it is binary only if
    the file is exactly the length the triangle count implies. Anything else
    (ASCII, truncated, junk) returns None and takes the Open3D path.

    The result is a triangle soup, three unshared vertices per triangle, which
    is what an STL *is*. Open3D's reader welds a handful (108 of 2.4M on a real
    mesh); we do not, and that difference shows up in the render."""
    size = path.stat().st_size
    if size < 84:
        return None
    with open(path, "rb") as fh:
        n = int(np.frombuffer(fh.read(84)[80:84], "<u4")[0])
        if n == 0 or size != 84 + 50 * n:
            return None
        # fromfile continues from the header rather than materialising the
        # whole record block as bytes first — 200 MB on a 4M-triangle mesh
        rec = np.fromfile(fh, dtype=STL_RECORD, count=n)
    if len(rec) != n:                       # short read despite the size check
        return None
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(rec["v"].reshape(-1, 3).astype(np.float64)),
        o3d.utility.Vector3iVector(np.arange(3 * n, dtype=np.int32).reshape(-1, 3)))


def load_mesh(mesh_path):
    mesh_path = Path(mesh_path)
    mesh = None
    if mesh_path.suffix.lower() == ".stl":
        mesh = read_binary_stl(mesh_path)
    if mesh is None:
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles():
        raise ValueError("no triangles")
    mesh.compute_vertex_normals()
    return mesh


DEFAULT_ELEVATIONS = [20.0]
UP_TILE_ELEVATION = 20.0  # pose contact sheet: fixed, independent of --elevations
UP_TILE_AZIMUTHS = 4      # views per up candidate for the ensemble; the sheet still gets one


def view_angles(n_views, elevations):
    """(azimuth, elevation) radian pairs: a full turntable ring per elevation.

    Elevation-major, so views 0..n_views-1 are the first ring — a run with one
    elevation lays out exactly as it did before elevations existed, and
    view0.png keeps meaning the same camera."""
    return [(2 * np.pi * i / n_views, np.deg2rad(e))
            for e in elevations for i in range(n_views)]


def orbit_camera(center, radius, az, elev):
    """(eye, camera-up, sun direction) for one turntable position, Z-up world."""
    eye = center + radius * np.array(
        [np.cos(az) * np.cos(elev), np.sin(az) * np.cos(elev), np.sin(elev)]
    )
    # Camera 'up' is world +Z carried along the orbit, not +Z itself: the two
    # frame the image identically, but past |elev| 87.44 (|up . view| > 0.999)
    # Filament calls +Z degenerate and swaps in a fixed fallback up, which
    # freezes the image orientation so azimuth stops changing the render.
    up = np.array([-np.cos(az) * np.sin(elev), -np.sin(az) * np.sin(elev), np.cos(elev)])
    # headlight: key light shines from the camera, tilted downward in world
    # space so shading is consistent with "up" from every orbit angle
    sun = (center - eye) / np.linalg.norm(center - eye) + np.array([0, 0, -0.6])
    return eye, up, sun / np.linalg.norm(sun)


def _shoot(renderer, cams):
    """Render one image per (center, eye, up, sun) with the geometry as loaded."""
    images = []
    for center, eye, up, sun in cams:
        renderer.setup_camera(45.0, center, eye, up)
        renderer.scene.scene.set_sun_light(sun, [1.0, 1.0, 1.0], SUN_INTENSITY)
        images.append(Image.fromarray(np.asarray(renderer.render_to_image())))
    return images


def _upload(renderer, mesh):
    """Put the mesh on the GPU and return its framing. Still the expensive half
    of rendering — 275 ms on an 800k-triangle STL and ~1.0 s on a 4.4M-triangle
    one, against ~30-50 ms per view — so callers should upload once and move the
    camera.

    (An earlier version of this said 15 s per upload. `eval/load_path.py`
    measures ~10x less; the cost tracks *vertices*, and an STL is a soup at 3.00
    verts per triangle, so quote a figure with its vertex count.)"""
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = [0.7, 0.7, 0.7, 1.0]
    renderer.scene.clear_geometry()
    renderer.scene.add_geometry("mesh", mesh, mat)
    bounds = mesh.get_axis_aligned_bounding_box()
    return bounds.get_center(), np.linalg.norm(bounds.get_extent()) * 1.4


def render_views(renderer, mesh, angles):
    """Render one image per (azimuth, elevation) pair. The mesh must already be
    rotated into Z-up world space (the light rig and camera 'up' assume it)."""
    center, radius = _upload(renderer, mesh)
    cams = [(center, *orbit_camera(center, radius, az, elev)) for az, elev in angles]
    return _shoot(renderer, cams)


def render_up_candidate_grid(renderer, mesh, n_az=UP_TILE_AZIMUTHS):
    """[6][n_az] renders — each candidate up, seen from n_az azimuths.

    One upload, not six. Rotating the mesh per candidate and re-uploading costs
    the expensive half of rendering six times over — measured, an upload is
    275 ms on an 800k-triangle STL against ~30 ms per tile, so five extra
    uploads would roughly triple this call; moving the camera instead costs
    nothing. The two are exactly
    equivalent here because all six candidate rotations are signed axis
    permutations: the rotated mesh's bounding box is the rotated box, so its
    centre is R@c and its extent is a permutation of e — leaving the framing
    radius ||e|| identical. Compute each camera in the rotated frame as before,
    then carry it back with R.T. Verified pixel-identical against the
    rotate-the-mesh version."""
    center, radius = _upload(renderer, mesh)
    angles = view_angles(n_az, [UP_TILE_ELEVATION])
    grid = []
    for up in pose.UP_CANDIDATES:
        R = rotation_to_z_up(up)
        c_rot = R @ center                      # exact: axis permutation
        cams = []
        for az, elev in angles:
            eye, cam_up, sun = orbit_camera(c_rot, radius, az, elev)
            cams.append((R.T @ c_rot, R.T @ eye, R.T @ cam_up, R.T @ sun))
        grid.append(_shoot(renderer, cams))
    return grid


def render_up_candidate_tiles(renderer, mesh):
    """One render per candidate up (fixed azimuth) — the VLM contact sheet."""
    return [row[0] for row in render_up_candidate_grid(renderer, mesh, 1)]


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


def render_key(f):
    """Per-file prefix for saved renders: '<stem>_<6 hex of the path>'.

    Stems are not unique — a collection routinely holds one Baal_Flaming_Sword_L
    per kit — and the renders all land in one flat directory, so keying by stem
    alone let the last file walked overwrite the others' images and every tool
    then showed one model's render for all of them. The path disambiguates;
    mtime and size deliberately do not, so re-rendering a file replaces its own
    images instead of accumulating a set per edit. Only the path is hashed, so
    the stem stays readable and searchable in a directory listing."""
    return f"{f.stem}_{hashlib.sha1(str(f.resolve()).encode()).hexdigest()[:6]}"


def render_index(rdir):
    """Map '<render_key>_view<i>' to the saved render, from one listing of the dir.

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


def save_renders(rdir, key, images, fmt):
    """Write the debug renders under a render_key() prefix. Never fails the run —
    like the pose sheet, these exist for a human to look at, not for the
    classifier."""
    ext, opts = RENDER_FORMATS[fmt]
    try:
        rdir.mkdir(parents=True, exist_ok=True)
        for i, im in enumerate(images):
            im.save(rdir / f"{key}_view{i}{ext}", **opts)
    except OSError as e:
        print(f"  could not save renders for {key}: {e}")


class MeshPrefetcher:
    """Load meshes a few files ahead of the consumer, on a background thread.

    Loading is disk+CPU while everything after it is GPU, and the reader
    releases the GIL — measured, this thread takes 1% of GIL time while being
    33% of all py-spy samples — so it overlaps with rendering for free. Bounded
    so a queue of 4M-triangle meshes cannot outrun memory.

    **It matters much less than it used to.** Against `read_triangle_mesh` this
    hid a real cost: `mesh-wait`, the main thread blocked here, was 18.6% of a
    602-model run (746 ms/model), and one thread at depth 2 was not keeping up.
    Since `read_binary_stl` the same measurement is 0.4% (10 ms/model) and a
    whole mesh parses in 11-66 ms, so what is left to hide is ~1-2% of a run.
    Do not re-tune depth or worker count off the old figures — see LEARNINGS.

    The caller passes only the files that will actually ask for a mesh, so the
    queue is consumed strictly in order and the head always matches the request.
    It used to take the whole file list and drop skipped files on the way past,
    which made one cache miss pay for every skip before it: 275 pose-cached
    files ahead of a miss meant 11.6 GB read and discarded inside one get()."""

    def __init__(self, files, depth=2, enabled=True):
        self.enabled = enabled and depth > 0
        self._done = False
        if not self.enabled:
            return
        # depth is the memory bound, not a speed knob: it caps how many loaded
        # meshes may sit unconsumed. The GPU eats one at a time and the p99 mesh
        # is 4M triangles, so running far ahead buys nothing and costs a lot.
        self._q = queue.Queue(maxsize=depth)
        threading.Thread(target=self._run, args=(list(files),), daemon=True).start()

    def _run(self, files):
        for f in files:
            try:
                self._q.put((f, load_mesh(f)))
            except Exception as e:                 # re-raised when the consumer asks
                self._q.put((f, e))
        self._q.put((None, None))                  # end of list

    def get(self, f):
        """The mesh for f, which must be the next file in the prefetch list.

        Anything but f at the head means the list is exhausted or the caller has
        diverged from it. Rather than hunt down the queue for f — the old
        behaviour, and the reason one miss could drag gigabytes through it — the
        prefetcher retires and this and every later get() loads directly."""
        if not self.enabled or self._done:
            return load_mesh(f)
        g, value = self._q.get()                   # blocks if the loader is behind
        if g != f:                                 # end of list, or divergence
            self._done = True
            return load_mesh(f)
        if isinstance(value, Exception):
            raise value
        return value


def resolve_up(mesh, args, get_renderer, vlm_backend, score_upright=None,
               sheet_path=None, defer=None):
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
    with stage("pose-geometry"):
        geo_scores = pose.up_axis_scores(mesh)
        geo_idx, ratio, best = pose.rank_up_scores(geo_scores)
    up, source, margin = pose.UP_CANDIDATES[geo_idx], "heuristic", None

    sheet_tiles = None
    if score_upright is not None:
        with stage("pose-render"):
            grid = render_up_candidate_grid(get_renderer(), mesh)
        sheet_tiles = [row[0] for row in grid]           # the VLM still sees six
        flat = [im for row in grid for im in row]
        with stage("pose-embed"):
            sig = np.asarray(score_upright(flat)).reshape(len(grid), -1).mean(axis=1)
        idx, margin = pose.combine_up(geo_scores, sig)
        if idx != geo_idx:
            up, source = pose.UP_CANDIDATES[idx], "ensemble"

    # Escalate on the ensemble's own doubt. Without SigLIP there is no ensemble
    # and no margin, so fall back to geometry's confidence.
    escalate = (pose.needs_arbiter_margin(margin, args.up_margin) if margin is not None
                else pose.needs_arbiter(ratio, best, args.up_conf))
    if vlm_backend and escalate:
        if sheet_tiles is None:
            with stage("pose-render"):
                sheet_tiles = render_up_candidate_tiles(get_renderer(), mesh)
        call = lambda: pose.ask_vlm_up(
            sheet_tiles, vlm_backend, args.cache_dir or ".",
            args.pose_vlm_model or pose.DEFAULT_VLM_MODELS.get(vlm_backend),
            save_to=sheet_path, project=getattr(args, "gemini_project", None))
        if defer is not None:
            # Hand the call off and keep going. A network arbiter averages 24 s
            # against 3-28 s of local work for a whole model, so waiting here
            # leaves the run majority idle; the answer only decides the pose, so
            # this file can be finished later. apply_arbiter closes the loop.
            defer(call)
            return up, ratio, source, margin
        with stage("arbiter-inline"):
            idx = call()
        if idx is not None and not np.allclose(pose.UP_CANDIDATES[idx], up):
            return pose.UP_CANDIDATES[idx], ratio, "vlm", margin
    return up, ratio, source, margin


def apply_arbiter(idx, up, ratio, source, margin):
    """Fold a deferred arbiter answer into a pose already resolved without it."""
    if idx is not None and not np.allclose(pose.UP_CANDIDATES[idx], up):
        return pose.UP_CANDIDATES[idx], ratio, "vlm", margin
    return up, ratio, source, margin


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


EMBED_BATCH = 0   # 0 = one call per image list, which is what --views implies


@torch.no_grad()
def embed_images(model, processor, images, device, batch=None):
    """Row-normalised embeddings, (n_images, dim).

    batch caps how many images go to the GPU at once; 0/None sends the whole
    list, which is the historical behaviour and fine at 16-40 images (measured
    peak 2.5 GB of a 7.8 GB card). Raise it to keep the GPU busier when the
    image list is long, lower it if SigLIP has to share the card."""
    batch = batch or EMBED_BATCH or len(images)
    out = []
    for i in range(0, len(images), batch):
        inputs = processor(images=images[i:i + batch], return_tensors="pt").to(device)
        out.append(as_tensor(model.get_image_features(**inputs)))
    feat = out[0] if len(out) == 1 else torch.cat(out)
    return torch.nn.functional.normalize(feat, dim=-1)


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
                             "<render config>/<stem>_<path hash>_view<i>.<ext> plus "
                             "<stem>_<path hash>_pose.png for each model whose up axis "
                             "the VLM had to arbitrate (the hash keeps two models that "
                             "share a filename from overwriting each other)")
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
    parser.add_argument("--pose-vlm", choices=["auto", "ollama", "claude", "gemini", "off"],
                        default="auto",
                        help="arbiter for uncertain up detection: gemini on Vertex AI, "
                             "local ollama vision model, claude CLI, or off. auto "
                             "(default) = gemini if gcloud ADC resolves, else ollama if "
                             "reachable, else none. gemini-3.5-flash is the only arbiter "
                             "measured to beat the ensemble (43/44 against 40/44) and "
                             "bills ~$0.30 per full-collection run; --pose-vlm ollama "
                             "keeps it local at 41/44")
    parser.add_argument("--pose-vlm-model", default=None,
                        help="model for --pose-vlm; defaults per backend "
                             f"({', '.join(f'{k}={v}' for k, v in pose.DEFAULT_VLM_MODELS.items() if v)})")
    parser.add_argument("--gemini-project", default=None,
                        help="GCP project for --pose-vlm gemini (default: "
                             "$GOOGLE_CLOUD_PROJECT or `gcloud config get-value project`)")
    parser.add_argument("--embed-batch", type=int, default=0,
                        help="images per SigLIP call (0 = the whole view list at once). "
                             "Raise to keep the GPU busier on long lists; lower if "
                             "SigLIP has to share the card")
    parser.add_argument("--prefetch", type=int, default=2,
                        help="meshes to load ahead on a background thread (0 disables). "
                             "Worth ~1-2%% of a run since the binary-STL parser landed; "
                             "it was worth 18.6%% before that, so do not re-tune this "
                             "off the older figures")
    parser.add_argument("--arbiter-workers", type=int, default=8,
                        help="concurrent pose-VLM calls for network backends. The call "
                             "averages 24s against 3-28s of local work per model, so "
                             "waiting inline leaves the run mostly idle")
    parser.add_argument("--no-defer-arbiter", dest="defer_arbiter", action="store_false",
                        help="wait for each pose-VLM answer inline instead of parking the "
                             "file and revisiting it. Slower by design — kept so the "
                             "overlap can be measured against the behaviour it replaced")
    parser.add_argument("--up-margin", type=float, default=pose.MARGIN_THRESHOLD,
                        help="escalate to the pose VLM when the ensemble's winning "
                             "candidate leads the runner-up by less than this (0-2). "
                             "Lower = fewer VLM calls")
    parser.add_argument("--up-conf", type=float, default=0.6,
                        help="fallback ambiguity threshold used only with --skip-embed, "
                             "where there is no ensemble margin: runner-up/best flat-base "
                             "score ratio above this escalates to the pose VLM")
    parser.add_argument("--no-up-ensemble", dest="up_ensemble", action="store_false",
                        help="decide the up axis from flat-base geometry alone, without "
                             "the SigLIP vote over the up-candidate tiles")
    parser.add_argument("--skip-embed", action="store_true",
                        help="skip embedding the generated images")
    parser.add_argument("--instrument", nargs="?", const="instrument.json",
                        default=None, metavar="PATH",
                        help="record per-stage timings and CPU/NVIDIA/amdgpu "
                             "utilization to PATH (default instrument.json), and "
                             "print the breakdown at the end. Rendering runs on the "
                             "amd iGPU and embedding on the nvidia card, so both "
                             "are sampled")
    args = apply_run_params(parser)
    if not args.input:
        sys.exit("no input given, and no directory recorded in "
                 f"{Path(args.cache_dir or '.') / RUN_PARAMS_FILE}")

    if args.instrument:
        instrument.enable(args.instrument)

    inp = Path(args.input)
    with stage("walk"):
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
    with stage("model-load"):
        model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()
        processor = AutoProcessor.from_pretrained(args.model)

    with stage("text-embed"):
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
        # gemini first: it is the only arbiter measured to beat the ensemble
        # (43/44 against 40/44), where gemma reaches 41/44 and haiku/sonnet on a
        # 256px sheet score below running no arbiter at all. It bills per call —
        # ~$0.30 for a 602-model run at ~120 escalations — so the choice is
        # always announced, and --pose-vlm ollama/off opts out.
        try:
            args.gemini_project = args.gemini_project or pose.gcloud_project()
            pose.gcloud_token()
            vlm_backend = "gemini"
        except Exception as e:
            vlm_backend = "ollama" if pose.ollama_available() else None
            print(f"pose VLM: gemini unavailable ({e}); "
                  + ("falling back to ollama" if vlm_backend
                     else "ollama not reachable either — ambiguous poses keep the "
                          "heuristic guess"))
    elif vlm_backend == "off":
        vlm_backend = None
    vlm_model = args.pose_vlm_model or pose.DEFAULT_VLM_MODELS.get(vlm_backend)
    if vlm_backend == "gemini":
        # Fail here rather than on the first ambiguous model, thousands of
        # renders into a run: resolving the project and minting a token are the
        # two things that go wrong, and both are cheap to check up front.
        # Explicit --pose-vlm gemini is an error if unavailable; auto already
        # fell back above and never reaches this.
        try:
            args.gemini_project = args.gemini_project or pose.gcloud_project()
            pose.gcloud_token()
        except Exception as e:
            raise SystemExit(f"--pose-vlm gemini: {e}")
        print(f"pose VLM: {vlm_model} on Vertex AI, project {args.gemini_project} "
              f"— billed per escalation")
    elif vlm_backend:
        print(f"pose VLM: {vlm_model or vlm_backend}")

    # The arbiter sheet scales each tile to SHEET_THUMB, and Image.thumbnail
    # never enlarges — so tiles rendered smaller than that sit padded in their
    # cells and the arbiter sees a smaller sheet than the number implies. Worth
    # saying out loud: sheet size is the knob that moved sonnet 10 of 44.
    if vlm_backend and args.render_size < pose.SHEET_THUMB:
        print(f"  note: --render-size {args.render_size} is below the {pose.SHEET_THUMB}px "
              f"sheet tile, so the arbiter sees {args.render_size}px tiles padded into "
              f"{pose.SHEET_THUMB}px cells, not a {pose.SHEET_THUMB}px sheet")

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
    arbiter_pool = None       # the finally below shuts it down; it must exist first
    try:
        # Only a pose-cache miss ever calls prefetch.get(): a forced --up-axis
        # skips pose resolution altogether, and the render path below loads its
        # own mesh. Anything else here is gigabytes read and thrown away.
        wanted = [] if args.up_axis in ("z", "y") else [
            f for f in files if pose.file_identity(f) not in pose_cache]
        prefetch = MeshPrefetcher(wanted, args.prefetch)
        # A network arbiter is worth overlapping; a local one is not — ollama
        # shares the GPU with SigLIP and they evict each other (a measured 10.1 s
        # reload against 0.49 s of inference), so running them concurrently is
        # slower than taking turns.
        defer_arbiter = args.defer_arbiter and vlm_backend in ("gemini", "claude")
        arbiter_pool = ThreadPoolExecutor(max_workers=args.arbiter_workers) \
            if defer_arbiter else None
        deferred, pending_box = [], []

        def process(f, deferred_answer=None):
            """One file end to end.

            deferred_answer is (arbiter_index, pose-as-resolved-without-it) for a
            file parked on its first visit. Pose resolution is *not* repeated on
            the revisit: the ensemble already decided, and re-running it would
            redo the mesh load, the 24 candidate renders and their embeddings —
            and, because the deferral hook is off the second time, ask the
            arbiter all over again. Measured: that cost more than the overlap
            saved, and two independent calls disagreed on three models."""
            nonlocal hits, redrawn
            mesh = None
            pose_changed = False
            rkey = render_key(f) if rdir else None  # resolves the path; do it once
            if args.up_axis in ("z", "y"):
                up = [0.0, 0.0, 1.0] if args.up_axis == "z" else [0.0, 1.0, 0.0]
                entry = {"up": up, "confidence": 0.0, "source": "forced"}
            else:
                entry = pose_cache.get(pose.file_identity(f))
                if entry is None:
                    if deferred_answer is not None:
                        idx, resolved = deferred_answer
                        up, ratio, source, margin = apply_arbiter(idx, *resolved)
                    else:
                        try:
                            # blocking here is the loader failing to keep ahead,
                            # which is exactly what loader_worker_count is for
                            with stage("mesh-wait"):
                                mesh = prefetch.get(f)
                        except Exception as e:
                            rows.append({"file": str(f), "top1": f"RENDER_ERROR: {e}"})
                            return
                        up, ratio, source, margin = resolve_up(
                            mesh, args, get_renderer, vlm_backend, score_upright,
                            sheet_path=rdir / f"{rkey}_pose.png" if rdir else None,
                            defer=(lambda call: pending_box.append(call))
                                  if defer_arbiter else None)
                        if pending_box:
                            # arbiter running elsewhere — park this file and move on
                            call = pending_box.pop()
                            def timed(call=call):
                                with instrument.arbiter_call():
                                    return call()
                            deferred.append((f, arbiter_pool.submit(timed),
                                             (up, ratio, source, margin)))
                            return
                    entry = {"up": [float(v) for v in up],
                             "confidence": round(ratio, 4), "source": source,
                             "margin": None if margin is None else round(margin, 4),
                             "v": pose.POSE_CACHE_VERSION}
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
                f"{rkey}_view{i}" in saved_renders for i in range(n_views)))

            if need_embeds or need_renders:
                try:
                    if mesh is None:
                        with stage("mesh-load"):
                            mesh = load_mesh(f)
                    with stage("view-render"):
                        mesh.rotate(rotation_to_z_up(np.array(entry["up"])), center=(0, 0, 0))
                        images = render_views(get_renderer(), mesh, angles)
                except Exception as e:
                    rows.append({"file": str(f), "top1": f"RENDER_ERROR: {e}"})
                    return
                # write whenever we rendered, not only when files were missing:
                # at 0.13 s it is cheaper than leaving a stale image on disk next
                # to a fresh embedding
                if rdir is not None:
                    with stage("save-renders"):
                        save_renders(rdir, rkey, images, args.render_format)
                    redrawn += not need_embeds

            if cached and not args.skip_embed:
                with stage("cache-load"):
                    img_embeds = torch.from_numpy(np.load(cache_file)).to(device, dtype=text_embeds.dtype)
                hits += 1
            elif need_embeds:
                with stage("embed"):
                    img_embeds = embed_images(model, processor, images, device,
                                              batch=args.embed_batch)
                if cache_file:
                    with stage("cache-save"):
                        np.save(cache_file, img_embeds.float().cpu().numpy())

            if not args.skip_embed:
                with stage("score"):
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

        for f in tqdm(files, desc="classifying"):
            process(f)

        if deferred:
            # The arbiter calls have been in flight since their file was parked,
            # so most are already answered; this is the tail, not the wait. The
            # mesh is reloaded rather than held — a 4M-triangle mesh costs more
            # memory than the reload costs time, and only ~20% of files land here.
            print(f"resolving {len(deferred)} deferred arbiter calls")
            for f, fut, resolved in tqdm(deferred, desc="arbiter"):
                try:
                    # the tail: how much of the arbiter did *not* overlap
                    with stage("arbiter-wait"):
                        idx = fut.result()
                except Exception as e:              # one bad call must not sink the rest
                    print(f"  arbiter failed for {f.stem}: {e}")
                    idx = None                      # keep the ensemble's answer
                process(f, deferred_answer=(idx, resolved))
    finally:
        # Drop queued arbiter calls rather than letting the interpreter join
        # them at exit: they are non-daemon threads at ~24 s each, so a Ctrl-C
        # with a full queue would otherwise hang for minutes with nothing to show
        # for it. Their files simply keep the ensemble's pose.
        if arbiter_pool is not None:
            arbiter_pool.shutdown(wait=False, cancel_futures=True)
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
    instrument.report()


if __name__ == "__main__":
    main()
