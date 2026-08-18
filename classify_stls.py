"""Zero-shot STL classification: multiview renders scored against text categories with SigLIP.

Usage:
  python classify_stls.py /path/to/stls --categories categories.txt --out results.csv
  python classify_stls.py model.stl --save-renders renders/   # single file, keep debug renders

Renders each mesh from several viewpoints (Open3D offscreen, in a render child
process), embeds the views with SigLIP in this process, and ranks the pooled
similarities against text embeddings of the categories.

**This file is the CLI entry, not the pipeline** (docs/actor-refactor/): args,
run-params, the cache guards and the wiring live here; the loop lives in
`src/driver.py`, and every stage it drives is one of the `src/` modules. The
functions kept below are the ones the eval harnesses and the sibling tools
import — the production render/embed helpers they measure against.

Nothing here is a second copy of a `src/` definition. The names the tools have
always imported from this module (`render_key`, `load_mesh`, `view_angles`,
`RENDER_FORMATS`, ...) are re-exported from the module that owns them, so
there is exactly one of each to keep in step — and `pool_sims`/`view_config`,
whose home `src/done.py` owns torch, are forwarded lazily by `__getattr__` at
the bottom of this file rather than imported at module scope.

Viewpoints are a turntable of --views azimuths at each --elevations pitch, so
--views 4 --elevations 20,-10 gives 8 renders per mesh. Every run records its
parameters in <cache-dir>/run-params.json; cluster_models.py and
test_categories.py default from that file, so cache-identity flags (and the
input directory) only have to be typed once, here.

Meshes are stood upright first, from three tiers of evidence: flat print-base
geometry with a confidence ratio, a SigLIP vote over the six up-candidate tiles
(the two averaged, always), and a VLM arbitrating low-confidence cases
(--pose-vlm). The front-facing view index is recorded per file (front_view
column) so downstream tools can show the render that actually faces the viewer,
and resolved poses persist in <cache-dir>/pose-cache.json.

**No module-scope torch here, deliberately.** `mp.get_context("spawn")` makes
the render child re-import this file as `__mp_main__` before it runs
`run_child`, so anything imported at module scope is imported in the child too
— and a torch import there is exactly what the child-side import rule forbids
(interfaces.md's import table: SigLIP lives in the parent, and torch in the
child costs VRAM and startup for nothing). The four embed helpers and `main`
import it where they use it.
"""
import argparse
import os
import hashlib
import json
import time
import sys
from pathlib import Path

import numpy as np
import open3d.visualization.rendering as rendering
from PIL import Image
from tqdm import tqdm

import instrument
from src import identity
from src import pose
from naming import SKIP_TAGS, skip
from instrument import stage
# The definitions the tools import from here, each from its one home — several
# are unused *in* this file and imported for the re-export alone, which is the
# point: one definition, and the callers' import path unchanged. None of these
# pulls torch, which is what lets them sit at module scope (docstring):
# `identity` is stdlib-only, `loader` and `renderer` are the child side.
from src.identity import (DEFAULT_ELEVATIONS, EMBED_CACHE_VERSION,
                          cache_key_from_identity, render_key)
from src.loader import STL_RECORD, load_mesh, read_binary_stl
from src.renderer import (FILL_INTENSITY, RENDER_FORMATS, SUN_INTENSITY,
                          UP_TILE_AZIMUTHS, UP_TILE_ELEVATION, make_offscreen,
                          orbit_camera, rotated_cams, rotation_to_z_up,
                          view_angles)


def as_tensor(feat):
    import torch
    return feat if isinstance(feat, torch.Tensor) else feat.pooler_output


def make_renderer(size):
    """The offscreen rig, the render child's own (`src.renderer`): the evals
    measure the production path, so they must build the production rig."""
    return make_offscreen(size)


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
    radius ||e|| identical. `renderer.rotated_cams` does that arithmetic — the
    same call the child's pose path makes, so these tiles and the pipeline's
    are framed by one piece of code. Verified pixel-identical against the
    rotate-the-mesh version."""
    center, radius = _upload(renderer, mesh)
    angles = view_angles(n_az, [UP_TILE_ELEVATION])
    return [_shoot(renderer, rotated_cams(rotation_to_z_up(up), center, radius, angles))
            for up in pose.UP_CANDIDATES]


def render_up_candidate_tiles(renderer, mesh):
    """One render per candidate up (fixed azimuth) — the VLM contact sheet."""
    return [row[0] for row in render_up_candidate_grid(renderer, mesh, 1)]


def render_subdir(args):
    """Renders live under the camera config that produced them.

    A filename carries only stem and view index, but cache_key covers render
    size, views and elevations — so without this a rerun at a different size
    leaves the previous config's images in place and the contact sheets stop
    describing what was actually classified."""
    # function-local: src.done owns torch and this module must not (docstring)
    from src.done import view_config
    return f"{args.render_size}px-{view_config(args)}"


# Cache layout. Everything a run derives from the collection lives under
# --cache-dir, so the cache is one directory rather than two that have to be
# passed around in step: the embeddings and the debug renders are both
# rebuildable, and both are worthless against a different --cache-dir.
EMBEDS_SUBDIR = "embeds"
RENDERS_SUBDIR = "renders"

# What the render child may hold in host-side meshes before its LRU evicts
# (RenderConfig.budget_bytes). A soft bound: in_flight meshes are never
# evicted, so the hard worst case is the admission window x the heaviest mesh
# (~450 MB at 3 x 150 MB — data_structures.md §residency). Not a flag: the one
# knob is the admission window, and this follows it.
RESIDENT_BUDGET_BYTES = 512 * 1024 * 1024


def embeds_dir(cache_dir):
    """Where the per-file .npy embeddings live, or None with caching off."""
    return Path(cache_dir) / EMBEDS_SUBDIR if cache_dir else None


def renders_dir(cache_dir, args):
    """Where --save-renders writes, under the camera config that produced them.

    Derived rather than passed: a renders directory paired with the wrong cache
    shows one run's images beside another run's embeddings, and the two have no
    way to notice."""
    return Path(cache_dir) / RENDERS_SUBDIR / render_subdir(args) if cache_dir else None


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


def resolve_up(mesh, args, get_renderer, vlm_backend, score_upright=None,
               sheet_path=None):
    """Resolve the up axis for --up-axis auto, cheapest evidence first:
    geometry, then SigLIP over the up-candidate tiles, then the VLM.

    **Not the production path any more** — `src/poser.py` is, driven by
    `src/driver.py`, with the geometry half computed in the render child. This
    stays because the pose evals (`eval/parser_gate.py`) measure against a
    single-process, in-line arrangement of the same three tiers, and they call
    this function rather than a copy of it. Its ensemble math is `src/pose.py`'s,
    the same functions the Poser calls.

    Returns (up, ratio, source). `source` records which tier *moved* the
    answer, not which ran (review P2.3-A): geometry's pick standing —
    including when the ensemble ran and agreed — stays "geometry"; an
    override becomes "siglip" or "vlm". Whether the ensemble ran at all is
    `margin is not None`, which pose_is_sufficient already keys on.

    score_upright(tiles) -> per-candidate SigLIP scores; None falls back to
    geometry alone, which only an eval arm asks for now (the flag that did,
    --no-up-ensemble, is retired) — and only that arm reaches the
    `args.up_conf` gate below, which the CLI no longer defines either. The
    ensemble runs on *every* model rather than only low-confidence ones:
    geometry can be confidently wrong with a real-looking base (32mm_Gate_L
    scores a 0.43 ratio on the wrong face), and those never reach the arbiter.

    sheet_path, when the arbiter runs at all, keeps that model's contact sheet
    beside its renders — the scratch copy in the cache dir is one fixed name
    every model overwrites, so it only ever shows the last file processed."""
    with stage("pose-geometry"):
        geo_scores = pose.up_axis_scores(mesh)
        geo_idx, ratio, best = pose.rank_up_scores(geo_scores)
    up, source, margin = pose.UP_CANDIDATES[geo_idx], "geometry", None

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
            up, source = pose.UP_CANDIDATES[idx], "siglip"

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
        # Inline, always: deferral left with the pipeline. Parking a file on an
        # in-flight call and folding the answer back is `src/poser.py` +
        # `src/arbiter.py` now, and the fold that used to close the loop here
        # (`apply_arbiter`) went with it.
        with stage("arbiter-inline"):
            idx = call()
        if idx is not None and not np.allclose(pose.UP_CANDIDATES[idx], up):
            return pose.UP_CANDIDATES[idx], ratio, "vlm", margin
    return up, ratio, source, margin


# The three embed helpers below are `src/embedder.py`'s methods in free-function
# form: same forwards, same normalisation, same dtypes — pinned equal by
# tests/test_embedder.py's parity suite. Production goes through the Embedder;
# these stay for the evals and the REPL, which hold their own model+processor
# and measure against exactly this arrangement. torch is imported inside each
# (see the module docstring: the render child re-imports this file), which is
# also why they carry `with torch.no_grad()` rather than the decorator.

def embed_raw(model, processor, texts, device):
    """Embed raw text strings (no category templates), row-normalized."""
    import torch
    with torch.no_grad():
        inputs = processor(text=texts, padding="max_length", return_tensors="pt").to(device)
        feat = as_tensor(model.get_text_features(**inputs))
        return torch.nn.functional.normalize(feat, dim=-1)  # (n_texts, dim)


def embed_texts(model, processor, categories, device):
    import torch
    from src.embedder import PROMPT_TEMPLATES      # one copy, the Embedder's
    with torch.no_grad():                          # (D-R1-1)
        embeds = []
        for cat in categories:
            prompts = [t.format(cat) for t in PROMPT_TEMPLATES]
            feat = embed_raw(model, processor, prompts, device).mean(0)
            embeds.append(torch.nn.functional.normalize(feat, dim=-1))
        return torch.stack(embeds)  # (n_categories, dim)


EMBED_BATCH = 0   # 0 = one call per image list, which is what --views implies


def embed_images(model, processor, images, device, batch=None):
    """Row-normalised embeddings, (n_images, dim).

    batch caps how many images go to the GPU at once; 0/None sends the whole
    list, which is the historical behaviour and fine at 16-40 images (measured
    peak 2.5 GB of a 7.8 GB card). Raise it to keep the GPU busier when the
    image list is long, lower it if SigLIP has to share the card."""
    import torch
    with torch.no_grad():
        batch = batch or EMBED_BATCH or len(images)
        out = []
        for i in range(0, len(images), batch):
            inputs = processor(images=images[i:i + batch], return_tensors="pt").to(device)
            out.append(as_tensor(model.get_image_features(**inputs)))
        feat = out[0] if len(out) == 1 else torch.cat(out)
        return torch.nn.functional.normalize(feat, dim=-1)


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


def cache_key(f, args, up_token, root):
    # The path is relative to the collection root (identity.py) so the library
    # can change drives without re-embedding everything.
    stat = f.stat()
    return cache_key_from_identity(
        f"{identity.rel_path(f, root)}|{identity.mtime_key(stat)}|{stat.st_size}",
        args, up_token)


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
    # the Embedder owns the default (D-R1-1): one copy, imported here rather
    # than duplicated, and deferred because src.embedder imports torch
    from src.embedder import DEFAULT_MODEL
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                        help="torch.compile the image forward: ~1.09x embed throughput for "
                             "~1e-03 embedding drift, which flips only coin-toss margins "
                             "(eval/compile_flips.py: 1 of 341, at margin 4.3e-06). "
                             "Compiled embeddings cache under their own keys, so the two "
                             "numeric regimes never mix")
    parser.add_argument("--up-axis", choices=["auto", "z", "y"], default="auto",
                        help="up axis of source meshes; auto detects the flat print base (default)")
    parser.add_argument("--cache-dir", default="embed-cache",
                        help="directory of cached per-file image embeddings; reruns with new "
                             "categories skip rendering/embedding entirely (set '' to disable)")
    # shared because every tool here walks the collection, and a stale list is
    # not merely slow — migrate_cache_keys drops entries for files it cannot see
    parser.add_argument("--rescan", action="store_true",
                        help="re-walk the input directory instead of using the cached file list")


RUN_PARAMS_FILE = "run-params.json"
# What a classify run records for the tools that read its cache. Keys are
# argparse dests; anything not declared by a given tool's parser is ignored.
# "pool" is deliberately absent: it is a scoring-time choice, not cache
# identity, and letting the classifier's afterthought default leak into
# test_categories overrode the REPL's own deliberate softmax default —
# querying happens there, so its default wins there.
RUN_PARAMS_KEYS = ("input", "views", "elevations", "render_size", "model",
                   "compile", "up_axis", "categories", "render_format",
                   "collection_root")

CACHE_META_FILE = "cache-meta.json"
# Bumped only when the *key scheme* changes incompatibly — never for
# byte-compatible additions like |compiled or |e:, which are designed to
# leave existing keys alone. That is why this is a hand-set integer and not a
# hash of the key format: an auto-derived stamp would fire on exactly the
# changes this repo makes carefully so it does not have to.
#   0 = unstamped (every cache from before the stamp existed): the up-token
#       elision, where deterministic poses keyed as the --up-axis string
#   1 = the up_str token (pose.embed_cache_token, review P2.3-B)
CACHE_VERSION = 1


def cache_version(cache_dir):
    """0 for any cache written before the stamp — i.e. every unstamped one."""
    p = Path(cache_dir) / CACHE_META_FILE
    return json.loads(p.read_text())["cache_version"] if p.exists() else 0


def stamp_cache_version(cache_dir):
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / CACHE_META_FILE).write_text(json.dumps({
        "cache_version": CACHE_VERSION,
        # informational only, never compared — see the CACHE_VERSION note
        "cache_key_format": "sha1(rel|mtime|size|views|render_size|up_token"
                            "|model|pv[|e:...][|compiled][|evN])",
    }, indent=2))


def require_cache_version(cache_dir):
    """Refuse a cache whose key scheme this code cannot read.

    A moved scheme does not error on its own — every lookup just misses, and
    the run silently re-renders and re-embeds the whole collection: hours,
    and real money once a pose entry is VLM-sourced. The stamp turns that
    into one line naming the fix. An empty cache is simply stamped current."""
    if not cache_dir:
        return
    v = cache_version(cache_dir)
    if v == CACHE_VERSION:
        return
    d = Path(cache_dir)
    # "populated" must include the pre-layout shape — root-level .npy with no
    # pose-cache.json or embeds/ (a forced --up-axis cache writes no pose
    # cache at all). Treating that as empty would stamp a genuinely
    # unmigrated cache as current, which is the exact failure this guard
    # exists to prevent (S2).
    if ((d / "pose-cache.json").exists() or (d / EMBEDS_SUBDIR).exists()
            or (d / RENDERS_SUBDIR).exists() or any(d.glob("*.npy"))):
        raise SystemExit(
            f"{cache_dir}: cache_version {v}, this code expects {CACHE_VERSION} — "
            f"every key would miss and the collection would re-embed from "
            f"scratch.\n  run: .venv/bin/python migrate_cache_keys.py "
            f"--cache-dir {cache_dir} --apply")
    stamp_cache_version(cache_dir)


def cache_root(inp, cache_dir, confirm=True, reanchor=False):
    """The root every cache key in `cache_dir` is taken relative to.

    Deliberately not just `collection_root(inp)`. The anchor belongs to the
    cache, not to the command line: running on one kit inside the library has
    to key the same way the whole-library run did, or the same file is indexed
    twice under two identities and re-rendered, re-embedded and re-arbitrated
    for the privilege.

    A mismatch stops to ask, because the two reasons for one are opposite: the
    library moved (re-key, free, everything still matches) or this cache
    belongs to a different collection (re-key, expensive, and the old entries
    are orphaned). Read-only tools pass confirm=False and only warn — they
    write nothing, and blocking a REPL on a prompt helps no one."""
    recorded = load_run_params(cache_dir).get("collection_root")
    root, note = identity.resolve_root(inp, recorded)
    if note == "subdir":
        print(f"cache keys stay anchored at {root} — this run is scoped to "
              f"{identity.collection_root(inp)}, but the cache is the library's")
    elif note in ("superdir", "mismatch"):
        if note == "superdir":
            why = (f"  every existing key is still valid under the wider root — it "
                   f"needs\n    {root and Path(recorded).relative_to(root)}/\n"
                   f"  on the front. migrate_cache_keys.py re-keys them; "
                   f"--reanchor without it orphans them.")
        else:
            gone = "" if Path(recorded).exists() else \
                " (which no longer exists, so this looks like the library moved)"
            why = f"  the recorded root{gone or ' still exists'}."
        print(f"\n  the cache in {cache_dir} was built against\n"
              f"    {recorded}\n  and you have asked for\n    {root}\n{why}")
        if reanchor:
            print("  --reanchor given; re-keying to the new root")
        elif not confirm:
            print("  read-only tool: using the root you asked for, which may miss "
                  "every cached entry")
        elif not sys.stdin.isatty():
            sys.exit("  refusing to re-key a cache without confirmation in a "
                     "non-interactive run — pass --reanchor if that is what you want")
        elif input("  re-key this cache to the new root? [y/N] ").strip().lower() \
                not in ("y", "yes"):
            sys.exit("  stopped; pass a path under the recorded root, or use a "
                     "separate --cache-dir for a different collection")
    return root


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
    # RUN_PARAMS_KEYS gates the read as well as the write: a key dropped from
    # the manifest must stop flowing even from run-params.json files that
    # recorded it back when it was one
    applied = {k: v for k, v in params.items() if k in dests and k in RUN_PARAMS_KEYS}
    parser.set_defaults(**applied)
    args = parser.parse_args()
    if applied:
        print(f"defaults from {Path(known.cache_dir) / RUN_PARAMS_FILE}: "
              + ", ".join(sorted(applied)) + " (command line overrides)")
    return args


def resolve_pose_vlm(args):
    """--pose-vlm to the backend the Poser is built with, announcing the choice.

    `ollama` is retired (2026-08-17, C-R1-4): the Arbiter is a thread pool with
    no inline arm, and a pooled ollama call would overlap SigLIP on the 4060 —
    10.1 s of model reload against 0.49 s of inference, this repo's one hard
    GPU constraint. So `auto` is gemini or nothing, and `VlmConfig` refuses the
    name at construction if it ever reaches it another way."""
    backend = args.pose_vlm
    if backend == "off":
        return None
    if backend == "auto":
        # gemini or nothing: it is the only arbiter measured to beat the
        # ensemble (43/44 against 40/44), where haiku/sonnet on a 256px sheet
        # score below running no arbiter at all. It bills per call — ~$0.30 for
        # a 602-model run at ~120 escalations — so the choice is announced.
        try:
            args.gemini_project = args.gemini_project or pose.gcloud_project()
            pose.gcloud_token()
            backend = "gemini"
        except Exception as e:
            print(f"pose VLM: gemini unavailable ({e}) — ambiguous poses keep "
                  f"the ensemble's answer")
            return None
    vlm_model = args.pose_vlm_model or pose.DEFAULT_VLM_MODELS.get(backend)
    if backend == "gemini":
        # Fail here rather than on the first ambiguous model, thousands of
        # renders into a run: resolving the project and minting a token are the
        # two things that go wrong, and both are cheap to check up front.
        # Explicit --pose-vlm gemini is an error if unavailable; auto already
        # returned above and never reaches this.
        try:
            args.gemini_project = args.gemini_project or pose.gcloud_project()
            pose.gcloud_token()
        except Exception as e:
            raise SystemExit(f"--pose-vlm gemini: {e}")
        print(f"pose VLM: {vlm_model} on Vertex AI, project {args.gemini_project} "
              f"— billed per escalation")
    else:
        print(f"pose VLM: {vlm_model or backend}")
    # The arbiter sheet scales each tile to SHEET_THUMB, and Image.thumbnail
    # never enlarges — so tiles rendered smaller than that sit padded in their
    # cells and the arbiter sees a smaller sheet than the number implies. Worth
    # saying out loud: sheet size is the knob that moved sonnet 10 of 44.
    if args.render_size < pose.SHEET_THUMB:
        print(f"  note: --render-size {args.render_size} is below the {pose.SHEET_THUMB}px "
              f"sheet tile, so the arbiter sees {args.render_size}px tiles padded into "
              f"{pose.SHEET_THUMB}px cells, not a {pose.SHEET_THUMB}px sheet")
    return backend


def main():
    parser = argparse.ArgumentParser()
    add_cache_args(parser, "STL file or directory of STL files "
                           "(defaults to the last run's directory)")
    parser.add_argument("--categories", default="categories.txt")
    parser.add_argument("--out", default="results.csv")
    parser.add_argument("--save-renders", action="store_true",
                        help="keep the render images for debugging, under "
                             "<cache-dir>/renders/<camera config>/ as "
                             "<stem>_<path hash>_view<i>.<ext>, plus <stem>_<path hash>"
                             "_pose.png for each model whose up axis the VLM had to "
                             "arbitrate (the hash keeps two models that share a "
                             "filename from overwriting each other)")
    parser.add_argument("--render-format", choices=sorted(RENDER_FORMATS), default="jpg",
                        help="encoding for --save-renders images (default jpg). Nothing "
                             "reads these back — the classifier embeds the in-memory "
                             "render — so lossy is safe here, and jpg encodes ~180x "
                             "faster and ~16x smaller than png at 2048 px")
    parser.add_argument("--reanchor", action="store_true",
                        help="accept a collection root that differs from the one this "
                             "cache was built against, re-keying every entry. Right after "
                             "the library moves, wrong when the cache belongs to another "
                             "collection")
    parser.add_argument("--pool", choices=["mean", "max", "softmax"], default="softmax",
                        help="how per-view scores combine: mean = whole-object consensus, "
                             "max = single-view features decide, softmax = in between")
    parser.add_argument("--pose-vlm", choices=["auto", "claude", "gemini", "off"],
                        default="auto",
                        help="arbiter for uncertain up detection: gemini on Vertex AI, "
                             "claude CLI, or off. auto (default) = gemini if gcloud ADC "
                             "resolves, else none. gemini-3.5-flash is the only arbiter "
                             "measured to beat the ensemble (43/44 against 40/44) and "
                             "bills ~$0.30 per full-collection run. `ollama` is retired: "
                             "the arbiter is a thread pool with no inline arm, and a "
                             "pooled ollama call would share the 4060 with SigLIP")
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
                        help="accepted and inert in v1: the render child loads each mesh "
                             "inline, and this becomes its loader_worker_count when the "
                             "child grows loader workers (actors_proposal.md migration "
                             "notes). Worth ~1-2%% of a run when it did apply")
    parser.add_argument("--arbiter-workers", type=int, default=8,
                        help="concurrent pose-VLM calls for network backends — the "
                             "Arbiter's window. The call averages 24s against 3-28s of "
                             "local work per model, so waiting inline leaves the run "
                             "mostly idle")
    parser.add_argument("--up-margin", type=float, default=pose.MARGIN_THRESHOLD,
                        help="escalate to the pose VLM when the ensemble's winning "
                             "candidate leads the runner-up by less than this (0-2). "
                             "Lower = fewer VLM calls")
    parser.add_argument("--skip-embed", action="store_true",
                        help="skip embedding and scoring the classification views; pose "
                             "resolution, including the SigLIP up-ensemble, still runs")
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
    # before cache_root: an unreadable cache should be one line of output,
    # not a re-anchor prompt followed by a refusal (S5)
    require_cache_version(args.cache_dir)
    root = cache_root(inp, args.cache_dir, reanchor=args.reanchor)
    # sticky, and only a directory run may set it: a loose file describes no
    # collection, and save_run_params drops None rather than overwriting
    args.collection_root = str(root) if inp.is_dir() else None
    with stage("walk"):
        files = load_file_list(inp, args.cache_dir, args.rescan) if inp.is_dir() else [inp]
    if not files:
        sys.exit(f"no STL files found under {inp}")
    n_views = total_views(args)
    print(f"{n_views} views per model: {args.views} azimuths at "
          f"{', '.join(f'{e:g}' for e in args.elevations)} degrees")
    categories = [l.strip() for l in open(args.categories) if l.strip()]

    # Every import below is deferred, and for one reason: src.done, src.embedder
    # and src.poser own torch, this module is re-imported by the spawned render
    # child (module docstring), and none of this runs there.
    from src import driver
    from src.arbiter import Arbiter
    from src.done import Done
    from src.driver import Admission, DriverConfig
    from src.embedder import Embedder
    from src.messages import CacheContext, Failure, RenderConfig
    from src.poser import Poser, VlmConfig
    from src.transport import MpQueueTransport

    vlm_backend = resolve_pose_vlm(args)
    print(f"loading {args.model} ...")
    with stage("model-load"):
        # the Embedder is the only owner of torch models: the fp16 load, the
        # --compile wrap on the image forward, the category text embeddings and
        # the four prompt banks all happen in here
        embedder = Embedder(categories, args.model,
                            compile_image_forward=args.compile,
                            embed_batch=args.embed_batch)
    if args.compile:
        print("torch.compile on the image forward; embeddings keyed as a "
              "separate cache regime")

    # the .npy files sit in their own subdirectory: they are the bulk of the
    # entries, and keeping them out of the cache root leaves pose-cache.json,
    # the walk lists and run-params.json legible in a listing
    edir = embeds_dir(args.cache_dir)
    if edir:
        edir.mkdir(parents=True, exist_ok=True)
    rdir = renders_dir(args.cache_dir, args) if args.save_renders else None
    # route()'s read-only world. `poses` is THE store Done owns from here on —
    # the same object, never a copy, so route sees this run's resolutions (I9).
    ctx = CacheContext(poses=pose.load_pose_cache(args.cache_dir), embeds_dir=edir,
                       render_index=render_index(rdir), args=args, root=root)

    # tasks unbounded, results bounded at the admission window (I2/Q1): the
    # parent never blocks on a send, and admission is the only forward pressure
    tasks = MpQueueTransport()
    results = MpQueueTransport(maxsize=driver.WINDOW)
    # ONE Admission per run (P2) — `admitted` is the driver's field, `retired`
    # is Done's, and the driver takes this very object back off Done rather
    # than being handed a second one.
    done = Done(Admission(), embedder.text_embeds, ctx, tasks,
                categories=categories, front_embeds=embedder.front_T,
                back_embeds=embedder.back_T)
    arbiter = Arbiter(workers=args.arbiter_workers, wrap=driver.instrumented)
    poser = Poser(embedder.up_T, embedder.down_T, arbiter, done.record_pose,
                  VlmConfig(backend=vlm_backend, model=args.pose_vlm_model,
                            scratch_dir=args.cache_dir or ".",
                            project=args.gemini_project,
                            margin_threshold=args.up_margin,
                            # keeps each escalation's contact sheet beside that
                            # model's renders; the scratch copy is one fixed
                            # name every model overwrites
                            sheet_path=(lambda f: rdir / f"{render_key(f, root)}_pose.png")
                                       if rdir else None))
    child = driver.spawn_render_child(tasks, results, RenderConfig(
        render_size=args.render_size, views=args.views,
        elevations=tuple(args.elevations), save_renders_dir=rdir,
        render_format=args.render_format, budget_bytes=RESIDENT_BUDGET_BYTES,
        collection_root=root))
    try:
        driver.run(DriverConfig(
            # the bar advances on admission, so it runs at most WINDOW files
            # ahead of what has actually retired
            walker=tqdm(files, desc="classifying"), ctx=ctx,
            tasks=tasks, results=results, child=child, poser=poser,
            embedder=embedder, done=done, arbiter=arbiter,
            skip_embed=args.skip_embed))
    finally:
        # still describes the cache a partial pass partly filled
        save_run_params(args)
    errors = sum(1 for r in done.rows.values() if isinstance(r, Failure))
    print(f"wrote {args.out} ({len(done.rows)} rows"
          + (f", {errors} of them render errors" if errors else "") + ")")
    instrument.report()


# The two names whose home imports torch. `src.done` owns the scoring, so
# `pool_sims` and `view_config` live there — but the render child re-imports
# this file as `__mp_main__` (module docstring), and a module-scope
# `from src.done import ...` would hand it torch. PEP 562 defers the import to
# first use, so `from classify_stls import pool_sims` keeps resolving for
# test_categories.py, cluster_models.py and the evals, while the child — which
# touches neither name — never pays for it. One definition either way.
_FORWARDED = {"pool_sims": "src.done", "view_config": "src.done"}


def __getattr__(name):
    if name in _FORWARDED:
        import importlib
        return getattr(importlib.import_module(_FORWARDED[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main()
