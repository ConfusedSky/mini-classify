"""Pose resolution for STL renders: up-axis detection with confidence, a
SigLIP tie-break over the up-candidate tiles, front-view scoring, a per-file
pose cache, and a VLM arbiter for ambiguous cases. classify_stls.py
orchestrates; this module never imports classify_stls (no rendering / model
code here) — the scorers here take embeddings as arguments and never load a
model."""
import base64
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

UP_CANDIDATES = [np.array(u, dtype=float) for u in
                 [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]]

ABS_SCORE_FLOOR = 0.02  # best flat-base score below this = "no print base found"
SAMPLE_SEED = 0

# Escalate to the VLM when the *ensemble* is unsure, not when geometry is.
# 0.45 was picked on the `orig` set and read on the holdout (21/21 there, 43/44
# pooled, firing on ~20% of models against the old gate's ~55%). See LEARNINGS.
MARGIN_THRESHOLD = 0.45

# How hard geometry's vote is attenuated when it found no print base. min-max
# maps geometry's *ratio* to its margin and is blind to magnitude, so a mesh
# with no flat base anywhere still votes confidently on evidence orders of
# magnitude under ABS_SCORE_FLOOR. Squaring is not tuning for its own sake: the
# vote has to be nearly silenced before it stops overriding SigLIP (the case
# this fixes lands at w=0.14), which is why a hard switch behaves identically.
GEO_FLOOR_POWER = 2

# Bumped when a change makes previously cached poses wrong. Entries written by
# an older version are dropped on load and re-resolved: v2 = the four-view
# ensemble and the margin gate; v3 = geometry attenuated by its base evidence.
POSE_CACHE_VERSION = 3


def up_axis_scores(mesh, n_samples=4000):
    """Flat print-base evidence per candidate up: the fraction of sampled
    surface that is both in the bottom 2% height slab and facing down.

    Seeded, because the winner can rest on ~30 of the 4000 points and an
    unseeded draw moves picks between runs on identical input — which would
    make the pose cache irreproducible and the ensemble below unstable."""
    o3d.utility.random.seed(SAMPLE_SEED)
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
    return np.array(scores)


def rank_up_scores(scores):
    """(index, ratio, best_score) — ratio = runner_up/best, lower is more
    confident (symmetric meshes like a barrel come out ~1.0)."""
    order = np.argsort(scores)[::-1]
    best, runner = float(scores[order[0]]), float(scores[order[1]])
    return int(order[0]), (runner / best if best > 0 else 1.0), best


def detect_up_axis(mesh, n_samples=4000):
    """Geometry-only up axis. Returns (up, ratio, best_score)."""
    idx, ratio, best = rank_up_scores(up_axis_scores(mesh, n_samples))
    return UP_CANDIDATES[idx], ratio, best


def needs_arbiter(ratio, best_score, threshold=0.6):
    """Geometry's own doubt. Superseded by needs_arbiter_margin for the
    production gate — kept as the fallback when SigLIP is unavailable
    (--skip-embed), where there is no ensemble margin to ask about."""
    return ratio > threshold or best_score < ABS_SCORE_FLOOR


def needs_arbiter_margin(margin, threshold=MARGIN_THRESHOLD):
    """The *ensemble's* doubt: how far the winning candidate leads the runner-up
    in the combined score. Geometry having no print base says nothing about
    whether the combination is unsure — that is precisely the population SigLIP
    was added to carry, and gating on it escalated models the ensemble already
    had right (17 of 18, measured)."""
    return margin < threshold


def file_identity(f):
    """Pose-cache key: same identity as the embedding cache (path+mtime+size)."""
    stat = f.stat()
    return f"{f.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"


def load_pose_cache(cache_dir):
    """Cached poses, minus any written by an older POSE_CACHE_VERSION.

    Stale entries are dropped rather than migrated: a pose decided under a
    different ensemble or a different escalation gate is not the pose this
    version would produce, and silently trusting it would mean the collection
    disagrees with every measurement made against it."""
    if not cache_dir:
        return {}
    p = Path(cache_dir) / "pose-cache.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    fresh = {k: v for k, v in raw.items() if v.get("v") == POSE_CACHE_VERSION}
    if len(fresh) < len(raw):
        print(f"pose cache: {len(raw) - len(fresh)} of {len(raw)} entries predate "
              f"v{POSE_CACHE_VERSION} and will be re-resolved")
    return fresh


def save_pose_cache(cache_dir, cache):
    if not cache_dir:
        return
    p = Path(cache_dir) / "pose-cache.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache))


def up_str(up):
    return ",".join(f"{float(v):g}" for v in up)


def embed_cache_token(entry, up_axis_arg):
    """Embedding-cache key token for a file's resolved pose. Heuristic poses
    are a deterministic function of the file, so the legacy token keeps
    existing caches valid; only sources that *moved* the pose off the geometry
    answer — a VLM override or an ensemble override — render differently and
    get their own token."""
    source = entry.get("source") if entry else None
    if source in ("vlm", "ensemble"):
        return f"{source}:" + up_str(entry["up"])
    return up_axis_arg


FRONT_PROMPTS = [
    "the front of a miniature figurine, facing the camera",
    "a miniature figurine seen from the front, face and chest visible",
]
BACK_PROMPTS = [
    "the back of a miniature figurine, facing away from the camera",
    "a miniature figurine seen from behind, back of the head visible",
]


def front_view_index(view_embeds, front_embeds, back_embeds):
    """Index of the view that best faces the camera. Front is metadata: this
    never triggers a re-render, it just names one of the existing views."""
    score = ((view_embeds @ front_embeds.T).mean(1)
             - (view_embeds @ back_embeds.T).mean(1))
    return int(np.argmax(score))


# Deliberately not phrased in terms of heads and feet: roughly half the
# collection is terrain and scatter, where anatomy probes score 0/12.
UPRIGHT_PROMPTS = [
    "a 3D printed model sitting the right way up on a table",
]
TOPPLED_PROMPTS = [
    "a 3D printed model tipped onto its side",
    "a 3D printed model turned upside down",
]


def upright_scores(tile_embeds, upright_embeds, toppled_embeds):
    """Per-candidate upright score for the up-candidate tiles. The toppled
    probes are not optional padding: without something to contrast against,
    raw similarity is near-flat across the six tiles and the argmax is noise."""
    return ((tile_embeds @ upright_embeds.T).mean(1)
            - (tile_embeds @ toppled_embeds.T).mean(1))


def _unit(v):
    """Min-max to [0,1]; all-equal input collapses to zeros (no vote)."""
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def combine_up_scores(geo_scores, siglip_scores):
    """Index of the up candidate favoured by geometry and SigLIP together.

    The two are in different units — a flat-base area fraction against a
    difference of cosine similarities — so each is min-maxed before the mean.
    That is not merely a scale fix: geometry's weakest candidate is almost
    always 0, so min-max maps its runner-up to runner/best, exactly the ratio
    rank_up_scores reports. Geometry therefore votes with a ~1.0 margin when it
    has real base evidence and with a ~0.0 margin when it is guessing, handing
    those models to SigLIP without either method being thresholded."""
    return combine_up(geo_scores, siglip_scores)[0]


def geo_weight(geo_scores):
    """How loudly geometry gets to vote, from how much base evidence it has.

    1.0 once a real print base is found, falling to ~0 when the best score is
    far under ABS_SCORE_FLOOR. This is *not* the absolute-scaling scheme that
    lost at 20/23 — that one replaced min-max with clip(geo/floor), saturating
    every candidate above the floor and destroying the margin. Here min-max is
    untouched inside the vote; only its amplitude changes."""
    best = float(np.max(geo_scores))
    return min(1.0, best / ABS_SCORE_FLOOR) ** GEO_FLOOR_POWER


def combine_up(geo_scores, siglip_scores):
    """(index, margin) — the ensemble's pick and how far it leads the runner-up.

    The margin is the gate's input, and it predicts error far better than
    geometry's confidence does: measured over 44 labelled models its median is
    1.31 where the ensemble is right and 0.22 where it is wrong.

    The margin is deliberately *not* rescaled to undo the geometry weight. A
    quieter geometry vote does compress the margin, and that is correct — less
    evidence should mean less certainty. Normalising it back looks tidier and
    measurably stops escalating a model the ensemble gets wrong."""
    combined = geo_weight(geo_scores) * _unit(geo_scores) + _unit(siglip_scores)
    top = np.sort(combined)[::-1]
    return int(np.argmax(combined)), float(top[0] - top[1])


OLLAMA_URL = "http://localhost:11434"

UP_PROMPT = (
    "Each numbered tile shows the same 3D model in a different orientation. "
    "Which tile shows the model standing upright, the way it would sit on a "
    'table? Answer with JSON only: {"tile": <number>}'
)


def make_contact_sheet(tiles, thumb=256, cols=3):
    """Grid of tiles labeled 1..n (red corner numbers) for the VLM prompt."""
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    draw = ImageDraw.Draw(sheet)
    for i, im in enumerate(tiles):
        im = im.copy()
        im.thumbnail((thumb, thumb))
        x, y = (i % cols) * thumb, (i // cols) * thumb
        sheet.paste(im, (x, y))
        draw.text((x + 8, y + 4), str(i + 1), fill="red")
    return sheet


def parse_tile_answer(text, n_tiles):
    """Extract {"tile": n} from a model reply. Returns 0-based index or None."""
    try:
        tile = json.loads(text[text.index("{"):text.rindex("}") + 1])["tile"]
    except (ValueError, KeyError, TypeError):
        return None
    if isinstance(tile, int) and 1 <= tile <= n_tiles:
        return tile - 1
    return None


def ollama_available():
    import requests
    try:
        requests.get(f"{OLLAMA_URL}/api/version", timeout=2)
        return True
    except requests.RequestException:
        return False


def _ask_ollama(png_bytes, n_tiles, model):
    import requests
    payload = {
        "model": model,
        "stream": False,
        # thinking models put every token in the thinking field and return an
        # empty content (measured: 3859 tokens / 281 s vs 7 tokens / 25 s)
        "think": False,
        "format": {"type": "object", "properties": {"tile": {"type": "integer"}},
                   "required": ["tile"]},
        "messages": [{"role": "user", "content": UP_PROMPT,
                      "images": [base64.b64encode(png_bytes).decode()]}],
    }
    resp = requests.post(f"{OLLAMA_URL}/api/chat", timeout=300, json=payload)
    if resp.status_code == 400 and "think" in resp.text:
        del payload["think"]  # older servers/models reject the field entirely
        resp = requests.post(f"{OLLAMA_URL}/api/chat", timeout=300, json=payload)
    resp.raise_for_status()
    return parse_tile_answer(resp.json()["message"]["content"], n_tiles)


def _ask_claude(sheet_path, n_tiles):
    out = subprocess.run(
        ["claude", "-p", f"Read the image at {sheet_path}. {UP_PROMPT}",
         "--output-format", "json", "--max-turns", "3"],
        capture_output=True, text=True, timeout=180)
    if out.returncode != 0:
        return None
    return parse_tile_answer(json.loads(out.stdout).get("result", ""), n_tiles)


def ask_vlm_up(tiles, backend, scratch_dir, vlm_model="gemma4:26b", save_to=None):
    """Ask the VLM which candidate orientation is upright. One retry on a
    bad/failed answer, then None — the caller keeps the heuristic guess.
    The pipeline never hard-fails because of the VLM.

    save_to keeps a per-model copy of the sheet next to the saved renders. It
    is the same image the VLM was shown, written whether or not the answer
    parses, so a wrong pose can be read back off disk."""
    try:
        sheet = make_contact_sheet(tiles)
    except Exception as e:
        print(f"  pose VLM error ({backend}): {e}")
        return None
    if save_to is not None:
        try:
            save_to = Path(save_to)
            save_to.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(save_to)
        except OSError as e:  # a debug artifact must never fail the run
            print(f"  could not save pose sheet {save_to}: {e}")
    for _attempt in range(2):
        try:
            if backend == "ollama":
                buf = io.BytesIO()
                sheet.save(buf, format="PNG")
                idx = _ask_ollama(buf.getvalue(), len(tiles), vlm_model)
            else:  # claude
                sheet_path = Path(scratch_dir) / "pose-sheet.png"
                sheet.save(sheet_path)
                idx = _ask_claude(sheet_path, len(tiles))
        except Exception as e:
            print(f"  pose VLM error ({backend}): {e}")
            idx = None
        if idx is not None:
            return idx
    return None
