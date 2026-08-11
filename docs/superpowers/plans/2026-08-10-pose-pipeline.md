# Canonical Pose Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every model renders upright with a known camera-facing view: up-axis detection gains a confidence score, an optional VLM arbitrates ambiguous cases, SigLIP picks the front view, and all decisions persist in a pose cache.

**Architecture:** New `pose.py` module holds up-detection scoring, the pose cache, front-view scoring, and VLM backends (ollama / claude CLI). `classify_stls.py` orchestrates: resolve pose → render → embed → classify, with the embedding-cache key derived so existing caches survive for heuristic-resolved files. `test_categories.py` and `cluster_models.py` become front-view-aware.

**Tech Stack:** Python (`.venv`), Open3D offscreen rendering, SigLIP via transformers, requests (already installed via transformers), ollama HTTP API, `claude` CLI, pytest.

**Spec:** `docs/superpowers/specs/2026-08-10-pose-pipeline-design.md`

## Global Constraints

- Run everything with `.venv/bin/python`; classifier runs need `HF_HUB_OFFLINE=1`.
- Install packages only via `uv pip install -p .venv/bin/python <pkg>` (uv venv has no pip module).
- No new runtime dependencies: stdlib + numpy/open3d/torch/PIL/requests (all present). pytest is dev-only.
- Thresholds (from spec): low-confidence when `runner_up/best > 0.6` (CLI `--up-conf`) or `best < 0.02` (code constant `ABS_SCORE_FLOOR`).
- Ollama endpoint `http://localhost:11434`; default vision model tag `gemma3` (CLI `--pose-vlm-model`). Ollama may have no model pulled — every VLM failure falls back to the heuristic; nothing may hard-fail.
- Pose cache entry shape (spec): `{"up": [x,y,z], "front_view": int, "confidence": ratio, "source": "heuristic"|"vlm"}` — `confidence` stores the runner-up/best ratio, **lower = more confident**.
- Never write to the real `embed-cache/` or `my_renders/` during tests/verification — use scratch dirs.
- Tests run as `.venv/bin/python -m pytest tests/ -v`. Repo must be green (imports work, tests pass) after every task.
- Commits: plain imperative messages matching repo style (e.g. "Add pose module with up-axis confidence"). Never add Claude as commit author; use default git settings.

---

### Task 1: `pose.py` — up-axis detection with confidence

Moves `detect_up_axis` out of `classify_stls.py` into a new `pose.py`, returning `(up, ratio, best_score)` instead of just the vector, plus the escalation predicate.

**Files:**
- Create: `pose.py`
- Create: `tests/test_pose.py`
- Modify: `classify_stls.py:50-71` (delete `UP_CANDIDATES` + `detect_up_axis`, import from pose), `classify_stls.py:91-92` (adapt call site)

**Interfaces:**
- Produces: `pose.UP_CANDIDATES: list[np.ndarray]` (6 unit axis vectors, +Z first); `pose.ABS_SCORE_FLOOR = 0.02`; `pose.detect_up_axis(mesh, n_samples=4000) -> (np.ndarray, float, float)` returning `(up, ratio, best_score)`; `pose.needs_arbiter(ratio, best_score, threshold=0.6) -> bool`.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Install pytest into the venv**

```bash
uv pip install -p .venv/bin/python pytest
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_pose.py`:

```python
import numpy as np
import open3d as o3d

import pose


def prepared(mesh):
    mesh.compute_vertex_normals()
    return mesh


def test_cone_up_is_decisive():
    # a cone has exactly one flat face (its base) -> unambiguous print base
    cone = prepared(o3d.geometry.TriangleMesh.create_cone(radius=0.5, height=2.0))
    up, ratio, best = pose.detect_up_axis(cone)
    assert np.allclose(up, [0, 0, 1])
    assert best > pose.ABS_SCORE_FLOOR
    assert ratio < 0.6
    assert not pose.needs_arbiter(ratio, best)


def test_rotated_cone_finds_new_up():
    cone = prepared(o3d.geometry.TriangleMesh.create_cone(radius=0.5, height=2.0))
    # Rx(-90deg) maps the model's up (+Z) onto +Y
    cone.rotate(o3d.geometry.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0)),
                center=(0, 0, 0))
    up, ratio, best = pose.detect_up_axis(cone)
    assert np.allclose(up, [0, 1, 0])


def test_cylinder_is_ambiguous():
    # flat cap on both ends: +Z and -Z score the same -> ratio ~ 1
    cyl = prepared(o3d.geometry.TriangleMesh.create_cylinder(radius=0.5, height=2.0))
    up, ratio, best = pose.detect_up_axis(cyl)
    assert abs(float(up @ np.array([0.0, 0.0, 1.0]))) > 0.99  # either cap wins
    assert ratio > 0.6
    assert pose.needs_arbiter(ratio, best)
```

Note: the spec's example says "box with flat bottom" for the decisive case, but a box has six flat faces and is maximally ambiguous under this scorer — a cone is the correct decisive fixture (single flat face). This implements the spec's *intent* (a decisive mesh).

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pose.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'pose'`

- [ ] **Step 4: Create `pose.py`**

```python
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
    pcd = mesh.sample_points_uniformly(n_samples)
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
```

- [ ] **Step 5: Wire `classify_stls.py` to the new module**

Delete lines 50–71 of `classify_stls.py` (the `UP_CANDIDATES` block and `detect_up_axis`). Add to the imports (after `from tqdm import tqdm`):

```python
import pose
from pose import detect_up_axis
```

In `render_views`, change the auto branch (was `up = detect_up_axis(mesh)`):

```python
    if up_axis == "auto":
        up, _, _ = detect_up_axis(mesh)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pose.py -v`
Expected: 3 passed. Also smoke-check the import wiring: `.venv/bin/python -c "import classify_stls"` (needs torch etc., should print nothing and exit 0).

- [ ] **Step 7: Commit**

```bash
git add pose.py tests/test_pose.py classify_stls.py
git commit -m "Extract up-axis detection into pose module with confidence ratio"
```

---

### Task 2: Pose cache and embedding-cache token

Per-file pose persistence in `pose-cache.json`, plus the rule that keeps existing embedding caches valid.

**Files:**
- Modify: `pose.py` (append functions)
- Modify: `tests/test_pose.py` (append tests)

**Interfaces:**
- Produces: `pose.file_identity(f: Path) -> str`; `pose.load_pose_cache(cache_dir) -> dict`; `pose.save_pose_cache(cache_dir, cache) -> None`; `pose.up_str(up) -> str` (e.g. `"0,0,1"`); `pose.embed_cache_token(entry: dict | None, up_axis_arg: str) -> str`.
- Consumes: nothing from other tasks.

Key design point (spec: "the rest of the warm cache survives"): the heuristic's answer is a deterministic function of the file, so files resolved by the heuristic keep the **legacy token** (`args.up_axis`, e.g. `"auto"`) and their existing cache entries stay valid. Only VLM overrides — where the render genuinely differs from what the heuristic would have produced — get a distinct `"vlm:<up_str>"` token. `resolve_up` (Task 5) only records `source: "vlm"` when the VLM *disagrees* with the heuristic, so a VLM confirmation also keeps the old cache entry.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pose.py`:

```python
def test_pose_cache_roundtrip(tmp_path):
    cache = {"some|identity": {"up": [0.0, 0.0, 1.0], "front_view": 2,
                               "confidence": 0.15, "source": "heuristic"}}
    pose.save_pose_cache(tmp_path, cache)
    assert pose.load_pose_cache(tmp_path) == cache
    assert pose.load_pose_cache(tmp_path / "missing") == {}
    assert pose.load_pose_cache(None) == {}  # cache disabled


def test_file_identity_changes_with_mtime_and_size(tmp_path):
    f = tmp_path / "a.stl"
    f.write_text("x")
    first = pose.file_identity(f)
    assert str(f.resolve()) in first
    f.write_text("xy")
    assert pose.file_identity(f) != first


def test_embed_cache_token_keeps_legacy_key_for_heuristic():
    heur = {"up": [0.0, 0.0, 1.0], "source": "heuristic"}
    vlm = {"up": [0.0, 1.0, 0.0], "source": "vlm"}
    assert pose.embed_cache_token(heur, "auto") == "auto"
    assert pose.embed_cache_token(None, "auto") == "auto"
    assert pose.embed_cache_token({"up": [0, 0, 1], "source": "forced"}, "z") == "z"
    assert pose.embed_cache_token(vlm, "auto") == "vlm:0,1,0"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pose.py -v`
Expected: the 3 new tests FAIL with `AttributeError: module 'pose' has no attribute 'save_pose_cache'` (etc.); Task 1 tests still pass.

- [ ] **Step 3: Implement in `pose.py`**

Append:

```python
def file_identity(f):
    """Pose-cache key: same identity as the embedding cache (path+mtime+size)."""
    stat = f.stat()
    return f"{f.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"


def load_pose_cache(cache_dir):
    if not cache_dir:
        return {}
    p = Path(cache_dir) / "pose-cache.json"
    return json.loads(p.read_text()) if p.exists() else {}


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
    existing caches valid; only VLM overrides (renders differ from what the
    heuristic would produce) get their own token."""
    if entry and entry.get("source") == "vlm":
        return "vlm:" + up_str(entry["up"])
    return up_axis_arg
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pose.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pose.py tests/test_pose.py
git commit -m "Add pose cache and embedding-cache token rules"
```

---

### Task 3: Front-view scoring

SigLIP-based "which view faces the camera", plus the raw-text embedding helper it needs.

**Files:**
- Modify: `pose.py` (append prompts + scorer)
- Modify: `classify_stls.py:128-137` (refactor `embed_texts`, add `embed_raw`)
- Modify: `tests/test_pose.py` (append test)

**Interfaces:**
- Produces: `pose.FRONT_PROMPTS: list[str]`, `pose.BACK_PROMPTS: list[str]`; `pose.front_view_index(view_embeds, front_embeds, back_embeds) -> int` (all args row-normalized 2-D numpy arrays); `classify_stls.embed_raw(model, processor, texts, device) -> torch.Tensor` of shape `(len(texts), dim)`, row-normalized.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pose.py`:

```python
def test_front_view_index_picks_frontmost():
    front = np.array([[1.0, 0.0]])
    back = np.array([[0.0, 1.0]])
    views = np.array([[0.0, 1.0],    # back-facing
                      [0.7, 0.7],
                      [1.0, 0.0],    # front-facing
                      [0.7, -0.7]])
    assert pose.front_view_index(views, front, back) == 2


def test_front_prompts_defined():
    assert pose.FRONT_PROMPTS and pose.BACK_PROMPTS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pose.py -v`
Expected: new tests FAIL with `AttributeError`.

- [ ] **Step 3: Implement in `pose.py`**

Append:

```python
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
```

- [ ] **Step 4: Refactor `classify_stls.py` text embedding**

Replace `embed_texts` (lines 128–137) with:

```python
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
```

(Behavior-identical for `embed_texts`: same tokenize → normalize → mean-over-templates → normalize sequence.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pose.py -v` — all pass.
Then: `.venv/bin/python -c "import classify_stls"` — exit 0.

- [ ] **Step 6: Commit**

```bash
git add pose.py tests/test_pose.py classify_stls.py
git commit -m "Add SigLIP front-view scoring and raw-text embedding helper"
```

---

### Task 4: VLM arbiter (contact sheet, ollama, claude CLI)

**Files:**
- Modify: `pose.py` (append; add `base64`, `io`, `subprocess` imports and `from PIL import Image, ImageDraw`)
- Modify: `tests/test_pose.py` (append tests; add `import pytest`, `from PIL import Image`)

**Interfaces:**
- Produces: `pose.make_contact_sheet(tiles: list[Image], thumb=256, cols=3) -> Image`; `pose.parse_tile_answer(text: str, n_tiles: int) -> int | None` (0-based); `pose.ollama_available() -> bool`; `pose.ask_vlm_up(tiles, backend: str, scratch_dir, vlm_model="gemma3") -> int | None` (0-based candidate index, `None` = fall back to heuristic); `pose.OLLAMA_URL`, `pose.UP_PROMPT`.
- Consumes: nothing from other tasks (Task 5 renders the tiles).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pose.py` (add `import pytest` and `from PIL import Image` at the top of the file):

```python
def test_parse_tile_answer():
    assert pose.parse_tile_answer('{"tile": 3}', 6) == 2
    assert pose.parse_tile_answer('The answer is {"tile": 1}.', 6) == 0
    assert pose.parse_tile_answer('{"tile": 9}', 6) is None
    assert pose.parse_tile_answer('{"tile": 0}', 6) is None
    assert pose.parse_tile_answer("no json here", 6) is None
    assert pose.parse_tile_answer('{"tile": "two"}', 6) is None


def test_make_contact_sheet_grid():
    tiles = [Image.new("RGB", (512, 512), "gray") for _ in range(6)]
    sheet = pose.make_contact_sheet(tiles, thumb=100, cols=3)
    assert sheet.size == (300, 200)  # 3x2 grid of 100px tiles


requires_ollama = pytest.mark.skipif(not pose.ollama_available(),
                                     reason="ollama not running")


@requires_ollama
def test_ask_vlm_up_live_transport(tmp_path):
    # exercises the real request/parse path; semantic quality isn't asserted,
    # and a missing/unpulled model must degrade to None, never raise
    tiles = [Image.new("RGB", (64, 64), "gray") for _ in range(6)]
    idx = pose.ask_vlm_up(tiles, "ollama", tmp_path)
    assert idx is None or 0 <= idx < 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pose.py -v`
Expected: new tests FAIL with `AttributeError` (the skipif line itself fails at collection because `ollama_available` doesn't exist yet — that's the expected failure mode).

- [ ] **Step 3: Implement in `pose.py`**

Add to the imports at the top: `import base64`, `import io`, `import subprocess`, `from PIL import Image, ImageDraw`. Then append:

```python
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
    resp = requests.post(f"{OLLAMA_URL}/api/chat", timeout=120, json={
        "model": model,
        "stream": False,
        "format": {"type": "object", "properties": {"tile": {"type": "integer"}},
                   "required": ["tile"]},
        "messages": [{"role": "user", "content": UP_PROMPT,
                      "images": [base64.b64encode(png_bytes).decode()]}],
    })
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


def ask_vlm_up(tiles, backend, scratch_dir, vlm_model="gemma3"):
    """Ask the VLM which candidate orientation is upright. One retry on a
    bad/failed answer, then None — the caller keeps the heuristic guess.
    The pipeline never hard-fails because of the VLM."""
    sheet = make_contact_sheet(tiles)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pose.py -v`
Expected: all pass; `test_ask_vlm_up_live_transport` runs (ollama is up on this machine) or is skipped — either is green. With no gemma model pulled yet, ollama returns an error → `ask_vlm_up` returns `None` → assertion holds.

- [ ] **Step 5: Commit**

```bash
git add pose.py tests/test_pose.py
git commit -m "Add VLM up-axis arbiter with ollama and claude CLI backends"
```

---

### Task 5: Pipeline integration in `classify_stls.py`

Restructure render entry points, resolve pose per file (cache → heuristic → VLM), key the embedding cache on the resolved pose, record pose columns in the CSV, add CLI flags. Also updates `test_categories.py`'s cache-key computation so the repo stays green (its front-view display is Task 6).

**Files:**
- Modify: `classify_stls.py` (render split, `resolve_up`, `cache_key`, `main`, CLI)
- Modify: `test_categories.py:32-44` (`load_embedding_matrix` key computation), `test_categories.py:22` (imports)

**Interfaces:**
- Consumes: everything `pose.py` produces (Tasks 1–4): `detect_up_axis`, `needs_arbiter`, `file_identity`, `load_pose_cache`, `save_pose_cache`, `up_str`, `embed_cache_token`, `front_view_index`, `FRONT_PROMPTS`, `BACK_PROMPTS`, `ask_vlm_up`, `ollama_available`, `UP_CANDIDATES`.
- Produces: `classify_stls.load_mesh(mesh_path) -> TriangleMesh` (raises `ValueError("no triangles")`); `classify_stls.render_views(renderer, mesh, n_views, elevation_deg=20) -> list[Image]` (mesh must already be Z-up rotated — signature change); `classify_stls.render_up_candidate_tiles(renderer, mesh) -> list[Image]` (6 tiles, one per `UP_CANDIDATES`); `classify_stls.resolve_up(mesh, args, get_renderer, vlm_backend) -> (np.ndarray, float, str)` (`source` is `"heuristic"` or `"vlm"`); `classify_stls.cache_key(f, args, up_token) -> str` (signature change); CLI flags `--pose-vlm {auto,ollama,claude,off}` (default auto), `--pose-vlm-model` (default gemma3), `--up-conf` (default 0.6); CSV columns `up`, `pose_conf`, `pose_source`, `front_view`.

- [ ] **Step 1: Split mesh loading from rendering**

Replace `render_views` (`classify_stls.py:86-125`) with:

```python
def load_mesh(mesh_path):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles():
        raise ValueError("no triangles")
    mesh.compute_vertex_normals()
    return mesh


def render_views(renderer, mesh, n_views, elevation_deg=20):
    """Render n azimuth views. The mesh must already be rotated into Z-up
    world space (the light rig and camera 'up' assume it)."""
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


def render_up_candidate_tiles(renderer, mesh):
    """One render per candidate up (fixed azimuth) for the VLM contact sheet."""
    tiles = []
    for up in pose.UP_CANDIDATES:
        m = o3d.geometry.TriangleMesh(mesh)
        m.rotate(rotation_to_z_up(up), center=(0, 0, 0))
        tiles.append(render_views(renderer, m, 1)[0])
    return tiles


def resolve_up(mesh, args, get_renderer, vlm_backend):
    """Resolve the up axis for --up-axis auto: heuristic first, VLM arbiter
    for low-confidence cases. Returns (up, ratio, source); source is "vlm"
    only when the VLM *disagrees* with the heuristic (a VLM confirmation
    keeps source "heuristic" so the embedding-cache key is unchanged)."""
    up, ratio, best = detect_up_axis(mesh)
    if vlm_backend and pose.needs_arbiter(ratio, best, args.up_conf):
        tiles = render_up_candidate_tiles(get_renderer(), mesh)
        idx = pose.ask_vlm_up(tiles, vlm_backend, args.cache_dir or ".",
                              args.pose_vlm_model)
        if idx is not None and not np.allclose(pose.UP_CANDIDATES[idx], up):
            return pose.UP_CANDIDATES[idx], ratio, "vlm"
    return up, ratio, "heuristic"
```

Also remove the now-unused `from pose import detect_up_axis`-adjacent shim if Task 1 left one — imports should now read `import pose` and `from pose import detect_up_axis`.

- [ ] **Step 2: Re-key the embedding cache on the resolved pose**

Replace `cache_key` (`classify_stls.py:209-213`):

```python
def cache_key(f, args, up_token):
    stat = f.stat()
    # "pv" = per-view cache format: (n_views, dim) instead of one pooled vector.
    # up_token is "auto"/"z"/"y" for deterministic poses (legacy-compatible)
    # and "vlm:<x,y,z>" when a VLM override changed the render.
    raw = f"{f.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{args.views}|{args.render_size}|{up_token}|{args.model}|pv"
    return hashlib.sha1(raw.encode()).hexdigest()
```

- [ ] **Step 3: Add CLI flags**

After the `--pool` argument in `main`:

```python
    parser.add_argument("--pose-vlm", choices=["auto", "ollama", "claude", "off"],
                        default="auto",
                        help="arbiter for low-confidence up detection: local ollama "
                             "vision model, claude CLI, or off (auto = ollama if reachable)")
    parser.add_argument("--pose-vlm-model", default="gemma3",
                        help="ollama model name used by --pose-vlm")
    parser.add_argument("--up-conf", type=float, default=0.6,
                        help="up-detection ambiguity threshold: runner-up/best flat-base "
                             "score ratio above this escalates to the pose VLM")
```

- [ ] **Step 4: Rewrite the per-file loop in `main`**

Replace the section from `renderer = None  # created lazily on first cache miss` through the end of the `for f in tqdm(...)` loop with:

```python
    renderer = None  # created lazily on first render

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    hits = 0

    rdir = Path(args.save_renders) if args.save_renders else None

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

    def get_renderer():
        nonlocal renderer
        if renderer is None:
            renderer = make_renderer(args.render_size)
        return renderer

    rows = []
    try:
        for f in tqdm(files, desc="classifying"):
            mesh = None
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
                    up, ratio, source = resolve_up(mesh, args, get_renderer, vlm_backend)
                    entry = {"up": [float(v) for v in up],
                             "confidence": round(ratio, 4), "source": source}
                    pose_cache[pose.file_identity(f)] = entry

            token = pose.embed_cache_token(entry, args.up_axis)
            cache_file = cache_dir / f"{cache_key(f, args, token)}.npy" if cache_dir else None
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
                    try:
                        if mesh is None:
                            mesh = load_mesh(f)
                        mesh.rotate(rotation_to_z_up(np.array(entry["up"])), center=(0, 0, 0))
                        images = render_views(get_renderer(), mesh, args.views)
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
        # interrupted cold passes keep their (expensive) pose resolutions
        if args.up_axis == "auto":
            pose.save_pose_cache(args.cache_dir, pose_cache)
```

Update the CSV fields line to:

```python
    fields = ["file", "top1", "score1", "top2", "score2", "top3", "score3",
              "up", "pose_conf", "pose_source", "front_view"]
```

Notes:
- Forced (`z`/`y`) poses are computed inline, never persisted — the pose cache holds only `auto` resolutions, so switching `--up-axis` flags between runs can't pollute it.
- Cache-hit files never load the mesh: pose comes from `pose-cache.json`, front comes from cached embeddings. First run after this change loads each mesh once to fill the pose cache (embedding cache still hits).

- [ ] **Step 5: Update `test_categories.py` key computation**

Change the import line (`test_categories.py:22`) to:

```python
import pose
from classify_stls import as_tensor, cache_key, embed_texts, load_file_list, pool_sims
```

In `load_embedding_matrix`, load the pose cache and compute per-file tokens:

```python
def load_embedding_matrix(files, args):
    cache_dir = Path(args.cache_dir)
    poses = pose.load_pose_cache(args.cache_dir)
    vecs, kept, missing = [], [], 0
    for f in files:
        token = pose.embed_cache_token(poses.get(pose.file_identity(f)), args.up_axis)
        p = cache_dir / f"{cache_key(f, args, token)}.npy"
        if p.exists():
            vecs.append(np.load(p))
            kept.append(f)
        else:
            missing += 1
    if not vecs:
        raise SystemExit("no cached embeddings found — run classify_stls.py first")
    return np.stack(vecs).astype(np.float32), kept, missing  # (n_files, n_views, dim)
```

(`cluster_models.py` imports `load_embedding_matrix` from here, so it stays green with no changes in this task.)

- [ ] **Step 6: Run unit tests and import checks**

Run: `.venv/bin/python -m pytest tests/ -v` — all pass.
Run: `.venv/bin/python -c "import classify_stls, test_categories, cluster_models"` — exit 0.

- [ ] **Step 7: Integration run on test-stls (scratch dirs only)**

```bash
mkdir -p /tmp/pose-itest
HF_HUB_OFFLINE=1 .venv/bin/python classify_stls.py test-stls \
  --out /tmp/pose-itest/results.csv --cache-dir /tmp/pose-itest/cache \
  --save-renders /tmp/pose-itest/renders --pose-vlm off
```

Verify:
- `/tmp/pose-itest/results.csv` has the 4 new columns filled for all 3 files (blocky_building, bunny, torus).
- torus has `pose_conf` near 1.0 (ambiguous) and `pose_source` `heuristic` (VLM off ⇒ fallback).
- `/tmp/pose-itest/cache/pose-cache.json` exists with 3 entries, each with `up`, `confidence`, `source`, `front_view`.
- Re-run the same command: output reports `3 from embedding cache` (pose + embedding caches both hit).

- [ ] **Step 8: Commit**

```bash
git add classify_stls.py test_categories.py
git commit -m "Resolve pose per file: cached, confidence-gated, VLM-arbitrated; pose columns in results"
```

---

### Task 6: Front-aware display in REPL and cluster sheets

**Files:**
- Modify: `test_categories.py` (`display` inside `main`)
- Modify: `cluster_models.py` (`contact_sheet` + its call site in `main`)

**Interfaces:**
- Consumes: `pose.load_pose_cache`, `pose.file_identity` (Task 2); `front_view` entries written by Task 5.
- Produces: no new APIs — display behavior only. `contact_sheet` gains a `poses` dict parameter (default `None` ⇒ view 0, preserving old behavior).

- [ ] **Step 1: REPL links open the front view**

In `test_categories.py` `main`, after `renders_dir = Path(args.renders_dir)`, add:

```python
    poses = pose.load_pose_cache(args.cache_dir)
```

and change `display` to:

```python
    def display(f):
        rel = str(f.relative_to(root)) if f.is_relative_to(root) else str(f)
        front = poses.get(pose.file_identity(f), {}).get("front_view", 0)
        render = renders_dir / f"{f.stem}_view{front}.png"
        target = render.resolve() if render.exists() else f
        return link(target, rel.replace("/No Supports", "").removesuffix(".stl"))
```

- [ ] **Step 2: Cluster contact sheets lead with the front view**

In `cluster_models.py`, add `import pose` to the imports, change `contact_sheet` to:

```python
def contact_sheet(members, renders_dir, out_path, thumb=160, cols=6, max_tiles=36,
                  poses=None):
    tiles = []
    for f in members[:max_tiles]:
        front = (poses or {}).get(pose.file_identity(f), {}).get("front_view", 0)
        img_path = renders_dir / f"{f.stem}_view{front}.png"
        if not img_path.exists():
            img_path = renders_dir / f"{f.stem}_view0.png"
        if img_path.exists():
            im = Image.open(img_path)
            im.thumbnail((thumb, thumb))
            tiles.append(im)
    if not tiles:
        return False
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
    sheet.save(out_path)
    return True
```

In `cluster_models.py`'s `main`, load `poses = pose.load_pose_cache(args.cache_dir)` next to where the embedding matrix is loaded, and pass `poses=poses` at the `contact_sheet(...)` call site (read the file to find it — it is below the excerpt shown here; keep all other arguments unchanged).

- [ ] **Step 3: Verify against the Task 5 scratch run**

```bash
echo q | HF_HUB_OFFLINE=1 .venv/bin/python test_categories.py test-stls \
  --cache-dir /tmp/pose-itest/cache --renders-dir /tmp/pose-itest/renders
HF_HUB_OFFLINE=1 .venv/bin/python cluster_models.py test-stls --k 2 \
  --cache-dir /tmp/pose-itest/cache --renders-dir /tmp/pose-itest/renders \
  --sheets-dir /tmp/pose-itest/sheets --out /tmp/pose-itest/clusters.csv
.venv/bin/python -m pytest tests/ -v
```

Expected: REPL starts, reports "3 models with cached embeddings", exits on `q`. Cluster run writes sheets without error. All tests pass. (Note: `cluster_models.py` may use different flag names for cache dir — read its argparse block and adapt the command.)

- [ ] **Step 4: Commit**

```bash
git add test_categories.py cluster_models.py
git commit -m "Open and lead with the detected front view in REPL links and cluster sheets"
```

---

### Task 7: Eyeball pass, docs, cleanup

**Files:**
- Modify: `classify_stls.py:1-9` (module docstring), `LEARNINGS.md` (append pose findings)

**Interfaces:** none — documentation and human verification.

- [ ] **Step 1: Eyeball the rendered poses**

Open `/tmp/pose-itest/renders/` and confirm: blocky_building sits upright; bunny sits upright; torus lies flat (either face — ambiguous is acceptable). For each file, note which `_view{front_view}.png` was chosen as front (from `/tmp/pose-itest/results.csv`) and confirm it's a reasonable "front" for the bunny (face visible). If the collection drive (`/run/media/masa/Files and S/STL/`) is mounted, spot-check a handful of known models (witch, gravedigger, barrel) the same way with a scratch cache dir; if it is not mounted, skip — do not block on it.

- [ ] **Step 2: If ollama has a vision model pulled by now, live-test the arbiter**

```bash
ollama list   # check for gemma3 / other vision model
HF_HUB_OFFLINE=1 .venv/bin/python classify_stls.py test-stls/torus.stl \
  --out /tmp/pose-itest/torus.csv --cache-dir /tmp/pose-itest/vlmcache \
  --pose-vlm ollama --pose-vlm-model gemma3
```

Expected: run completes; `pose_source` in the CSV is `vlm` (if the VLM disagreed with the heuristic) or `heuristic` (agreed / failed — both fine). If no vision model is pulled, skip this step; the skipif unit test already covers transport.

- [ ] **Step 3: Update the module docstring**

Extend `classify_stls.py`'s docstring (lines 1–9) with two sentences: up detection now reports confidence and ambiguous meshes can be arbitrated by a local VLM (`--pose-vlm`); the front-facing view index is recorded per file (`front_view` column) and poses persist in `<cache-dir>/pose-cache.json`.

- [ ] **Step 4: Record learnings**

Append a short section to `LEARNINGS.md` under the rendering section: front view as metadata (never re-render), the embedding-cache token rule (heuristic = deterministic ⇒ legacy key survives; only VLM overrides re-key), and whatever the eyeball/VLM steps actually showed.

- [ ] **Step 5: Final verification and commit**

```bash
.venv/bin/python -m pytest tests/ -v
git add classify_stls.py LEARNINGS.md
git commit -m "Document pose pipeline; record pose-cache learnings"
```

---

## Self-Review Notes

- Spec coverage: tier 1 (Task 1), tier 2 (Task 3 + wiring in Task 5), tier 3 (Task 4 + `resolve_up` in Task 5), pose cache + cache-key rule (Tasks 2, 5), CLI/CSV (Task 5), REPL/sheets (Task 6), testing incl. skipif + eyeball (Tasks 1–7). "Never hard-fails on VLM" — `ask_vlm_up` catches all exceptions, returns None.
- Deviations from spec, both intentional: (1) decisive-mesh test fixture is a cone, not a box — a box is ambiguous under the flat-base scorer; (2) added `--pose-vlm-model` so a differently-tagged ollama pull works without code edits.
- Type consistency: `cache_key(f, args, up_token)` used identically in Tasks 5 (classify + test_categories); `entry` dict shape matches the spec example everywhere; `ask_vlm_up` returns 0-based, `parse_tile_answer` converts from 1-based labels.
