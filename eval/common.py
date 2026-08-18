"""Shared setup for the pose-evaluation harnesses: labels, scoring, and the
cached render sets every scorer reads.

These scripts measure the up-axis pipeline against hand-labelled ground truth.
What "the real code path" means here, precisely, since it used to be claimed
more broadly than it was true:

* **Pixels and embeddings are production's.** Everything rendered below goes
  through `eval/rig.py` — `src.renderer.Renderer` on a real `RenderConfig` —
  and everything embedded goes through `src.embedder.Embedder`. There is no
  second copy of either forward. (Until 2026-08-18 there was: the harnesses
  called `classify_stls`' `make_renderer`/`render_up_candidate_grid`/
  `embed_images`, a single-process re-arrangement of the same maths that had
  measurably drifted from what shipped. Those functions have since been
  deleted, and nothing under `eval/` imports `classify_stls` at all —
  cache layout comes from `src.cachedir`, keys from `src.identity`, text
  embeddings from `src.embedder`.)
* **The maths is production's.** `src.pose` — `up_axis_scores`,
  `rank_up_scores`, `combine_up`, `needs_arbiter_margin`, `upright_scores`,
  `make_contact_sheet` — is imported, never reimplemented.
* **The scheduling is not.** Production runs the Poser/Renderer/Embedder as
  actors across a process boundary (`src/driver.py`). A harness composes the
  same calls in one process, in one thread, in a loop. So a harness result is
  a statement about the ensemble, the probes, the gate and the pixels — not
  about the pipeline's ordering, queueing or caching.

`eval/rig.py` holds the renderer for the process lifetime and is never
destroyed (CLAUDE.md); a script that renders must leave through
`rig.exit_without_teardown()`.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))    # so `import rig` works from anywhere

# Scratch space for renders, contact sheets and per-run prediction dumps.
# Derived and regenerable — gitignored. Override with EVAL_OUT to keep runs apart.
OUT = Path(os.environ.get("EVAL_OUT", REPO / "eval" / "out"))
OUT.mkdir(parents=True, exist_ok=True)

AX = ["+Z", "-Z", "+Y", "-Y", "+X", "-X"]   # order must match pose.UP_CANDIDATES
IDX = {a: i for i, a in enumerate(AX)}

LABELS_FILE = REPO / "up_axis_labels.json"


def load_labels(which=None):
    """Hand-labelled up axes as [{stem, path, set, up, gold}], gold being the
    index into AX / pose.UP_CANDIDATES.

    which: None for all, or "orig" / "holdout". Read LEARNINGS before quoting a
    number off "orig" — the probes and the min-max scheme were tuned against
    that set, so it scores optimistically. "holdout" was drawn from files the
    original never touched, with the method frozen.
    """
    raw = json.loads(LABELS_FILE.read_text())
    root = Path(raw["collection_root"])
    out = []
    for l in raw["labels"]:
        if which and l["set"] != which:
            continue
        out.append({"stem": l["stem"], "path": root / l["path"], "set": l["set"],
                    "up": l["up"], "gold": IDX[l["up"]]})
    return out


def collection_root():
    """The collection the labels point into — also the base every cache key is
    taken relative to (identity.py), so harnesses that read the pose or
    embedding caches have to agree with it."""
    return Path(json.loads(LABELS_FILE.read_text())["collection_root"])


def mark(pick, gold):
    """Axis name for a prediction, starred when it disagrees with the label."""
    if pick is None:
        return "--"
    return AX[pick] + ("" if pick == gold else "*")


def score(rows, key):
    """(correct, total) over rows that have a non-None prediction for key."""
    have = [r for r in rows if r.get(key) is not None]
    return sum(r[key] == r["gold"] for r in have), len(have)


RESULTS_FILE = REPO / "eval" / "results-2026-08-12.json"


def load_baselines():
    """The 2026-08-12 per-model predictions behind the LEARNINGS tables, keyed
    by stem: geometry, ensemble_2048, needs_arbiter, and each VLM at both sheet
    sizes. Axis *names*, not indices. Reused so a new model can be compared
    against the published numbers without re-running the SigLIP pass."""
    return {p["stem"]: p for p in json.loads(RESULTS_FILE.read_text())["predictions"]}


def load_arbiters(stems):
    """{arbiter: {stem: idx}} from every recorded VLM run that covers these
    models — the published 2026-08-12 predictions plus any `gauntlet_hard`
    dump on disk. Answers already paid for, so a harness can score a gate
    against real arbiters without an API key.

    Lived in `arbiter_gate.py` until that harness was retired (2026-08-18);
    `tile_count.py` is the remaining caller."""
    picks = {}
    published = {p["stem"]: p for p in json.loads(RESULTS_FILE.read_text())["predictions"]}
    for stem, rec in published.items():
        for k, v in rec.items():
            if "_sheet" in k and isinstance(v, str):
                picks.setdefault(k, {})[stem] = IDX[v]
    g = OUT / "gauntlet_hard.json"
    if g.exists():
        for k, per in json.loads(g.read_text())["vlm"].items():
            m, t = k.rsplit("@", 1)
            for stem, v in per.items():
                if v is not None:
                    picks.setdefault(f"{m}_sheet{t}", {})[stem] = IDX[v]
    return {k: v for k, v in picks.items() if any(s in v for s in stems)}


def contact_sheet(tiles, thumb, cols=3):
    """The production sheet. pose.make_contact_sheet owns the layout and the
    scaled numerals now — this stays only so harnesses keep one import."""
    from src import pose
    return pose.make_contact_sheet(tiles, thumb, cols)


def ask_claude(model, sheet, n_tiles=6):
    """Arbiter answer from a Claude model through the CLI. One retry, then None
    — the pipeline never hard-fails on the VLM, so neither does a harness."""
    import subprocess
    from src import pose
    prompt = f"Read the image at {sheet}. {pose.UP_PROMPT}"
    for _ in range(2):
        try:
            out = subprocess.run(["claude", "-p", prompt, "--model", model,
                                  "--output-format", "json", "--max-turns", "3"],
                                 capture_output=True, text=True, timeout=300)
            if out.returncode != 0:
                continue
            v = pose.parse_tile_answer(json.loads(out.stdout).get("result", ""), n_tiles)
            if v is not None:
                return v
        except Exception:
            pass
    return None


def ask_gemma(sheet, model="gemma4:26b", n_tiles=6):
    """Arbiter answer from the local ollama VLM. Same contract as ask_claude.

    Reaches for `pose._ask_ollama` — a **private** function, for a backend
    production retired: `src/arbiter.py` ships gemini and claude, and nothing
    in the pipeline calls the ollama path any more. It is kept because the
    gemma column in `gauntlet.py` is a recorded comparison and re-running it
    needs the call; treat it as a harness-only dependency on a private name,
    and if `pose._ask_ollama` ever goes, this goes with it rather than growing
    a copy of the request here.
    """
    from src import pose
    for _ in range(2):
        try:
            v = pose._ask_ollama(sheet.read_bytes(), n_tiles, model)
            if v is not None:
                return v
        except Exception as e:
            print(f"  ollama error: {e}")
    return None


def build_tiles(labels=None, render_px=2048):
    """The 6 up-candidate tiles per labelled model, plus geometry's score
    vector, cached in OUT/tiles<render_px>.

    Returns {stem: {"tiles": [Path]*6, "geo": [float]*6}}. Splitting this out
    of the embedding pass keeps the renderer and SigLIP off the GPU at the same
    time (they evict each other on an 8 GB card) and lets a backbone sweep
    re-embed identical pixels rather than re-rendering per backbone.
    """
    import numpy as np
    from PIL import Image
    import rig
    from src import pose as P
    labels = labels if labels is not None else load_labels()
    d = OUT / f"tiles{render_px}"
    d.mkdir(parents=True, exist_ok=True)
    geo_file = d / "geo.json"
    geo = json.loads(geo_file.read_text()) if geo_file.exists() else {}
    tiles = {l["stem"]: [d / f"{l['stem']}_up{i}.png" for i in range(6)] for l in labels}
    todo = [l for l in labels
            if l["stem"] not in geo or any(not p.exists() for p in tiles[l["stem"]])]
    if todo:
        print(f"rendering {len(todo)} models x 6 up-candidate tiles at {render_px}px -> {d}")
        r = rig.rig(render_px)
        for n, l in enumerate(todo, 1):
            lm = rig.load(l["path"])
            # arrays out of the renderer, PIL in on the way to disk (rig docstring)
            for p, im in zip(tiles[l["stem"]], rig.pose_sheet_tiles(r, lm)):
                Image.fromarray(im).save(p)
            geo[l["stem"]] = [float(x) for x in np.asarray(P.up_axis_scores(lm.mesh))]
            print(f"  [{n}/{len(todo)}] {l['stem']}", flush=True)
        geo_file.write_text(json.dumps(geo, indent=1))
    # geo comes back as an array, not the cached JSON list — pose.combine_up_scores
    # min-maxes it and a list has no .min().
    return {l["stem"]: {"tiles": tiles[l["stem"]], "geo": np.asarray(geo[l["stem"]])}
            for l in labels}


def build_sheets(thumbs, labels=None, render_px=2048):
    """Up-candidate contact sheets per labelled model, in OUT/sheets<thumb>.

    thumbs is one size or a list; returns {thumb: {stem: path}}. The 6 tiles
    are rendered once and shared across sizes — rendering is the expensive
    part, the sheet is a resize. Existing sheets are reused, so every harness
    can call this cheaply. Only the *sheet* size matters to a VLM; render_px
    (the tile render resolution) was swept over 384..2048 and made no
    difference — see LEARNINGS, those two knobs are easy to conflate.
    """
    from PIL import Image
    import rig
    thumbs = [thumbs] if isinstance(thumbs, int) else list(thumbs)
    labels = labels if labels is not None else load_labels()
    paths = {}
    for t in thumbs:
        (OUT / f"sheets{t}").mkdir(parents=True, exist_ok=True)
        paths[t] = {l["stem"]: OUT / f"sheets{t}" / f"{l['stem']}.png" for l in labels}
    todo = [l for l in labels if any(not paths[t][l["stem"]].exists() for t in thumbs)]
    if todo:
        print(f"rendering {len(todo)} models -> sheets at {thumbs}")
        r = rig.rig(render_px)
        for n, l in enumerate(todo, 1):
            # pose.make_contact_sheet resizes and pastes, so it wants PIL —
            # the renderer hands back arrays (rig docstring)
            tiles = [Image.fromarray(im)
                     for im in rig.pose_sheet_tiles(r, rig.load(l["path"]))]
            for t in thumbs:
                contact_sheet(tiles, t).save(paths[t][l["stem"]])
            print(f"  [{n}/{len(todo)}] {l['stem']}", flush=True)
    return paths


# --- the 24-tile orbit set --------------------------------------------------
#
# Lived in `front_first.py` until that harness was retired (a recorded negative
# result; git history keeps it). Two surviving scorers read these pixels —
# `tile_count.py --source orbit` and `geo_floor.py` (`arbiter_gate.py` was the
# third until it was retired the same day) — and every
# published azimuth number in LEARNINGS was measured on them, so the cache and
# the code that fills it outlive the harness they were written for.
#
# These are NOT production's pose tiles: they rotate the mesh per candidate
# where `Renderer.pose_tiles` carries the rotation back through the cameras.
# Measured, the two agree to the noise floor only for +Z, where R is the
# identity — `tile_count.py --compare` reprints the difference. Reach for
# `rig.pose_tiles` when you want the pipeline's pixels.

ORBIT_N_AZ = 4       # the four fronts perpendicular to each up
ORBIT_PX = 384       # measured as good as 2048 here, and ~3x faster to embed
ORBIT_ELEV = 20.0


def build_orbit_tiles(labels, render_px=ORBIT_PX):
    """24 tiles per model — 6 up candidates x 4 azimuths — cached on disk.

    `Renderer.views` is the production call that does this: it rotates a *copy*
    of the mesh into the candidate's frame and orbits it, which is exactly the
    rotate-the-mesh path these tiles have always used. The camera config
    (`views=4`, one 20 deg elevation) is what makes the ring four azimuths.
    """
    from PIL import Image
    import rig
    from src import pose
    d = OUT / f"orbit{render_px}x{ORBIT_N_AZ}"
    d.mkdir(parents=True, exist_ok=True)
    paths = {l["stem"]: [[d / f"{l['stem']}_u{u}a{k}.png" for k in range(ORBIT_N_AZ)]
                         for u in range(6)] for l in labels}
    todo = [l for l in labels
            if any(not p.exists() for row in paths[l["stem"]] for p in row)]
    if todo:
        print(f"rendering {len(todo)} models x {6 * ORBIT_N_AZ} orientations "
              f"at {render_px}px")
        r = rig.rig(render_px, views=ORBIT_N_AZ, elevations=(ORBIT_ELEV,))
        for n, l in enumerate(todo, 1):
            lm, i = rig.load(l["path"]), rig.index()   # one residency slot, six visits
            for u, up in enumerate(pose.UP_CANDIDATES):
                for k, im in enumerate(rig.views(r, lm, up, index=i)):
                    Image.fromarray(im).save(paths[l["stem"]][u][k])
            print(f"  [{n}/{len(todo)}] {l['stem']}", flush=True)
    return paths
