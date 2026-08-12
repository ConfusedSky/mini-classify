"""Shared setup for the pose-evaluation harnesses.

These scripts measure the up-axis pipeline against hand-labelled ground truth.
They import the app modules directly, so they always test the real code path
rather than a copy of it.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

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


def sheet_font(thumb):
    """Tile numerals scaled to the tile size.

    PIL's default face is a ~11 px bitmap — legible on a 768x512 sheet and
    proportionally invisible on a 1536x1024 one, and a model cannot answer
    {"tile": n} about numerals it cannot read. A naive thumb=512 therefore
    measures *worse* than 256; that is the trap OPEN_QUESTIONS records.
    """
    from PIL import ImageFont
    size = max(11, thumb * 44 // 512)
    try:
        return ImageFont.load_default(size=size)     # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()              # bitmap, fixed ~11 px


def contact_sheet(tiles, thumb, cols=3):
    """pose.make_contact_sheet with the numerals scaled — see sheet_font."""
    from PIL import Image, ImageDraw
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    draw = ImageDraw.Draw(sheet)
    font = sheet_font(thumb)
    for i, im in enumerate(tiles):
        im = im.copy()
        im.thumbnail((thumb, thumb))
        x, y = (i % cols) * thumb, (i // cols) * thumb
        sheet.paste(im, (x, y))
        draw.text((x + thumb // 36, y + thumb // 64), str(i + 1), fill="red", font=font)
    return sheet


def build_sheets(thumbs, labels=None, render_px=2048):
    """Up-candidate contact sheets per labelled model, in OUT/sheets<thumb>.

    thumbs is one size or a list; returns {thumb: {stem: path}}. The 6 tiles
    are rendered once and shared across sizes — rendering is the expensive
    part, the sheet is a resize. Existing sheets are reused, so every harness
    can call this cheaply. Only the *sheet* size matters to a VLM; render_px
    (the tile render resolution) was swept over 384..2048 and made no
    difference — see LEARNINGS, those two knobs are easy to conflate.
    """
    import classify_stls as C
    thumbs = [thumbs] if isinstance(thumbs, int) else list(thumbs)
    labels = labels if labels is not None else load_labels()
    paths = {}
    for t in thumbs:
        (OUT / f"sheets{t}").mkdir(parents=True, exist_ok=True)
        paths[t] = {l["stem"]: OUT / f"sheets{t}" / f"{l['stem']}.png" for l in labels}
    todo = [l for l in labels if any(not paths[t][l["stem"]].exists() for t in thumbs)]
    if todo:
        print(f"rendering {len(todo)} models -> sheets at {thumbs}")
        renderer = C.make_renderer(render_px)
        for n, l in enumerate(todo, 1):
            tiles = C.render_up_candidate_tiles(renderer, C.load_mesh(l["path"]))
            for t in thumbs:
                contact_sheet(tiles, t).save(paths[t][l["stem"]])
            print(f"  [{n}/{len(todo)}] {l['stem']}", flush=True)
    return paths
