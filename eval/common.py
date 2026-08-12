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
