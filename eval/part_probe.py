"""Can SigLIP see part-ness? Zero-shot complete-model vs part/accessory over
the cached embeddings — no rendering, one text embed and a matmul.

The question behind it (2026-09-03): ~20% of the corpus is split-kit pieces
(hands, weapons, quivers), and a name filter is too blunt for a class that
big. If the cached embeddings separate parts from complete models, the verdict
ships as a query-time flag; if the boundary is too wide — complete-but-handless
bodies (DM Stash `_Body`) or terrain landing on the part side — the fallback
is the deterministic walk-level cuts only.

Terrain is deliberately in the COMPLETE bank: a throne or a ruin is a model in
its own right, and a probe that reads "not a creature" as "a part" fails the
brief. DM Stash bodies (a miniature whose hands ship separately) must land
complete for the same reason.

Cohorts scored, all structural (no hand labels yet — the pseudo-labels are
directories that say what they hold):
  parts:    files under a "Standalone Weapons ..." directory (accessory by
            construction), plus the _L/_R handed-suffix stems
  terrain:  Loot Studios "- Environment" paths
  body:     DM Stash *_Body* files (complete, hands sold separately)
  rest:     everything else

  .venv/bin/python eval/part_probe.py --cache-dir embed-cache512

Writes eval/out/part_probe.json (per-model margins) and prints the cohort
table plus the most part-like models by name for eyeballing.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COMPLETE = [
    "a complete tabletop miniature of a character or creature",
    "a full-body miniature figure on a base",
    "a tabletop terrain piece: a building, throne, ruin or scenery model",
    "a miniature bust sculpture of a character",
]
PARTS = [
    "a disembodied hand holding a weapon",
    "a single sword, axe or staff: a weapon accessory piece",
    "a detached arm, leg or head piece of a model kit",
    "a small accessory piece: a quiver, backpack, shield or pouch",
]


def cohort_of(rel):
    low = rel.lower()
    if "standalone weapons" in low:
        return "parts:standalone"
    if re.search(r"_[lr]\d?( \(repaired\))?(\s*\(side[^)]*\))?( \(repaired\))?\.stl$", low):
        return "parts:handed"
    if "- environment/" in low or "/environment/" in low:
        return "terrain"
    if low.startswith("dm stash/") and "body" in low:
        return "body"
    return "rest"


def main():
    from src.cachedir import add_cache_args, apply_run_params
    from src.collection import Collection
    from src.embedder import Embedder
    from src import query as Q

    parser = argparse.ArgumentParser()
    add_cache_args(parser, "STL directory (defaults to the last classify run)")
    args = apply_run_params(parser)
    c = Collection.load(args)

    emb = Embedder(categories=["placeholder"], model_name=args.model)
    texts = COMPLETE + PARTS
    feats = emb._embed_raw(texts).float().cpu().numpy()
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    sims = Q.score(c.matrix, feats.T, "softmax")          # (n_models, n_texts)
    best_complete = sims[:, : len(COMPLETE)].max(axis=1)
    best_part = sims[:, len(COMPLETE):].max(axis=1)
    margin = best_part - best_complete                    # >0 reads as a part

    rels = ["/".join(r) for r in c._rel]
    cohorts = [cohort_of(r) for r in rels]
    out = [{"rel": r, "cohort": co, "margin": round(float(m), 5),
            "complete": round(float(bc), 5), "part": round(float(bp), 5)}
           for r, co, m, bc, bp in zip(rels, cohorts, margin, best_complete, best_part)]
    Path("eval/out").mkdir(exist_ok=True)
    json.dump(out, open("eval/out/part_probe.json", "w"), indent=0)

    print(f"{len(out)} models; margin = best part sim - best complete sim\n")
    print(f"{'cohort':18} {'n':>5} {'median':>8} {'flagged @0':>11} {'@ -0.005':>9} {'@ +0.005':>9}")
    for co in ["parts:standalone", "parts:handed", "terrain", "body", "rest"]:
        ms = np.array([o["margin"] for o in out if o["cohort"] == co])
        if not len(ms):
            print(f"{co:18} {0:>5}")
            continue
        print(f"{co:18} {len(ms):>5} {np.median(ms):>8.4f} "
              f"{(ms > 0).mean():>10.1%} {(ms > -0.005).mean():>8.1%} "
              f"{(ms > 0.005).mean():>8.1%}")

    for co, want, k in (("terrain", "most part-like (should be complete)", 8),
                        ("body", "most part-like (should be complete)", 8),
                        ("rest", "most part-like (candidates)", 25),
                        ("parts:standalone", "most COMPLETE-like (probe misses)", 8)):
        rows = sorted((o for o in out if o["cohort"] == co),
                      key=lambda o: -o["margin"])
        if co == "parts:standalone":
            rows = rows[::-1]
        print(f"\n--- {co}: {want}")
        for o in rows[:k]:
            print(f"  {o['margin']:+.4f}  {o['rel']}")


if __name__ == "__main__":
    main()
