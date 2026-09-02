"""Does dropping an elevation ring cost search quality?

embed-cache512 is 8 azimuths x 2 elevations (+20/-20, elevation-major, so
views 0-7 are the +20 ring and 8-15 the -20 ring). Rank the pilot's 12
queries from the live cache's rows of the 301-model sample under view
subsets: production's 16, each ring alone, and alternate azimuths of both
rings (the other way to halve the view count). One GLM judge throughout —
`out/judgments_glm.json` + the ev2 delta cover the known pairs; only pairs a
subset ranking newly surfaces get judged, into `out/judgments_glm_elev.json`.

  .venv/bin/python eval/renderer_pilot/elev_check.py
"""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import ev2_check  # noqa: E402  (new_matrix, rank_arm, ndcg, QUERIES, OUT)

OUT = ev2_check.OUT
QUERIES = ev2_check.QUERIES


def main():
    from src.embedder import Embedder
    import judge, judge_glm

    M, missing = ev2_check.new_matrix()
    print(f"sample {M.shape[0]}, missing from cache: {missing or 'none'}")
    arms = {
        "16v": M,                 # production: both rings
        "8v +20": M[:, :8],       # top ring alone
        "8v -20": M[:, 8:],       # bottom ring alone
        "8v az/2": M[:, ::2],     # both rings, every other azimuth
    }
    emb = Embedder(categories=["placeholder"],
                   model_name="google/siglip2-so400m-patch16-512")
    ranked = {name: ev2_check.rank_arm(Ms, emb) for name, Ms in arms.items()}
    json.dump(ranked, open(OUT / "elev_rankings.json", "w"))

    J = json.load(open(OUT / "judgments_glm.json"))
    for extra in ("judgments_glm_ev2.json",):
        p = OUT / extra
        if p.exists():
            J.update(json.load(open(p)))
    delta_p = OUT / "judgments_glm_elev.json"
    delta = json.load(open(delta_p)) if delta_p.exists() else {}
    J.update(delta)
    todo = sorted({(q, n) for res in ranked.values() for q in QUERIES
                   for n in res[q] if f"{q}|{n}" not in J})
    print(f"{len(todo)} new (query, model) pairs to judge with GLM")
    for q, n in todo:
        r = judge_glm.ask(q, judge.BY_N[n])
        delta[f"{q}|{n}"] = r; J[f"{q}|{n}"] = r
        json.dump(delta, open(delta_p, "w"), indent=0)
    errs = [k for k, v in delta.items() if "error" in v]
    if errs:
        print("judge errors:", errs)
    if delta:
        print(f"delta judge cost ${sum(v.get('usd', 0) for v in delta.values()):.3f}")

    names = list(arms)
    print(f"\n{'query':34}" + "".join(f" {n:>9}" for n in names))
    means = {n: [] for n in names}
    for q in QUERIES:
        row = ""
        for n in names:
            v = ev2_check.ndcg(q, ranked[n][q], J)
            means[n].append(v); row += f" {v:9.3f}"
        print(f"{q[:34]:34}{row}")
    print(f"{'mean nDCG@20 (GLM judge)':34}"
          + "".join(f" {np.mean(means[n]):9.3f}" for n in names))
    base = means[names[0]]
    for n in names[1:]:
        wins = sum(a > b + 1e-9 for a, b in zip(means[n], base))
        losses = sum(a < b - 1e-9 for a, b in zip(means[n], base))
        ov = np.mean([len(set(ranked[names[0]][q][:10]) & set(ranked[n][q][:10])) / 10
                      for q in QUERIES])
        print(f"{n}: {np.mean(means[n]) - np.mean(base):+.3f} vs 16v, "
              f"better/worse {wins}/{losses}, mean top-10 overlap {ov:.2f}")


if __name__ == "__main__":
    main()
