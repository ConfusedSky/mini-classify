"""Score every (render arm x prompt arm x query) with the production scorer and
pool the top-K candidates per query for judging. With judgments.json present,
report P@10 / nDCG@20 per arm.

  python score.py            # -> candidates.json (what to judge)
  python score.py --report   # -> metrics from judgments.json
"""
import json, sys
from pathlib import Path

import numpy as np

REPO = Path.home() / "Documents/tests/mini-classify"
sys.path.insert(0, str(REPO))

HERE = Path(__file__).parent
S = json.load(open(HERE / "sample.json"))
QUERIES = list(S["queries"])
K, POOL = 20, "softmax"
ARMS = ["open3d", "open3d_tight", "f3d", "three_ao", "three_noao", "three_white"]
GREY = ["a 3D render of a {} miniature", "a photo of a {} figurine", "a tabletop miniature of a {}"]
PROMPTS = {
    "production": {a: GREY for a in ARMS},
    "described": {
        "open3d": ["an untextured grey 3D render of a {} miniature", "a grey resin 3D print of a {}",
                   "a tabletop miniature of a {}, unpainted grey render"],
        "f3d": ["an untextured grey 3D render of a {} miniature", "a grey resin 3D print of a {}",
                "a tabletop miniature of a {}, unpainted grey render"],
        "three": ["a 3D render of a {} miniature with red and blue rim lighting on a dark background",
                  "a grey {} figurine lit by red and blue rim lights, dark background",
                  "a tabletop miniature of a {}, studio render with coloured rim lighting"],
    },
}


def prompts_for(kind, arm):
    table = PROMPTS[kind]
    if arm in table:
        return table[arm]
    return table["three" if arm.startswith("three") else "open3d"]


def text_bank(emb):
    """{(kind, arm): (dim, n_queries)} templated query embeddings, averaged per query."""
    out = {}
    for kind in PROMPTS:
        for arm in ARMS:
            tmpl = prompts_for(kind, arm)
            key = (kind, tuple(tmpl))
            if key not in out:
                rows = []
                for q in QUERIES:
                    feats = emb._embed_raw([t.format(q) for t in tmpl]).float().cpu().numpy()
                    v = feats.mean(0); rows.append(v / np.linalg.norm(v))
                out[key] = np.stack(rows).T
    return {(kind, arm): out[(kind, tuple(prompts_for(kind, arm)))] for kind in PROMPTS for arm in ARMS}


def rankings():
    from src.embedder import Embedder
    from src import query as Q
    emb = Embedder(categories=["placeholder"], model_name="google/siglip2-so400m-patch16-512")
    banks = text_bank(emb)
    res = {}
    for arm in ARMS:
        M = np.load(HERE / f"arm_{arm}.npy")
        valid = np.abs(M).sum(axis=(1, 2)) > 0
        for kind in PROMPTS:
            sims = Q.score(M, banks[(kind, arm)], POOL)          # (n_models, n_queries)
            sims[~valid] = -1
            for qi, q in enumerate(QUERIES):
                r = Q.rank(sims[:, qi], top=K)
                res[f"{arm}|{kind}|{q}"] = [int(i) for i in r.order]
    return res


def report(res, judg):
    def rel(q, n):
        return judg.get(f"{q}|{n}", {}).get("relevance")
    def p10(q, order):
        r = [rel(q, n) for n in order[:10]]
        return sum(x == 2 for x in r) / 10
    def ndcg(q, order, k=20):
        gains = [(rel(q, n) or 0) for n in order[:k]]
        dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
        ideal = sorted([(judg[j]["relevance"]) for j in judg if j.startswith(q + "|")], reverse=True)[:k]
        idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal)) or 1
        return dcg / idcg
    print(f"{'arm':14} {'prompt':11} {'P@10':>6} {'nDCG@20':>8}   per-query P@10")
    rows = []
    for arm in ARMS:
        for kind in PROMPTS:
            ps = [p10(q, res[f"{arm}|{kind}|{q}"]) for q in QUERIES]
            ns = [ndcg(q, res[f"{arm}|{kind}|{q}"]) for q in QUERIES]
            rows.append((arm, kind, np.mean(ps), np.mean(ns), ps))
    for arm, kind, p, n, ps in rows:
        print(f"{arm:14} {kind:11} {p:6.3f} {n:8.3f}   " + " ".join(f"{x:.1f}" for x in ps))
    print("\nqueries:", " | ".join(f"{i}:{q}" for i, q in enumerate(QUERIES)))
    unj = sum(1 for k in res for n in res[k][:K] if rel(k.split('|')[2], n) is None)
    if unj:
        print(f"\nWARNING {unj} ranked slots unjudged")


if __name__ == "__main__":
    if "--report" in sys.argv:
        res = json.load(open(HERE / "rankings.json"))
        judg = json.load(open(HERE / "judgments.json"))
        report(res, judg)
    else:
        res = rankings()
        json.dump(res, open(HERE / "rankings.json", "w"))
        pool = {}
        for k, order in res.items():
            q = k.split("|")[2]
            pool.setdefault(q, set()).update(order[:K])
        cands = {q: sorted(v) for q, v in pool.items()}
        json.dump(cands, open(HERE / "candidates.json", "w"))
        print({q: len(v) for q, v in cands.items()}, "total", sum(map(len, cands.values())))
