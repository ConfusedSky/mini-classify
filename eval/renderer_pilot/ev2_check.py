"""Before/after for the ev2 (tight-framing) rebuild on the pilot's 12 queries.

Before = the pilot's `open3d` control arm (arm_open3d.npy, the pre-ev2
production embeddings of the 301-model sample). After = the same models'
rows read from the rebuilt embed-cache512. One judge for both sides:
GLM-5.3-Flash (out/judgments_glm.json covers every pre-ev2 pair; only pairs
the new ranking surfaces get delta-judged, into out/judgments_glm_ev2.json).

  .venv/bin/python eval/renderer_pilot/ev2_check.py          # rank + judge + report
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

HERE = Path(__file__).parent
OUT = HERE / "out"
S = json.load(open(HERE / "sample.json"))
QUERIES = list(S["queries"])
K, POOL = 20, "softmax"
TEMPLATES = ["a 3D render of a {} miniature", "a photo of a {} figurine",
             "a tabletop miniature of a {}"]


def new_matrix():
    """The rebuilt cache's rows for the pilot sample, aligned to sample order.

    Missing rows (the six Orconspiracy models whose paths moved in the
    library reorganisation, if sampled) stay zero and are pushed to the
    bottom at scoring time, mirroring how the pilot handled f3d's two
    unloadable models."""
    from src.cachedir import add_cache_args, apply_run_params, cache_root, load_file_list
    from src.embed_store import load_embedding_matrix
    saved, sys.argv = sys.argv, [sys.argv[0], "--cache-dir", str(REPO / "embed-cache512")]
    parser = argparse.ArgumentParser()
    add_cache_args(parser, "")
    args = apply_run_params(parser)
    sys.argv = saved
    root = cache_root(Path(args.input), args.cache_dir, confirm=False)
    files = load_file_list(Path(args.input), args.cache_dir, False)
    matrix, kept, _ = load_embedding_matrix(files, args, root)
    by_path = {str(f): i for i, f in enumerate(kept)}
    dim = matrix.shape[2]
    rows = np.zeros((len(S["sample"]), matrix.shape[1], dim), np.float32)
    missing = []
    for m in S["sample"]:
        i = by_path.get(m["path"])
        if i is None:
            missing.append(m["key"])
        else:
            rows[m["n"]] = matrix[i]
    return rows, missing


def rank_arm(M, emb):
    from src import query as Q
    valid = np.abs(M).sum(axis=(1, 2)) > 0
    res = {}
    for q in QUERIES:
        feats = emb._embed_raw([t.format(q) for t in TEMPLATES]).float().cpu().numpy()
        v = feats.mean(0); v /= np.linalg.norm(v)
        sims = Q.score(M, v[:, None], POOL)[:, 0]
        sims[~valid] = -1
        res[q] = [int(i) for i in Q.rank(sims, top=K).order]
    return res


def ndcg(q, order, J, k=20):
    g = [J.get(f"{q}|{n}", {}).get("relevance", 0) for n in order[:k]]
    dcg = sum(x / np.log2(i + 2) for i, x in enumerate(g))
    ideal = sorted([v.get("relevance", 0) for kk, v in J.items()
                    if kk.startswith(q + "|")], reverse=True)[:k]
    return dcg / (sum(x / np.log2(i + 2) for i, x in enumerate(ideal)) or 1)


def main():
    from src.embedder import Embedder
    import judge, judge_glm

    M_new, missing = new_matrix()
    M_old = np.load(HERE / "arm_open3d.npy")
    print(f"sample {len(S['sample'])}, missing from rebuilt cache: {missing or 'none'}")
    emb = Embedder(categories=["placeholder"], model_name="google/siglip2-so400m-patch16-512")
    after = rank_arm(M_new, emb)
    before = rank_arm(M_old, emb)
    json.dump({"before": before, "after": after}, open(OUT / "ev2_rankings.json", "w"))

    J = json.load(open(OUT / "judgments_glm.json"))
    delta_p = OUT / "judgments_glm_ev2.json"
    delta = json.load(open(delta_p)) if delta_p.exists() else {}
    J.update(delta)
    todo = sorted({(q, n) for src in (before, after) for q in QUERIES
                   for n in src[q] if f"{q}|{n}" not in J})
    print(f"{len(todo)} new (query, model) pairs to judge with GLM")
    for q, n in todo:
        r = judge_glm.ask(q, judge.BY_N[n])
        delta[f"{q}|{n}"] = r; J[f"{q}|{n}"] = r
        json.dump(delta, open(delta_p, "w"), indent=0)
    errs = [k for k, v in delta.items() if "error" in v]
    if errs:
        print("judge errors:", errs)

    print(f"\n{'query':34} {'before':>7} {'after':>7} {'d':>6}")
    b_all, a_all = [], []
    for q in QUERIES:
        b, a = ndcg(q, before[q], J), ndcg(q, after[q], J)
        b_all.append(b); a_all.append(a)
        print(f"{q[:34]:34} {b:7.3f} {a:7.3f} {a-b:+6.3f}")
    print(f"{'mean nDCG@20 (GLM judge)':34} {np.mean(b_all):7.3f} {np.mean(a_all):7.3f} "
          f"{np.mean(a_all)-np.mean(b_all):+6.3f}")
    wins = sum(a > b + 1e-9 for a, b in zip(a_all, b_all))
    losses = sum(a < b - 1e-9 for a, b in zip(a_all, b_all))
    ov = np.mean([len(set(before[q][:10]) & set(after[q][:10])) / 10 for q in QUERIES])
    print(f"queries better/worse: {wins}/{losses}; mean top-10 overlap {ov:.2f}")


if __name__ == "__main__":
    main()
