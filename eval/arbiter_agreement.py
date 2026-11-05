"""Gemini against Jev-fused-into-the-ensemble on unlabelled models, as agreement.

    .venv/bin/python eval/arbiter_agreement.py --n 1000           # sample, render, call both
    .venv/bin/python eval/arbiter_agreement.py --n 1000 --report-only

Every hand label is spent on the 201 poolable models, so a larger comparison
has no ground truth. This draws `--n` unlabelled models from embed-cache512's
pose cache, runs the production gate over them (`pose_baseline.py`'s tiers),
and asks only the gated ones of both arbiters: gemini on the production sheet,
and Jev `jev-tiles-clean` fused as `ensemble + w * log(p + 0.01)` with `w`
frozen at `jev_fusion.py`'s 0.35. Where the two disagree, a hand label decides
who was right — the models that separate them are the only ones worth labelling.
"""
import argparse, hashlib, itertools, json, random, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from common import AX, OUT, build_sheets, build_tiles, collection_root, load_labels, production_model

import decisions_vlm, gemini_vlm
from src import pose

W = 0.35


def sample(n, seed):
    root = collection_root()
    cache = json.loads((Path(__file__).resolve().parent.parent / "embed-cache512" / "pose-cache.json").read_text())
    labelled = {str(l["path"].relative_to(root)) for l in load_labels()}
    rels = sorted({k.split("|")[0] for k in cache} - labelled)
    random.Random(seed).shuffle(rels)
    out = []
    for rel in rels:
        if (root / rel).exists():
            # Stems collide across the collection; the hash keeps tile files apart.
            out.append({"stem": f"{Path(rel).stem[:40]}_{hashlib.sha1(rel.encode()).hexdigest()[:8]}",
                        "rel": rel, "path": root / rel})
        if len(out) == n:
            return out
    return out


def gate(items):
    import rig
    tiles = build_tiles(items, 512)
    emb = rig.embedder(production_model())
    for it in items:
        geo = tiles[it["stem"]]["geo"]
        imgs = [Image.open(t).convert("RGB") for t in tiles[it["stem"]]["tiles"]]
        sig = pose.upright_scores(rig.embed(emb, imgs), emb.up_T, emb.down_T)
        idx, margin = pose.combine_up(geo, sig)
        it.update(ens=int(idx), margin=float(margin), gated=bool(pose.needs_arbiter_margin(margin)),
                  combined=(pose.geo_weight(geo) * pose._unit(geo) + pose._unit(sig)).tolist())
    del emb


def call_both(gated, workers):
    sheets = build_sheets([512], gated)[512]
    tok, usage = gemini_vlm.token(), {"gemini": [], "jev": []}
    t0, done, lock = time.time(), itertools.count(1), threading.Lock()

    def one(it):
        g = gemini_vlm.ask(pose.GEMINI_MODEL, sheets[it["stem"]], tok, usage=usage["gemini"])
        j, conf, probs = decisions_vlm.ask_jev_tiles(sheets[it["stem"]], usage["jev"], blank=True)
        with lock:
            print(f"  [{next(done):>4}/{len(gated)}] {it['stem'][:48]:48} gemini "
                  f"{'--' if g is None else AX[g]:>3} jev {'--' if j is None else AX[j]:>3} "
                  f"{time.time()-t0:5.0f}s", flush=True)
        return g, j, conf, probs

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for it, (g, j, conf, probs) in zip(gated, ex.map(one, gated)):
            it.update(gemini=g, jev=j, jev_conf=conf, jev_probs=probs)
            if probs is not None:
                it["fused"] = int(np.argmax(np.array(it["combined"]) + W * np.log(np.array(probs) + 0.01)))
    return usage


def report(d):
    items = d["items"]
    gated = [it for it in items if it["gated"]]
    both = [it for it in gated if it.get("gemini") is not None and it.get("fused") is not None]
    print(f"{len(items)} sampled | gate fires on {len(gated)} ({100*len(gated)/len(items):.0f}%) | "
          f"both answered {len(both)}")
    pair = lambda a, b: sum(it[a] == it[b] for it in both)
    for a, b in [("gemini", "fused"), ("gemini", "jev"), ("gemini", "ens"), ("fused", "ens"), ("jev", "ens")]:
        print(f"  {a:6} = {b:6} {pair(a, b):4}/{len(both)} ({100*pair(a, b)/len(both):.0f}%)")
    dis = [it for it in both if it["gemini"] != it["fused"]]
    print(f"\n{len(dis)} disagreements (gemini vs fused) — label these:")
    for it in dis:
        print(f"  {it['rel']}\n      gemini {AX[it['gemini']]}  fused {AX[it['fused']]}  "
              f"ens {AX[it['ens']]}  jev {AX[it['jev']]} conf {it['jev_conf']}")
    for name, u in d["usage"].items():
        secs = sorted(x["secs"] for x in u)
        if secs:
            print(f"\n{name}: {len(u)} calls, median {secs[len(secs)//2]:.1f}s, "
                  f"p90 {secs[int(.9*len(secs))]:.1f}s, max {secs[-1]:.1f}s, "
                  f"mean prompt {np.mean([x['prompt'] for x in u]):.0f} tok, "
                  f"mean thoughts+output {np.mean([x['thoughts'] + x['output'] for x in u]):.0f} tok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    path = OUT / f"arbiter_agreement_{args.n}_s{args.seed}.json"
    if not args.report_only:
        items = sample(args.n, args.seed)
        print(f"sampled {len(items)} unlabelled models (seed {args.seed})")
        gate(items)
        gated = [it for it in items if it["gated"]]
        print(f"gate fires on {len(gated)}; calling gemini {pose.GEMINI_MODEL} and Jev on each")
        usage = call_both(gated, args.workers)
        for it in items:
            it["path"] = str(it["path"])
        path.write_text(json.dumps({"w": W, "items": items, "usage": usage}, indent=1))
        print(f"wrote {path}\n")
    report(json.loads(path.read_text()))
    if not args.report_only:
        import rig
        rig.exit_without_teardown()


if __name__ == "__main__":
    main()
