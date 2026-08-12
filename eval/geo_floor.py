"""Should geometry's vote shrink when it found no print base at all?

  python geo_floor.py

`combine_up` min-maxes each score vector, which maps geometry's *ratio* to its
vote margin — the documented feature, and a good one. What it cannot see is
*magnitude*: a mesh with no flat base anywhere still produces an unequal score
vector, so geometry votes with a confident-looking margin off evidence that is
two orders of magnitude below `ABS_SCORE_FLOOR`. `32mm_Orguss_Head` scores
0.0075 with ratio 0.43, votes with a ~0.57 margin, and overrides a four-view
SigLIP answer that was right.

Tested here: keep min-max, but scale the whole geometry vote by a confidence
weight w = min(1, best/floor) ** p.

    w * _unit(geo) + _unit(siglip)

This is *not* the "absolute-scaled geometry" scheme in `ensemble.py` that
scored 20/23. That one replaced min-max with `clip(geo/floor, 0, 1)`, which
saturates every candidate above the floor at 1.0 and destroys the margin. Here
the margin inside geometry's vote is untouched; only how loudly it votes changes.

Because the gate reads the same combined vector, an attenuated geometry vote
also shrinks every margin — so the escalation rate is reported beside the
accuracy. A scheme that buys a model by escalating twice as often has not
bought anything.
"""
import argparse, json

import numpy as np

from common import AX, OUT, build_tiles, load_labels  # puts REPO on sys.path

import pose

BACKBONE = "google/siglip2-so400m-patch14-384"


def state(labels):
    """{stem: (geo(6), sig4(6), best)} — geometry and the four-view upright score."""
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    import classify_stls as C
    from front_first import N_AZ, RENDER_PX as ORBIT_PX, build_orbit_tiles

    orbit = build_orbit_tiles(labels, ORBIT_PX)
    geo_src = build_tiles(labels, 2048)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(BACKBONE, torch_dtype=torch.float16).to(dev).eval()
    proc = AutoProcessor.from_pretrained(BACKBONE)
    up = C.embed_raw(model, proc, pose.UPRIGHT_PROMPTS, dev).float().cpu().numpy()
    dn = C.embed_raw(model, proc, pose.TOPPLED_PROMPTS, dev).float().cpu().numpy()

    out = {}
    for l in labels:
        flat = [p for row in orbit[l["stem"]] for p in row]
        emb = C.embed_images(model, proc, [Image.open(p).convert("RGB") for p in flat],
                             dev).float().cpu().numpy()
        sig = pose.upright_scores(emb, up, dn).reshape(6, N_AZ).mean(1)
        geo = np.asarray(geo_src[l["stem"]]["geo"])
        out[l["stem"]] = (geo, sig, float(pose.rank_up_scores(geo)[2]))
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


def combine(geo, sig, best, floor, p):
    """(index, margin, w). p=0 reproduces production exactly (w=1 always)."""
    w = 1.0 if p == 0 else min(1.0, best / floor) ** p
    c = w * pose._unit(geo) + pose._unit(sig)
    top = np.sort(c)[::-1]
    # rescale so the margin stays on production's 0..2 axis whatever w is —
    # otherwise a quieter geometry vote silently raises the escalation rate
    scale = 2.0 / (1.0 + w) if w > 0 else 1.0
    return int(np.argmax(c)), float((top[0] - top[1]) * scale), w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floors", default="0.02")
    ap.add_argument("--powers", default="0,0.5,1,2,99",
                    help="attenuation exponent; 0 = production, 99 ~ hard switch")
    args = ap.parse_args()

    labels = load_labels()
    sc = state(labels)
    gold = {l["stem"]: l["gold"] for l in labels}
    sets = {}
    for l in labels:
        sets.setdefault(l["set"], []).append(l["stem"])
    sets["orig+hold"] = sets.get("orig", []) + sets.get("holdout", [])
    order = [n for n in ("orig", "holdout", "orig+hold", "hard") if sets.get(n)]

    hdr = (f"{'scheme':34} " + " ".join(f"{n:>11}" for n in order)
           + f" {'escalates':>11}")
    print(f"\nfour-view ensemble, geometry attenuated by w = min(1, best/floor)**p")
    print(hdr + "\n" + "-" * len(hdr))
    results = {}
    for floor in [float(f) for f in args.floors.split(",")]:
        for p in [float(x) for x in args.powers.split(",")]:
            picks = {s: combine(*sc[s], floor, p) for s in gold}
            cells = []
            for grp in order:
                ok = sum(picks[s][0] == gold[s] for s in sets[grp])
                cells.append(f"{ok}/{len(sets[grp])}")
            esc = sum(pose.needs_arbiter_margin(picks[s][1]) for s in sets["orig+hold"])
            name = ("production (p=0)" if p == 0 else
                    f"hard switch (p={p:g})" if p >= 99 else
                    f"floor {floor:g}, p={p:g}")
            print(f"{name:34} " + " ".join(f"{c:>11}" for c in cells)
                  + f" {esc:>7}/{len(sets['orig+hold'])}")
            results[name] = {s: (AX[picks[s][0]], round(picks[s][1], 3),
                                 round(picks[s][2], 3)) for s in gold}

    base = results["production (p=0)"]
    print("\nmodels whose answer changes, against production")
    for name, r in results.items():
        if name == "production (p=0)":
            continue
        moved = [s for s in gold if r[s][0] != base[s][0]]
        if not moved:
            print(f"  {name:32} no change")
            continue
        gain = sum(r[s][0] == AX[gold[s]] for s in moved)
        lose = sum(base[s][0] == AX[gold[s]] for s in moved)
        print(f"  {name:32} {len(moved)} moved: fixed {gain}, broke {lose}")
        for s in moved:
            tag = ("fixed" if r[s][0] == AX[gold[s]]
                   else "broke" if base[s][0] == AX[gold[s]] else "still wrong")
            print(f"      {s[:40]:40} truth {AX[gold[s]]:>3}  "
                  f"{base[s][0]:>3} -> {r[s][0]:>3}  w={r[s][2]:.2f}  ({tag})")

    json.dump(results, open(OUT / "geo_floor.json", "w"), indent=1)
    print(f"\nwrote {OUT}/geo_floor.json")


if __name__ == "__main__":
    main()
