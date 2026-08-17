"""Gate the VLM arbiter on the ensemble's own margin instead of geometry's
confidence.

  python arbiter_gate.py                 # sweep, all arbiters
  python arbiter_gate.py --arbiter gemini-3.5-flash_sheet512

Today `needs_arbiter(ratio, best)` asks geometry how sure *it* is, and fires
whenever geometry has no print base — including on models the ensemble already
had right, which is how the tier keeps measuring net zero. The obvious
alternative is to ask the combined score vector how sure *it* is:

    margin = top1 - top2 of (_unit(geo) + _unit(siglip))    # 0..2

and escalate only the models where that margin is small. This sweeps the
threshold, reports pipeline accuracy against how often the gate fires (each
firing is one API call), and puts the current gate on the same axes.

The threshold is a tuned parameter. Pick it on `orig` and read `holdout` —
LEARNINGS has the story of a 17-point gap that became −5 on fresh data.
"""
import argparse, json

import numpy as np

from common import AX, IDX, OUT, RESULTS_FILE, build_tiles, load_labels  # sys.path

from src import pose

BACKBONE = "google/siglip2-so400m-patch14-384"
RENDER_PX = 2048


def ensemble_state(labels, views):
    """{stem: {"ens", "margin", "ratio", "best"}} per view-count.

    `views=1` is today's pipeline: one up-candidate tile per axis. `views=4`
    averages the upright score over the four azimuths perpendicular to each
    axis. Both read the *same* orbit tiles — azimuth 0 is the 1-view tile — so
    the comparison is view count alone, with render size held fixed.
    """
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    import classify_stls as C
    from front_first import N_AZ, RENDER_PX as ORBIT_PX, build_orbit_tiles

    orbit = build_orbit_tiles(labels, ORBIT_PX)
    geo_src = build_tiles(labels, RENDER_PX)          # geometry only; already cached
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(BACKBONE, torch_dtype=torch.float16).to(dev).eval()
    proc = AutoProcessor.from_pretrained(BACKBONE)
    up = C.embed_raw(model, proc, pose.UPRIGHT_PROMPTS, dev).float().cpu().numpy()
    dn = C.embed_raw(model, proc, pose.TOPPLED_PROMPTS, dev).float().cpu().numpy()

    out = {v: {} for v in views}
    for l in labels:
        flat = [p for row in orbit[l["stem"]] for p in row]
        emb = C.embed_images(model, proc,
                             [Image.open(p).convert("RGB") for p in flat],
                             dev).float().cpu().numpy()
        grid = pose.upright_scores(emb, up, dn).reshape(6, N_AZ)
        geo = np.asarray(geo_src[l["stem"]]["geo"])
        _, ratio, best = pose.rank_up_scores(geo)
        for v in views:
            sig = grid[:, 0] if v == 1 else grid.mean(1)
            idx, margin = pose.combine_up(geo, sig)   # the real combination, weights and all
            out[v][l["stem"]] = {"ens": idx, "margin": margin,
                                 "ratio": float(ratio), "best": float(best)}
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


def load_arbiters(stems):
    """{arbiter: {stem: idx}} from every recorded VLM run that covers these models."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arbiter", default=None, help="only sweep this one")
    ap.add_argument("--steps", type=int, default=21)
    ap.add_argument("--views", default="1,4",
                    help="up-candidate views per axis to fold into the ensemble")
    args = ap.parse_args()

    views = [int(v) for v in args.views.split(",")]
    labels = load_labels()
    states = ensemble_state(labels, views)
    arbiters = load_arbiters([l["stem"] for l in labels])
    if args.arbiter:
        arbiters = {k: v for k, v in arbiters.items() if k == args.arbiter}
        if not arbiters:
            raise SystemExit(f"no recorded run for {args.arbiter!r}")

    gold = {l["stem"]: l["gold"] for l in labels}
    sets = {"orig": [], "holdout": [], "hard": []}
    for l in labels:
        sets[l["set"]].append(l["stem"])
    sets["orig+hold"] = sets["orig"] + sets["holdout"]

    def evaluate(stems, fires, pick):
        """(correct, n, n_fired) for the pipeline: ensemble unless the gate fires."""
        n_fired = ok = 0
        for s in stems:
            v = state[s]["ens"]
            if fires(s) and pick.get(s) is not None:
                v, n_fired = pick[s], n_fired + 1
            elif fires(s):
                n_fired += 1          # would have cost a call even if unanswered
            ok += v == gold[s]
        return ok, len(stems), n_fired

    geo_gate = lambda s: pose.needs_arbiter(state[s]["ratio"], state[s]["best"])
    thresholds = [round(t, 3) for t in np.linspace(0, 1.0, args.steps)]

    for view, name in [(v, n) for v in views
                       for n in ("orig", "holdout", "orig+hold", "hard")]:
        state = states[view]
        stems = [s for s in sets[name] if s in state]
        if not stems:
            continue
        base_ok = sum(state[s]["ens"] == gold[s] for s in stems)
        n_geo = sum(geo_gate(s) for s in stems)
        print(f"\n=== {name} (n={len(stems)}), {view}-view ensemble — alone "
              f"{base_ok}/{len(stems)}, "
              f"geometry gate fires on {n_geo} ({100*n_geo/len(stems):.0f}%)")
        hdr = f"{'arbiter':30} {'geometry gate':>14} | " + \
              " ".join(f"{t:>5}" for t in thresholds[:11])
        print(hdr)
        print(f"{'':30} {'':>14} | " + " ".join(f"{'':>5}" for _ in thresholds[:11])
              + "   <- margin threshold, correct/fires")
        for a in sorted(arbiters):
            cov = [s for s in stems if s in arbiters[a]]
            if len(cov) < len(stems):
                continue                     # only compare on full coverage
            ok, n, fired = evaluate(stems, geo_gate, arbiters[a])
            cells = []
            for t in thresholds[:11]:
                o, _, f = evaluate(stems, lambda s, t=t: state[s]["margin"] < t,
                                   arbiters[a])
                cells.append(f"{o:>2}/{f:<2}")
            print(f"{a[:30]:30} {f'{ok}/{fired}':>14} | " + " ".join(f"{c:>5}" for c in cells))

    # where the two gates disagree, on the set that matters
    state = states[views[0]]
    print("\nmodels the geometry gate escalates but a tight margin gate would not")
    for s in sets["orig+hold"]:
        if s not in state:
            continue
        st = state[s]
        if geo_gate(s) and st["margin"] >= 0.5:
            right = "ensemble right" if st["ens"] == gold[s] else "ensemble WRONG"
            print(f"  {s[:44]:44} margin {st['margin']:.2f}  best {st['best']:.4f}  {right}")

    json.dump({f"{view}view": {s: dict(v, gold=AX[gold[s]], ens_ax=AX[v["ens"]])
                              for s, v in states[view].items()} for view in views},
              open(OUT / "arbiter_gate.json", "w"), indent=1)
    print(f"\nwrote {OUT}/arbiter_gate.json")


if __name__ == "__main__":
    main()
