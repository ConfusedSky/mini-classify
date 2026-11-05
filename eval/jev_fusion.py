"""Jev's per-tile probabilities as a third vote inside the up-axis ensemble,
instead of an arbiter that overrides it.

    .venv/bin/python eval/jev_fusion.py

Reads `pose_baseline.py --out pose_baseline_scores.json` (the geometry and
SigLIP vectors `pose.combine_up` fuses) and the saved `decisions_vlm.py
--mode jev-tiles-clean` runs, so it spends nothing on the API. Each variant
adds `w * f(probs)` to the ensemble's combined score and is swept over `w`.
The best `w` is fitted on the models it is scored on, so the script also
reports it held out: fitted on one Jev run and scored on the other, and
fitted on a random half and scored on the rest.
"""
import json, random

import numpy as np

from common import AX, OUT, IDX                                      # REPO on sys.path

from src import pose

POOLED_OUT = ("hard", "reclaim")        # labels/README.md: never pooled
RUNS = {"run1": "decisions_jev-tiles-clean_219.json",
        "run2": "decisions_jev-tiles-clean_219_rerun.json"}
KEY = "jev-tiles-clean_sheet512"
WS = np.round(np.arange(0, 4.01, 0.05), 2)


def combined(b):
    geo, sig = np.array(b["geo_scores"]), np.array(b["siglip_scores"])
    return pose.geo_weight(geo) * pose._unit(geo) + pose._unit(sig)


VARIANTS = {
    # Probabilities added to every model's combined score.
    "add": lambda c, p, gated, w: c + w * p,
    # Only where the production gate would have escalated; the rest keep the ensemble.
    "add-gated": lambda c, p, gated, w: c + w * p if gated else c,
    # Log-probabilities, so a confident "no" vetoes harder than a confident "yes" promotes.
    "logp": lambda c, p, gated, w: c + w * np.log(p + 0.01),
    "logp-gated": lambda c, p, gated, w: c + w * np.log(p + 0.01) if gated else c,
}


def picks(rows, variant, w):
    f = VARIANTS[variant]
    return [int(np.argmax(f(r["c"], r["p"], r["gated"], w))) for r in rows]


def score(rows, variant, w):
    return sum(p == r["gold"] for p, r in zip(picks(rows, variant, w), rows))


def best_w(rows, variant):
    s = [score(rows, variant, w) for w in WS]
    return float(WS[int(np.argmax(s))]), max(s)


def main():
    base = json.loads((OUT / "pose_baseline_scores.json").read_text())["baselines"]
    runs = {}
    for name, f in RUNS.items():
        preds = {p["stem"]: p for p in json.loads((OUT / f).read_text())["predictions"]}
        rows = []
        for stem, b in base.items():
            p = preds[stem].get(f"{KEY}_probs")
            if b["set"] in POOLED_OUT or p is None:
                continue
            p = np.array(p, dtype=float)
            # Tile order == AX order; probs are saved at 2 dp, so near-ties may argmax elsewhere.
            assert p.max() - p[IDX[preds[stem][KEY]]] <= 0.02, stem
            rows.append({"stem": stem, "gold": IDX[b["gold"]], "c": combined(b), "p": p,
                         "gated": b["needs_arbiter"], "ens": IDX[b["ensemble"]]})
        runs[name] = rows
    n = len(runs["run1"])
    ens = sum(r["ens"] == r["gold"] for r in runs["run1"])
    jev = sum(int(np.argmax(r["p"])) == r["gold"] for r in runs["run1"])
    print(f"{n} pooled models | ensemble alone {ens} | Jev alone (run1) {jev}\n")

    print(f"{'variant':10} {'run':5} {'best w':>6} {'best':>5} {'other-run w':>11} "
          f"{'held out':>8} {'2-fold CV':>9} {'rescues/breaks @best':>21}")
    for v in VARIANTS:
        for name, rows in runs.items():
            other = runs["run2" if name == "run1" else "run1"]
            w, s = best_w(rows, v)
            wo, _ = best_w(other, v)
            # Random halves: fit w on one, score the other, both ways; mean over splits.
            rng, cv = random.Random(0), []
            for _ in range(50):
                idx = list(range(n)); rng.shuffle(idx)
                a, b = [rows[i] for i in idx[:n // 2]], [rows[i] for i in idx[n // 2:]]
                cv.append(score(b, v, best_w(a, v)[0]) + score(a, v, best_w(b, v)[0]))
            pk = picks(rows, v, w)
            res = sum(p == r["gold"] != r["ens"] for p, r in zip(pk, rows))
            brk = sum(p != r["gold"] == r["ens"] for p, r in zip(pk, rows))
            print(f"{v:10} {name:5} {w:6.2f} {s:5} {wo:11.2f} {score(rows, v, wo):8} "
                  f"{np.mean(cv):9.1f} {f'+{res} / -{brk}':>21}")
            if v == "logp":
                print("    changed:", ", ".join(
                    f"{r['stem']} {AX[r['ens']]}->{AX[p]}{'' if p == r['gold'] else ' WRONG'}"
                    f"{' gated' if r['gated'] else ''}"
                    for p, r in zip(pk, rows) if p != r["ens"]))

    print("\nscore by w (run1 / run2):")
    for v in VARIANTS:
        print(f"  {v}")
        for w in [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]:
            print(f"    w={w:<5} {score(runs['run1'], v, w):4} / {score(runs['run2'], v, w):4}")


if __name__ == "__main__":
    main()
