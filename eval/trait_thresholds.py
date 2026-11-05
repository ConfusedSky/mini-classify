"""Sweep the yes/no cutoff on Jev's two `model_traits.py` answers, each question
on its own, from the saved run. No API calls.

    .venv/bin/python eval/trait_thresholds.py

`part` is scored against the hand triage. `ambiguous` has no hand truth, so it
is scored against two stand-ins: the whole models, which a person gave an up
axis, and gemini's saved answers, which are a reference and not ground truth.
Cutoffs are chosen on the models they are scored on, so `part` also reports
them held out on random halves.
"""
import json, random

import numpy as np

from common import OUT

CUTS = np.round(np.arange(0.05, 0.96, 0.05), 2)
VARIANTS = ("jev", "jev_name")


def auroc(pos, neg):
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def rates(pos, neg, t):
    """(share of pos at or above t, share of neg at or above t)."""
    return np.mean(np.array(pos) >= t), np.mean(np.array(neg) >= t)


def youden(pos, neg):
    """Cutoff with the widest gap between the two rates; ties take the middle."""
    j = [np.subtract(*rates(pos, neg, t)) for t in CUTS]
    return float(np.median(CUTS[np.isclose(j, max(j))]))


def table(pos, neg, pos_name, neg_name):
    print(f"  {'cut':>5} {pos_name:>14} {neg_name:>14} {'gap':>6}")
    for t in CUTS:
        a, b = rates(pos, neg, t)
        print(f"  {t:5.2f} {f'{int(round(a * len(pos)))}/{len(pos)}':>14} "
              f"{f'{int(round(b * len(neg)))}/{len(neg)}':>14} {a - b:6.2f}")


def held_out(pos, neg, splits=200):
    rng, got = random.Random(0), []
    for _ in range(splits):
        p, n = rng.sample(pos, len(pos)), rng.sample(neg, len(neg))
        for fit, test in (((p[::2], n[::2]), (p[1::2], n[1::2])),
                          ((p[1::2], n[1::2]), (p[::2], n[::2]))):
            got.append(rates(*test, youden(*fit)))
    return np.mean(got, axis=0)


def main():
    d = json.loads((OUT / "model_traits.json").read_text())
    items = d["items"]
    truth = [r for r in items if r["part_truth"] is not None]
    gem = [r for r in items if r.get("gemini")]
    dis = [r for r in items if r["disagree"]]
    whole = [r for r in truth if not r["part_truth"]]

    for v in VARIANTS:
        pos = [r[v]["part"] for r in truth if r["part_truth"]]
        neg = [r[v]["part"] for r in whole]
        t = youden(pos, neg)
        hr, hf = held_out(pos, neg)
        print(f"\npart / {v}: hand truth, AUROC {auroc(pos, neg):.3f}, best cut {t:.2f}, "
              f"held out {100 * hr:.0f}% caught / {100 * hf:.0f}% whole wrongly called")
        table(pos, neg, "parts caught", "whole wrong")

    for v in VARIANTS:
        pos = [r[v]["ambiguous"] for r in gem if r["gemini"]["ambiguous"]]
        neg = [r[v]["ambiguous"] for r in gem if not r["gemini"]["ambiguous"]]
        print(f"\nambiguous / {v}: against gemini, AUROC {auroc(pos, neg):.3f}, "
              f"best cut {youden(pos, neg):.2f}")
        table(pos, neg, "gemini yes", "gemini no")
        w = np.array([r[v]["ambiguous"] for r in whole])
        s = np.array([r[v]["ambiguous"] for r in items if r["disagree"] is not None])
        print("  yes on hand-whole models / on all gated sample:")
        print("  " + "  ".join(f"{t:.2f}: {(w >= t).sum()}/{(s >= t).sum()}" for t in CUTS[::2]))

    print(f"\ndisagreements left to label, of {len(dis)} (Jev+name says neither, by cutoff pair):")
    cuts = [.5, .6, .7, .8, .9]
    print("  part\\amb " + "".join(f"{c:>6.2f}" for c in cuts))
    for pc in cuts:
        print(f"  {pc:8.2f} " + "".join(
            f"{sum(r['jev_name']['part'] < pc and r['jev_name']['ambiguous'] < ac for r in dis):6}"
            for ac in cuts))


if __name__ == "__main__":
    main()
