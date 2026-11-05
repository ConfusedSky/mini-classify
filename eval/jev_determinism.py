"""How much Jev's answers and confidences move between identical runs.

    .venv/bin/python eval/jev_determinism.py

Reads every saved `decisions_vlm.py --mode jev-tiles-clean` run on the 219
labels (`decisions_jev-tiles-clean_219*.json`) and reports, over the runs:
per-model confidence spread, how many models change answer, and each run's
standalone, override and fused (`jev_fusion.py`'s logp, w=0.35) scores on the
201 poolable models.
"""
import json

import numpy as np

from common import AX, IDX, OUT
from jev_fusion import KEY, POOLED_OUT, combined

W = 0.35


def main():
    files = sorted(OUT.glob("decisions_jev-tiles-clean_219*.json"))
    base = json.loads((OUT / "pose_baseline_scores.json").read_text())["baselines"]
    runs = [{p["stem"]: p for p in json.loads(f.read_text())["predictions"]} for f in files]
    stems = sorted(base)
    print(f"{len(files)} runs x {len(stems)} models\n")

    # Spread of the answer's confidence and of each tile's probability, per model.
    conf_sd, prob_range, picks, unanswered = [], [], [], 0
    for s in stems:
        rs = [r[s] for r in runs if r[s].get(f"{KEY}_probs") is not None]
        unanswered += len(runs) - len(rs)
        conf_sd.append(np.std([r[f"{KEY}_conf"] for r in rs]))
        P = np.array([r[f"{KEY}_probs"] for r in rs])
        prob_range.append((P.max(0) - P.min(0)).max())
        picks.append([r[KEY] for r in rs])
    conf_sd, prob_range = np.array(conf_sd), np.array(prob_range)
    flips = [s for s, p in zip(stems, picks) if len(set(p)) > 1]
    q = lambda a: " / ".join(f"{np.percentile(a, x):.3f}" for x in (50, 90, 100))
    print(f"unanswered calls: {unanswered}")
    print(f"confidence sd per model     median / p90 / max: {q(conf_sd)}")
    print(f"widest tile-prob range      median / p90 / max: {q(prob_range)}")
    print(f"models whose answer changes across runs: {len(flips)}/{len(stems)}")
    for s in flips:
        c = {a: picks[stems.index(s)].count(a) for a in set(picks[stems.index(s)])}
        print(f"    {s[:44]:44} gold {base[s]['gold']}  "
              + "  ".join(f"{a}x{n}" for a, n in sorted(c.items(), key=lambda x: -x[1])))

    pooled = [s for s in stems if base[s]["set"] not in POOLED_OUT]
    print(f"\nper run, {len(pooled)} poolable models (ensemble alone "
          f"{sum(base[s]['ensemble'] == base[s]['gold'] for s in pooled)}):")
    print(f"  {'run':42} {'standalone':>10} {'override':>8} {'fused':>5} {'AUROC':>5}")
    for f, r in zip(files, runs):
        alone = over = fused = 0
        right, wrong = [], []
        for s in pooled:
            b, p = base[s], r[s]
            gold, ens = IDX[b["gold"]], IDX[b["ensemble"]]
            if p.get(KEY) is None:
                over += ens == gold; fused += ens == gold
                continue
            j = IDX[p[KEY]]
            alone += j == gold
            (right if j == gold else wrong).append(p[f"{KEY}_conf"])
            over += (j if b["needs_arbiter"] else ens) == gold
            c = combined(b)
            if b["needs_arbiter"]:
                c = c + W * np.log(np.array(p[f"{KEY}_probs"]) + 0.01)
            fused += int(np.argmax(c)) == gold
        auc = sum((a > w) + 0.5 * (a == w) for a in right for w in wrong) / (len(right) * len(wrong))
        print(f"  {f.stem:42} {alone:10} {over:8} {fused:5} {auc:5.2f}")


if __name__ == "__main__":
    main()
