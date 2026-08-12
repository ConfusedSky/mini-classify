"""How do geometry and object_generic combine? The two score vectors are in
different units (flat-base area fraction vs a difference of cosine similarities),
so 'average' only means something after a normalisation choice — and the choice
is the whole experiment."""
import json
from pathlib import Path

import numpy as np

from common import OUT, AX, IDX, load_labels, mark, score

D = OUT / "siglip_up"
AX = ["+Z", "-Z", "+Y", "-Y", "+X", "-X"]
I = {a: i for i, a in enumerate(AX)}
FLOOR = 0.02  # pose.ABS_SCORE_FLOOR — "a real print base was found"

GOLD = {l["stem"]: l["gold"] for l in load_labels()}  # from up_axis_labels.json
res = json.load(open(D / "results.json"))
rows = [(i, r) for i, r in enumerate(res) if i in GOLD]
assert len(rows) == len(GOLD), f"expected {len(GOLD)} labelled rows, got {len(rows)}"

mm = lambda v: (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else np.zeros_like(v)
zs = lambda v: (v - v.mean()) / v.std() if v.std() > 0 else np.zeros_like(v)
def sm(v, T):
    e = np.exp((v - v.max()) / T)
    return e / e.sum()
def borda(v):
    r = np.empty(len(v))
    r[np.argsort(v)] = np.arange(len(v))
    return r / (len(v) - 1)


def evaluate(name, fn):
    ok = sum(int(np.argmax(fn(np.array(r["geo"]["scores"]),
                              np.array(r["siglip"]["object_generic"]["scores"]),
                              r["geo"]["best"])) == I[GOLD[i]]) for i, r in rows)
    print(f"  {name:38} {ok:2}/{len(rows)}  {ok/len(rows)*100:5.1f}%")
    return ok


print(f"{len(rows)} labelled models\n")
print("baselines")
evaluate("geometry alone", lambda g, s, b: g)
evaluate("object_generic alone", lambda g, s, b: s)
orc = sum(int(np.argmax(np.array(r["geo"]["scores"])) == I[GOLD[i]]
              or np.argmax(np.array(r["siglip"]["object_generic"]["scores"])) == I[GOLD[i]])
          for i, r in rows)
print(f"  {'oracle (either one right)':38} {orc:2}/{len(rows)}  {orc/len(rows)*100:5.1f}%   <- ceiling\n")

print("averaging schemes (equal weight)")
evaluate("min-max normalise each, mean", lambda g, s, b: mm(g) + mm(s))
evaluate("z-score each, mean", lambda g, s, b: zs(g) + zs(s))
evaluate("rank (Borda), mean", lambda g, s, b: borda(g) + borda(s))
evaluate("softmax each, mean", lambda g, s, b: sm(g, 0.03) + sm(s, 0.01))
evaluate("absolute-scaled geometry, mean",
         lambda g, s, b: np.clip(g / FLOOR, 0, 1) + mm(s))

print("\nabsolute-scaled geometry, weight swept  (w*geometry + (1-w)*siglip)")
for w in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
    evaluate(f"w = {w:.1f}", lambda g, s, b, w=w: w * np.clip(g / FLOOR, 0, 1) + (1 - w) * mm(s))

print("\nnot an average — switch on whether a base was actually found")
evaluate("geometry if best >= 0.02 else siglip",
         lambda g, s, b: g if b >= FLOOR else s)

print("\nper-model detail (absolute-scaled mean, w=0.5)")
print(f"  {'model':34} {'truth':>5} {'geo':>5} {'sig':>5} {'avg':>5}  {'best':>7}")
for i, r in rows:
    g = np.array(r["geo"]["scores"]); s = np.array(r["siglip"]["object_generic"]["scores"])
    a = 0.5 * np.clip(g / FLOOR, 0, 1) + 0.5 * mm(s)
    t = GOLD[i]
    mark = lambda p: AX[p] + ("" if AX[p] == t else "*")
    print(f"  {r['stem'][:34]:34} {t:>5} {mark(int(np.argmax(g))):>5} "
          f"{mark(int(np.argmax(s))):>5} {mark(int(np.argmax(a))):>5}  {r['geo']['best']:7.4f}")
print("  (* = wrong)")
