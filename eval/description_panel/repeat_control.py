"""Determinism check: the matrix's control cell (single describer, temp 0,
standard 4-view pack) run 20x per renderer per effort. -> repeat_control.json
Then --report: exact-duplicate structure, grade distribution, pairwise
similarity, so 'temp 0 is deterministic' is measured instead of assumed."""
import json, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import os

import panel_matrix as P

if os.environ.get("GLM_MODEL"):
    P.GLM = os.environ["GLM_MODEL"]          # e.g. "@preset/glm-fixed-provider"

HERE = Path(__file__).parent
OUT = HERE / "out" / os.environ.get("REPEAT_OUT", "repeat_control.json")
N = 20


def run():
    R = json.load(open(OUT)) if OUT.exists() else {}
    jobs = []
    packs = {r: P.viewpacks(r)[0] for r in P.RENDERERS}
    for r in P.RENDERERS:
        for e in P.EFFORTS:
            key = f"{r}|{e}"
            R.setdefault(key, [])
            jobs += [(key, packs[r], e)] * (N - len(R[key]))
    print(f"{len(jobs)} calls")
    t0 = time.time()

    def one(j):
        key, pack, effort = j
        for _ in range(3):                      # a sustained 429 burst is not a result
            try:
                d, c = P.describe(pack, effort, 0)
                return key, d
            except RuntimeError as e:
                print(f"  {key}: {str(e)[:120]} — retrying", flush=True); time.sleep(30)
        return key, None

    with ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", 6))) as ex:
        for i, (key, d) in enumerate(ex.map(one, jobs), 1):
            if d is not None:
                R[key].append(d)
            if i % 20 == 0:
                json.dump(R, open(OUT, "w"), indent=1); print(f"  {i} {time.time()-t0:.0f}s", flush=True)
    json.dump(R, open(OUT, "w"), indent=1)
    json.dump(P.TELEMETRY, open(OUT.with_suffix(".telemetry.json"), "w"), indent=0)
    print(f"done {time.time()-t0:.0f}s, {len(P.TELEMETRY)} calls logged")


def grade(t):
    import re
    t = t.lower()
    burst = re.search(r"burst|erupt|emerg|protrud|break(ing)? (out|through)|tearing (out|through)|latched", t)
    host = re.search(r"human(?!oid)|host|person|crew|soldier|victim|pilot|astronaut|survivor|wearer", t)
    trans = re.search(r"transform|mutat|becoming|turning into|infected|replaced", t)
    if burst and host:
        return "B"
    if host and trans:
        return "T"
    return "A"


def report():
    import difflib
    R = json.load(open(OUT))
    print(f"{'cell':22} {'n':>3} {'exact-unique':>12} {'largest dup':>11} {'grades':>14} {'mean pairwise sim':>18}")
    for key, texts in sorted(R.items()):
        groups = Counter(t.strip() for t in texts)
        sims = [difflib.SequenceMatcher(None, a, b).ratio()
                for i, a in enumerate(texts) for b in texts[i + 1:]]
        g = Counter(grade(t) for t in texts)
        gs = " ".join(f"{k}:{v}" for k, v in sorted(g.items()))
        print(f"{key:22} {len(texts):>3} {len(groups):>12} {max(groups.values()):>11} {gs:>14} "
              f"{sum(sims)/len(sims):>18.3f}")
    print("\nexample divergence (first cell with >1 unique): ")
    for key, texts in sorted(R.items()):
        groups = Counter(t.strip() for t in texts)
        if len(groups) > 1:
            uniq = list(groups)
            for u in uniq[:3]:
                print(f"  [{grade(u)}] ({groups[u]}x) {u[:180]}...".replace("\n", " "))
            print(f"  ({key}; {len(uniq)} unique)")
            break


if __name__ == "__main__":
    report() if "--report" in sys.argv else run()
