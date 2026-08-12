"""Run the pose-arbiter prompt through Claude models via the CLI and compare
with gemma4:26b on the same 44 hand-labelled models."""
import json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import OUT, AX, IDX, load_labels, mark, score  # puts REPO on sys.path

import pose

I = IDX
SC = OUT
MODELS = ["haiku", "sonnet"]
SHEETS = OUT / "sheets"        # one <stem>.png contact sheet per model
PREDS = OUT / "preds.json"     # geometry/ensemble/gemma predictions, keyed by stem

# Labels come from up_axis_labels.json and predictions are matched by stem.
# Do not key either on sample index: the directory walk grew 509 -> 602 files
# mid-2026-08-12, so the same random seed no longer draws the same models.
preds = {p["stem"]: p for p in json.loads(PREDS.read_text())} if PREDS.exists() else {}
items = []
for l in load_labels():
    p = preds.get(l["stem"])
    if p is None:
        continue                # no ensemble/gemma prediction yet — run tile_and_vlm.py
    items.append({"stem": l["stem"], "set": l["set"], "gold": l["gold"],
                  "sheet": SHEETS / f"{l['stem']}.png",
                  "geo": p["geo"], "ens": p["ens"], "gemma": p.get("vlm"),
                  "arb": p["arb"]})
if not items:
    raise SystemExit(f"no predictions in {PREDS} — run tile_and_vlm.py first")
missing = [it["stem"] for it in items if not it["sheet"].exists()]
if missing: sys.exit(f"missing sheets: {missing}")
print(f"{len(items)} labelled models ({sum(i['set']=='orig' for i in items)} orig "
      f"+ {sum(i['set']=='holdout' for i in items)} holdout)\n")


def ask(model, sheet):
    prompt = f"Read the image at {sheet}. {pose.UP_PROMPT}"
    for _ in range(2):
        try:
            out = subprocess.run(["claude", "-p", prompt, "--model", model,
                                  "--output-format", "json", "--max-turns", "3"],
                                 capture_output=True, text=True, timeout=300)
            if out.returncode != 0: continue
            v = pose.parse_tile_answer(json.loads(out.stdout).get("result", ""), 6)
            if v is not None: return v
        except Exception:
            pass
    return None


for model in MODELS:
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(lambda it: ask(model, it["sheet"]), items))
    for it, v in zip(items, res): it[model] = v
    n_ok = sum(v is not None for v in res)
    print(f"{model:8} done in {time.time()-t0:.0f}s  ({n_ok}/{len(items)} answered)")

json.dump([{k: (str(v) if isinstance(v, Path) else v) for k, v in it.items()} for it in items],
          open(SC/"claude_vlm.json", "w"), indent=1)

print(f"\n{'model':38} {'gold':>5} {'ens':>5} {'gemma':>6} " + " ".join(f"{m:>7}" for m in MODELS))
for it in items:
    g = it["gold"]; mk = lambda p: "--" if p is None else AX[p] + ("" if p == g else "*")
    print(f"{it['stem'][:38]:38} {AX[g]:>5} {mk(it['ens']):>5} {mk(it['gemma']):>6} "
          + " ".join(f"{mk(it[m]):>7}" for m in MODELS))

print(f"\n{'method':12} {'orig(23)':>10} {'holdout(21)':>12} {'pooled(44)':>11}")
def acc(key, sel):
    s = [x for x in sel if x[key] is not None]
    return f"{sum(x[key]==x['gold'] for x in s)}/{len(s)}"
o = [x for x in items if x["set"]=="orig"]; h = [x for x in items if x["set"]=="holdout"]
for key in ["geo","ens","gemma"] + MODELS:
    print(f"{key:12} {acc(key,o):>10} {acc(key,h):>12} {acc(key,items):>11}")

print("\n--- as the arbiter tier (overrides ensemble when needs_arbiter fires)")
arb = [x for x in items if x["arb"]]
for key in ["gemma"] + MODELS:
    s = [x for x in arb if x[key] is not None]
    resc = [x["stem"] for x in s if x[key]==x["gold"] and x["ens"]!=x["gold"]]
    brk  = [x["stem"] for x in s if x["ens"]==x["gold"] and x[key]!=x["gold"]]
    pipe = sum((x[key] if x["arb"] else x["ens"])==x["gold"] for x in items if x[key] is not None)
    print(f"  {key:8} on {len(s)} arbiter models: rescued {len(resc)}, broke {len(brk)}"
          f"  -> net {len(resc)-len(brk)}   full pipeline {pipe}/{len(items)}")
    if resc: print(f"           rescued: {resc}")
    if brk:  print(f"           broke:   {brk}")
