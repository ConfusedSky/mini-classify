"""Two questions on the 23 hand-labelled models:
  1. does up-candidate tile resolution change ensemble accuracy?
  2. what does the ollama VLM tier actually add over the ensemble?
"""
import json, sys, os
import numpy as np, torch
from pathlib import Path

import classify_stls as C, pose
from transformers import AutoModel, AutoProcessor

from common import OUT, AX, IDX, load_labels, mark, score

D = OUT / "siglip_up"
AX = ["+Z", "-Z", "+Y", "-Y", "+X", "-X"]
I = {a: i for i, a in enumerate(AX)}
GOLD = {l["stem"]: l["gold"] for l in load_labels()}  # from up_axis_labels.json
SIZES = [384, 512, 1024, 2048]

res = json.load(open(D / "results.json"))
rows = [(i, res[i]) for i in sorted(GOLD)]

dev = "cuda" if torch.cuda.is_available() else "cpu"
mid = "google/siglip2-so400m-patch14-384"
model = AutoModel.from_pretrained(mid, torch_dtype=torch.float16).to(dev).eval()
proc = AutoProcessor.from_pretrained(mid)
upT = C.embed_raw(model, proc, pose.UPRIGHT_PROMPTS, dev).float().cpu().numpy()
dnT = C.embed_raw(model, proc, pose.TOPPLED_PROMPTS, dev).float().cpu().numpy()
renderers = {s: C.make_renderer(s) for s in SIZES}   # kept alive; teardown warns otherwise

print(f"{len(rows)} labelled models | tile sizes {SIZES} | VLM = ollama gemma4:26b\n")
hdr = f"{'model':26} {'gold':>5} {'geo':>5} " + " ".join(f"{'e'+str(s):>6}" for s in SIZES) + f" {'vlm':>5} {'arb':>4}"
print(hdr); print("-" * len(hdr))

out = []
for idx, r in rows:
    f = Path(r["file"])
    mesh = C.load_mesh(f)
    gs = pose.up_axis_scores(mesh)
    gi, ratio, best = pose.rank_up_scores(gs)
    arb = pose.needs_arbiter(ratio, best, 0.6)
    ens = {}
    tiles_big = None
    for s in SIZES:
        tiles = C.render_up_candidate_tiles(renderers[s], mesh)
        if s == 2048:
            tiles_big = tiles
        sc = pose.upright_scores(
            C.embed_images(model, proc, tiles, dev).float().cpu().numpy(), upT, dnT)
        ens[s] = pose.combine_up_scores(gs, sc)
    vlm = pose.ask_vlm_up(tiles_big, "ollama", ".", "gemma4:26b")
    g = I[GOLD[idx]]
    mk = lambda p: ("--" if p is None else AX[p] + ("" if p == g else "*"))
    print(f"{r['stem'][:26]:26} {GOLD[idx]:>5} {mk(gi):>5} "
          + " ".join(f"{mk(ens[s]):>6}" for s in SIZES)
          + f" {mk(vlm):>5} {'Y' if arb else '.':>4}", flush=True)
    out.append({"stem": r["stem"], "gold": g, "geo": gi, "ens": ens, "vlm": vlm, "arb": arb})

json.dump(out, open(D / "tile_vlm.json", "w"), indent=1, default=int)
n = len(out)
print(f"\n{'method':34} {'correct':>9}")
print(f"{'geometry alone':34} {sum(o['geo']==o['gold'] for o in out):>4}/{n}")
for s in SIZES:
    print(f"{'ensemble @ '+str(s)+'px tiles':34} {sum(o['ens'][s]==o['gold'] for o in out):>4}/{n}")
answered = [o for o in out if o["vlm"] is not None]
print(f"{'VLM alone (all models)':34} {sum(o['vlm']==o['gold'] for o in answered):>4}/{len(answered)}"
      f"   ({n-len(answered)} unanswered)")

# production pipeline: ensemble, then VLM overrides when needs_arbiter fired
E = 2048
for label, sel in (("all models", out), ("needs_arbiter only", [o for o in out if o["arb"]])):
    sub = [o for o in sel if o["vlm"] is not None]
    if not sub: continue
    ens_ok = sum(o["ens"][E] == o["gold"] for o in sub)
    pipe_ok = sum((o["vlm"] if o["arb"] else o["ens"][E]) == o["gold"] for o in sub)
    dis = [o for o in sub if o["vlm"] != o["ens"][E]]
    helped = sum(o["vlm"] == o["gold"] and o["ens"][E] != o["gold"] for o in dis)
    hurt = sum(o["ens"][E] == o["gold"] and o["vlm"] != o["gold"] for o in dis)
    print(f"\n--- {label} (n={len(sub)})")
    print(f"  ensemble alone      {ens_ok}/{len(sub)}")
    print(f"  ensemble + VLM tier {pipe_ok}/{len(sub)}")
    print(f"  disagreements {len(dis)}: VLM rescued {helped}, VLM broke {hurt}, "
          f"both wrong {len(dis)-helped-hurt}")
