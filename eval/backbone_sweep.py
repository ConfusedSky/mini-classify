"""Does a higher-resolution SigLIP backbone improve the up-axis ensemble?

  python backbone_sweep.py                       # the two so400m variants
  python backbone_sweep.py --models a,b,c        # any HF SigLIP ids

Scores geometry, SigLIP alone, and the ensemble on the 44 hand-labelled models
for each backbone, with the probes and the min-max combination **frozen** —
only the vision tower changes. Renders the up-candidate tiles once and re-embeds
identical pixels per backbone, so a difference is the backbone and nothing else.

Read LEARNINGS before quoting the `orig` column: the probes were selected
against that set, so it scores optimistically for every backbone. The holdout
is the honest number, and the head-to-head at the bottom is the one that does
not depend on sample composition. `hard` models are hand-picked failures and
are reported separately — they are not a sample of anything.
"""
import argparse, json, time

from common import AX, OUT, build_tiles, load_labels  # puts REPO on sys.path

import pose

BASE = "google/siglip2-so400m-patch14-384"       # what production runs today
CANDIDATE = "google/siglip2-so400m-patch16-512"  # higher input resolution


def embed_backbone(model_id, tiles_by_stem, order, note=""):
    """Ensemble/SigLIP picks for every model under one backbone.

    Loads the tower, embeds the frozen probes and every model's 6 tiles, then
    frees the GPU before the next backbone — two so400m towers do not comfortably
    share an 8 GB card with the renderer's EGL context.
    """
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    import classify_stls as C

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to(dev).eval()
    proc = AutoProcessor.from_pretrained(model_id)
    px = getattr(proc.image_processor, "size", {}).get("height", "?")
    print(f"{model_id} {note}  loaded in {time.time()-t0:.0f}s "
          f"(processor resizes to {px}px)")

    up = C.embed_raw(model, proc, pose.UPRIGHT_PROMPTS, dev).float().cpu().numpy()
    dn = C.embed_raw(model, proc, pose.TOPPLED_PROMPTS, dev).float().cpu().numpy()
    out, t0 = {}, time.time()
    for stem in order:
        rec = tiles_by_stem[stem]
        imgs = [Image.open(p).convert("RGB") for p in rec["tiles"]]
        emb = C.embed_images(model, proc, imgs, dev).float().cpu().numpy()
        sig = pose.upright_scores(emb, up, dn)
        out[stem] = {"sig": int(sig.argmax()),
                     "ens": int(pose.combine_up_scores(rec["geo"], sig))}
    print(f"  embedded {len(order)} models in {time.time()-t0:.0f}s "
          f"({(time.time()-t0)/len(order):.2f}s each)")
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=f"{BASE},{CANDIDATE}")
    ap.add_argument("--render-px", default="384,512,1024,2048",
                    help="source render size(s) of the up-candidate tiles. The "
                         "processor downsamples to the tower's native size, so this "
                         "is what a tower has to work *from*, not what it sees.")
    args = ap.parse_args()
    model_ids = args.models.split(",")
    sizes = [int(s) for s in args.render_px.split(",")]

    labels = load_labels()
    order = [l["stem"] for l in labels]
    gold = {l["stem"]: l["gold"] for l in labels}
    which = {l["stem"]: l["set"] for l in labels}
    # Print the composition rather than assuming it: the label file grows, and a
    # bare "n=44" in a table is how a number outlives the set it was measured on.
    counts = {s: sum(1 for l in labels if l["set"] == s)
              for s in dict.fromkeys(l["set"] for l in labels)}
    print("labels: " + ", ".join(f"{k} {v}" for k, v in counts.items())
          + f" (total {len(order)})")

    tiles = {px: build_tiles(labels, px) for px in sizes}   # render phase — GPU, then freed
    geo = {s: int(pose.rank_up_scores(tiles[sizes[-1]][s]["geo"])[0]) for s in order}

    # embed phase: load each tower once, run every render size through it
    picks = {}
    for m in model_ids:
        for px in sizes:
            picks[(m, px)] = embed_backbone(m, tiles[px], order, note=f"tiles@{px}px")

    json.dump({"gold": {s: AX[gold[s]] for s in order},
               "geometry": {s: AX[geo[s]] for s in order},
               "render_px": sizes,
               "backbones": {f"{m}@{px}": {s: {k: AX[v] for k, v in p.items()}
                                           for s, p in picks[(m, px)].items()}
                             for m in model_ids for px in sizes}},
              open(OUT / "backbone_sweep.json", "w"), indent=1)

    def acc(pick, sel):
        return f"{sum(pick[s] == gold[s] for s in sel)}/{len(sel)}"
    # `hard` models were chosen because they fail, so they are reported on their
    # own — folding them into a pooled figure silently redefines what "pooled"
    # means against every number already recorded in LEARNINGS.
    sets = [("orig", [s for s in order if which[s] == "orig"]),
            ("holdout", [s for s in order if which[s] == "holdout"]),
            ("orig+hold", [s for s in order if which[s] in ("orig", "holdout")]),
            ("hard", [s for s in order if which[s] == "hard"])]
    sets = [(n, sel) for n, sel in sets if sel]

    print("\ngeometry alone (backbone- and resolution-independent): "
          + "  ".join(f"{n} {acc(geo, sel)}" for n, sel in sets))
    for key, label in (("sig", "SigLIP alone"), ("ens", "ensemble")):
        print(f"\n{label} — by source render size")
        for name, sel in sets:
            print(f"{name + f'  (n={len(sel)})':26} "
                  + " ".join(f"{str(px)+'px':>9}" for px in sizes))
            for m in model_ids:
                row = [acc({s: picks[(m, px)][s][key] for s in order}, sel) for px in sizes]
                print(f"  {m.split('/')[-1]:24} " + " ".join(f"{c:>9}" for c in row))

    if len(model_ids) == 2:
        a, b = model_ids
        print(f"\nhead-to-head on the ensemble, per render size")
        for px in sizes:
            diff = [s for s in order if picks[(a, px)][s]["ens"] != picks[(b, px)][s]["ens"]]
            wins = sum(picks[(b, px)][s]["ens"] == gold[s] for s in diff)
            loses = sum(picks[(a, px)][s]["ens"] == gold[s] for s in diff)
            print(f"  {px:>5}px: differ on {len(diff):>2} of {len(order)} — "
                  f"{b.split('/')[-1]} right {wins}, {a.split('/')[-1]} right {loses}, "
                  f"both wrong {len(diff)-wins-loses}")
        print("\nA handful of differing models is not a result — see the holdout "
              "lesson in LEARNINGS.\nSign-test the disagreements before believing "
              "either direction.")


if __name__ == "__main__":
    main()
