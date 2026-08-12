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
not depend on sample composition.
"""
import argparse, json, time

from common import AX, OUT, build_tiles, load_labels  # puts REPO on sys.path

import pose

BASE = "google/siglip2-so400m-patch14-384"       # what production runs today
CANDIDATE = "google/siglip2-so400m-patch16-512"  # higher input resolution


def embed_backbone(model_id, tiles_by_stem, order):
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
    print(f"{model_id}  loaded in {time.time()-t0:.0f}s  (processor resizes to {px}px)")

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
    ap.add_argument("--render-px", type=int, default=2048)
    args = ap.parse_args()
    model_ids = args.models.split(",")

    labels = load_labels()
    tiles = build_tiles(labels, args.render_px)          # render phase — GPU, then freed
    order = [l["stem"] for l in labels]
    gold = {l["stem"]: l["gold"] for l in labels}
    which = {l["stem"]: l["set"] for l in labels}
    geo = {s: int(pose.rank_up_scores(tiles[s]["geo"])[0]) for s in order}

    picks = {m: embed_backbone(m, tiles, order) for m in model_ids}   # embed phase
    json.dump({"gold": {s: AX[gold[s]] for s in order},
               "geometry": {s: AX[geo[s]] for s in order},
               "backbones": {m: {s: {k: AX[v] for k, v in p.items()}
                                 for s, p in picks[m].items()} for m in model_ids}},
              open(OUT / "backbone_sweep.json", "w"), indent=1)

    def acc(pick, sel):
        return f"{sum(pick[s] == gold[s] for s in sel)}/{len(sel)}"
    o = [s for s in order if which[s] == "orig"]
    h = [s for s in order if which[s] == "holdout"]

    print(f"\n{'method':52} {'orig':>6} {'holdout':>8} {'pooled':>7}")
    print(f"{'geometry alone (backbone-independent)':52} "
          f"{acc(geo,o):>6} {acc(geo,h):>8} {acc(geo,order):>7}")
    for m in model_ids:
        for key, label in (("sig", "SigLIP alone"), ("ens", "ensemble")):
            pick = {s: picks[m][s][key] for s in order}
            print(f"{label + '  ' + m.split('/')[-1]:52} "
                  f"{acc(pick,o):>6} {acc(pick,h):>8} {acc(pick,order):>7}")

    if len(model_ids) == 2:
        a, b = model_ids
        diff = [s for s in order if picks[a][s]["ens"] != picks[b][s]["ens"]]
        wins = sum(picks[b][s]["ens"] == gold[s] for s in diff)
        loses = sum(picks[a][s]["ens"] == gold[s] for s in diff)
        print(f"\nhead-to-head on the ensemble — the two disagree on {len(diff)} of "
              f"{len(order)} models")
        print(f"  {b.split('/')[-1]} right {wins}, {a.split('/')[-1]} right {loses}, "
              f"both wrong {len(diff)-wins-loses}")
        for s in diff:
            print(f"    {s[:44]:44} truth {AX[gold[s]]:>3}  "
                  f"{AX[picks[a][s]['ens']]:>3} -> {AX[picks[b][s]['ens']]:>3}"
                  + ("  (fixed)" if picks[b][s]["ens"] == gold[s] else
                     "  (broke)" if picks[a][s]["ens"] == gold[s] else ""))
        print("\nA handful of differing models is not a result — see the holdout "
              "lesson in LEARNINGS.\nSign-test the disagreements before believing "
              "either direction.")


if __name__ == "__main__":
    main()
