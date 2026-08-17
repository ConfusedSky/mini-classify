"""Run one label set through every method this project has measured.

  python gauntlet.py                    # the `hard` set
  python gauntlet.py --set holdout      # any set, or "all"

Geometry, SigLIP alone and the ensemble under both backbones at every render
size, then every arbiter: gemma4:26b locally, haiku/sonnet through the Claude
CLI, and three Gemini models on Vertex — each at both contact-sheet sizes.

This is a *per-model* instrument, not an accuracy measurement. On five
hand-picked models a percentage means nothing; what the table is for is seeing
which methods survive a given failure and which do not.

Phases are ordered SigLIP → sheets → VLM on purpose: gemma4:26b is 17 GB on an
8 GB card and evicts SigLIP between calls if they interleave (10.1 s reload
against 0.49 s of inference).
"""
import argparse, json, time

from common import (AX, OUT, ask_claude, ask_gemma, build_sheets, build_tiles,  # sys.path
                    load_labels)

from src import pose

BACKBONES = ["google/siglip2-so400m-patch14-384", "google/siglip2-so400m-patch16-512"]
RENDER_SIZES = [384, 512, 1024, 2048]
SHEETS = [256, 512]
CLAUDE = ["haiku", "sonnet"]
GEMMA = "gemma4:26b"


def siglip_phase(labels, tiles_by_px, backbones):
    """{(backbone, px): {stem: {"sig": i, "ens": i}}} — one tower loaded at a time."""
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    import classify_stls as C

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for mid in backbones:
        model = AutoModel.from_pretrained(mid, torch_dtype=torch.float16).to(dev).eval()
        proc = AutoProcessor.from_pretrained(mid)
        up = C.embed_raw(model, proc, pose.UPRIGHT_PROMPTS, dev).float().cpu().numpy()
        dn = C.embed_raw(model, proc, pose.TOPPLED_PROMPTS, dev).float().cpu().numpy()
        for px, tiles in tiles_by_px.items():
            picks = {}
            for l in labels:
                rec = tiles[l["stem"]]
                imgs = [Image.open(p).convert("RGB") for p in rec["tiles"]]
                emb = C.embed_images(model, proc, imgs, dev).float().cpu().numpy()
                sig = pose.upright_scores(emb, up, dn)
                picks[l["stem"]] = {"sig": int(sig.argmax()),
                                    "ens": int(pose.combine_up_scores(rec["geo"], sig))}
            out[(mid, px)] = picks
        print(f"  {mid.split('/')[-1]} done")
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="hard", help="hard | orig | holdout | all")
    ap.add_argument("--skip-vlm", action="store_true")
    args = ap.parse_args()

    labels = load_labels(None if args.set == "all" else args.set)
    if not labels:
        raise SystemExit(f"no labels in set {args.set!r}")
    stems = [l["stem"] for l in labels]
    gold = {l["stem"]: l["gold"] for l in labels}
    print(f"gauntlet: {len(labels)} models from set {args.set!r}\n"
          + "\n".join(f"  {l['stem'][:44]:44} truth {l['up']}" for l in labels) + "\n")

    print("render phase")
    tiles = {px: build_tiles(labels, px) for px in RENDER_SIZES}
    geo = {s: int(pose.rank_up_scores(tiles[2048][s]["geo"])[0]) for s in stems}
    # geometry's own confidence — the hard set was chosen partly on this
    conf = {s: pose.rank_up_scores(tiles[2048][s]["geo"])[1:] for s in stems}

    print("\nSigLIP phase")
    sig = siglip_phase(labels, tiles, BACKBONES)

    print("\nsheet phase")
    sheets = build_sheets(SHEETS, labels)

    vlm = {}
    if not args.skip_vlm:
        import gemini_vlm as G
        print("\nVLM phase")
        tok = G.token()
        for t in SHEETS:
            for m in G.MODELS:
                t0 = time.time()
                vlm[(m, t)] = {s: G.ask(m, sheets[t][s], tok) for s in stems}
                print(f"  {m:20} @{t:<5} {time.time()-t0:5.0f}s")
            for m in CLAUDE:
                t0 = time.time()
                vlm[(m, t)] = {s: ask_claude(m, sheets[t][s]) for s in stems}
                print(f"  {m:20} @{t:<5} {time.time()-t0:5.0f}s")
            if pose.ollama_available():
                t0 = time.time()
                vlm[(GEMMA, t)] = {s: ask_gemma(sheets[t][s], GEMMA) for s in stems}
                print(f"  {GEMMA:20} @{t:<5} {time.time()-t0:5.0f}s")
            else:
                print(f"  {GEMMA:20} skipped — ollama not reachable at {pose.OLLAMA_URL}")

    def show(pick):
        cells = []
        for s in stems:
            v = pick.get(s)
            cell = "--" if v is None else AX[v] + ("" if v == gold[s] else "*")
            cells.append(f"{cell:>5}")
        return " ".join(cells)

    # Columns are m1..mN against the list printed above — three of the hard
    # stems share a "32mm_" prefix, so truncated names are unreadable.
    w = 40
    print(f"\n{'':{w}} " + " ".join(f"{'m'+str(i+1):>5}" for i in range(len(stems)))
          + "   correct")
    print(f"{'truth':{w}} " + " ".join(f"{AX[gold[s]]:>5}" for s in stems))
    print("-" * (w + 6 * len(stems) + 10))

    rows = [("geometry", geo)]
    for mid in BACKBONES:
        short = "p" + mid.split("-patch")[-1]
        for key, lbl in (("sig", "SigLIP alone"), ("ens", "ensemble")):
            picks = {px: {s: sig[(mid, px)][s][key] for s in stems} for px in RENDER_SIZES}
            stable = all(picks[px] == picks[RENDER_SIZES[0]] for px in RENDER_SIZES)
            rows.append((f"{lbl} {short}" + ("" if stable else "  [render-size unstable]"),
                         picks[2048]))
    rows.append(("", {}))   # separator between the local tiers and the arbiters
    for (m, t), pick in vlm.items():
        rows.append((f"{m} @{t}", pick))

    for name, pick in rows:
        if not name:
            print()
            continue
        n = sum(pick.get(s) == gold[s] for s in stems)
        print(f"{name[:w]:{w}} {show(pick)}   {n}/{len(stems)}")

    print("\nlegend: " + ",  ".join(f"m{i+1} = {s}" for i, s in enumerate(stems)))

    print("\ngeometry confidence (ratio, best score) — near-zero best = no print base")
    for s in stems:
        r, b = conf[s]
        print(f"  {s[:44]:44} ratio {r:.2f}  best {b:.4f}"
              + ("   <- has a base" if b > 0.02 else ""))

    json.dump({"set": args.set, "gold": {s: AX[gold[s]] for s in stems},
               "geometry": {s: AX[geo[s]] for s in stems},
               "siglip": {f"{m}@{px}": {s: {k: AX[v] for k, v in p.items()}
                                        for s, p in sig[(m, px)].items()}
                          for m in BACKBONES for px in RENDER_SIZES},
               "vlm": {f"{m}@{t}": {s: (AX[v] if v is not None else None)
                                    for s, v in p.items()} for (m, t), p in vlm.items()}},
              open(OUT / f"gauntlet_{args.set}.json", "w"), indent=1)
    print(f"\nwrote {OUT}/gauntlet_{args.set}.json")


if __name__ == "__main__":
    main()
