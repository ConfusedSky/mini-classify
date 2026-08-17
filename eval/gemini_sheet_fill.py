"""Does rendering pose tiles below SHEET_THUMB hurt the Gemini arbiter?

`make_contact_sheet` scales each tile with `Image.thumbnail`, which shrinks but
never enlarges. So at `--render-size 384` the tiles sit in the top-left of their
512 px cells with white gutters, and the arbiter sees a sheet whose content
covers 56% of the pixels a 512 px render would fill. `classify_stls.py` warns
about it; nothing had measured it.

It is not covered by the existing sweeps either. `tile_and_vlm.py` sweeps tile
resolution 384..2048 but hands the VLM only its 2048 tiles (`tiles_big`), and
`backbone_sweep.py` crosses render size with SigLIP towers — both measure the
*ensemble*. `common.build_sheets` says "only the sheet size matters to a VLM;
render_px ... made no difference", which is the ensemble result applied to a
question it was not asked.

Three configurations, one variable at a time:

  padded-384   384 px tiles, 512 px cells   what --render-size 384 does today
  filled-384   384 px tiles upscaled to 512 same detail, cells filled
  native-512   512 px tiles                 the configuration LEARNINGS measured

padded vs filled isolates the layout; filled vs native isolates real detail.

Tiles come from `build_tiles`, so rendering happens once per size and is shared.
Usage (tiles must be cached first, one render size per process — see LEARNINGS
on OffscreenRenderer):

    python eval/gemini_sheet_fill.py [--model gemini-3.5-flash] [--workers 8]

Costs one Vertex call per model per configuration: 49 x 3 with the default set.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from common import AX, OUT, build_tiles, load_labels   # puts REPO on sys.path

from src import pose

CONFIGS = ("padded-384", "filled-384", "native-512")


def sheet_for(config, tiles384, tiles512):
    """The six tiles as `config` would present them to the arbiter."""
    if config == "native-512":
        return [Image.open(p) for p in tiles512]
    tiles = [Image.open(p) for p in tiles384]
    if config == "filled-384":
        # what thumbnail refuses to do: enlarge to the cell. Adds no detail,
        # only removes the padding, which is the point of the comparison.
        tiles = [t.resize((pose.SHEET_THUMB, pose.SHEET_THUMB), Image.LANCZOS)
                 for t in tiles]
    return tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=pose.GEMINI_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    args = ap.parse_args()
    configs = args.configs.split(",")

    labels = sorted(load_labels(), key=lambda l: l["stem"])
    t384 = build_tiles(labels, render_px=384)
    t512 = build_tiles(labels, render_px=512)
    project = pose.gcloud_project()
    print(f"{len(labels)} models x {len(configs)} configs = "
          f"{len(labels) * len(configs)} Vertex calls to {args.model}\n")

    def ask(job):
        l, config = job
        tiles = sheet_for(config, t384[l["stem"]]["tiles"], t512[l["stem"]]["tiles"])
        idx = pose.ask_vlm_up(tiles, "gemini", str(OUT), args.model, project=project)
        return l["stem"], config, idx

    jobs = [(l, c) for l in labels for c in configs]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(ask, jobs))

    by_stem = {}
    for stem, config, idx in results:
        by_stem.setdefault(stem, {})[config] = idx
    gold = {l["stem"]: l["gold"] for l in labels}
    sets = {l["stem"]: l["set"] for l in labels}

    hdr = f"{'model':30} {'set':8} {'gold':>5} " + " ".join(f"{c:>12}" for c in configs)
    print(hdr + "\n" + "-" * len(hdr))
    for stem in sorted(by_stem):
        g = gold[stem]
        cells = []
        for c in configs:
            i = by_stem[stem].get(c)
            cells.append("--" if i is None else AX[i] + ("" if i == g else "*"))
        flag = "" if len(set(cells)) == 1 else "   <-- DISAGREE"
        print(f"{stem[:30]:30} {sets[stem]:8} {AX[g]:>5} "
              + " ".join(f"{c:>12}" for c in cells) + flag)

    print()
    for name in ("all", "orig", "holdout", "hard"):
        sel = [s for s in by_stem if name == "all" or sets[s] == name]
        if not sel:
            continue
        row = []
        for c in configs:
            ok = sum(by_stem[s].get(c) == gold[s] for s in sel)
            miss = sum(by_stem[s].get(c) is None for s in sel)
            row.append(f"{c} {ok}/{len(sel)}" + (f" ({miss} unanswered)" if miss else ""))
        print(f"{name:9} n={len(sel):<3} " + "   ".join(row))

    base = configs[0]
    for c in configs[1:]:
        moved = [s for s in by_stem if by_stem[s].get(c) != by_stem[s].get(base)]
        print(f"\n{base} -> {c}: answer changed on {len(moved)}/{len(by_stem)}")
        for s in moved:
            a, b = by_stem[s].get(base), by_stem[s].get(c)
            fmt = lambda i: "--" if i is None else AX[i]
            print(f"  {s[:34]:36} gold {AX[gold[s]]:>3}  {fmt(a):>3} -> {fmt(b):>3}")

    out = OUT / "gemini_sheet_fill.json"
    json.dump({"model": args.model, "configs": configs,
               "results": by_stem, "gold": gold, "sets": sets},
              open(out, "w"), indent=1, default=int)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
