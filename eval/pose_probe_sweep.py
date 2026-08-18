"""Which upright/toppled probe wording picks the right up-candidate tile — and
does the answer depend on the tower?

  python pose_probe_sweep.py                        # every probe set, so400m
  python pose_probe_sweep.py --models a,b           # any HF SigLIP ids
  python pose_probe_sweep.py --set holdout

Renders the six up-candidate tiles per **labelled** model (`common.build_tiles`,
cached), embeds them once per backbone, and scores every probe set in `PROBES`
against the hand-labelled up axis — SigLIP alone, and ensembled with geometry
through `pose.combine_up`. Only the probe text changes between rows; the
pixels, the geometry vector and the combination are frozen.

Serves two open questions:

* **which wording** — the spread across these six sets was 83% to 4% when they
  were first tried, which is why a new phrasing gets measured here before it
  goes anywhere near `pose.UPRIGHT_PROMPTS`.
* **"would a much smaller tower do for pose?"** (OPEN_QUESTIONS) —
  `UPRIGHT_PROMPTS`/`TOPPLED_PROMPTS` were tuned against so400m, so a smaller
  tower has to be read against *its own* best wording, not against so400m's.
  Pass `--models so400m,small` and compare the columns per row, not just the
  production row.

Read LEARNINGS before quoting the `orig` column: production's probes were
selected against that set, so it scores optimistically for the production row
specifically. `holdout` is the honest number and `hard` is neither — five
hand-picked failures, reported on their own.

This replaces the labelled half of the retired `siglip_up.py`, which drew its
models from a seeded `random.sample` over a hardcoded walk file. That is the
convention this repo has a rule about (CLAUDE.md): the collection grew 509 ->
602 mid-session and the same seed stopped drawing the same models, so ground
truth loads through `common.load_labels()` and nowhere else. The unlabelled
half of siglip_up — sampling the collection for sheets to hand-label — is
`pose_label_sheets.py`.
"""
import argparse
import json

import numpy as np

from common import AX, OUT, build_tiles, load_labels   # puts REPO on sys.path

from src import pose

BASE = "google/siglip2-so400m-patch14-384"    # what production runs today
RENDER_PX = 384                               # what run_classify.sh renders at

# Wordings, positive and negative. `production` is `src/pose.py`'s own pair, so
# the table always carries the shipping answer as its baseline row.
PROBES = {
    "production": (pose.UPRIGHT_PROMPTS, pose.TOPPLED_PROMPTS),
    "upright_toppled": (
        ["a miniature figurine standing upright on its base, the way it sits on a table",
         "a miniature standing upright, head at the top and feet at the bottom"],
        ["a miniature figurine lying on its side, toppled over",
         "a miniature figurine upside down, head at the bottom"],
    ),
    "plain_orientation": (
        ["an upright 3D render of a miniature"],
        ["a sideways 3D render of a miniature",
         "an upside-down 3D render of a miniature"],
    ),
    "anatomical": (
        ["a figure with its head at the top of the image and its feet at the bottom"],
        ["a figure with its head at the bottom of the image",
         "a figure lying horizontally across the image"],
    ),
    # the collection is ~half terrain (walls, floors, crates), where "head at the
    # top" is meaningless — these avoid assuming the subject has anatomy
    "object_generic": (
        ["a 3D printed model sitting the right way up on a table"],
        ["a 3D printed model tipped onto its side",
         "a 3D printed model turned upside down"],
    ),
    "gravity": (
        ["an object resting stably on the ground the right way up"],
        ["an object lying on its side", "an object upside down"],
    ),
}


def score_backbone(model_id, tiles_by_stem, order):
    """{probe_set: {stem: {"sig": idx, "ens": idx}}} for one tower.

    One tower load, one embedding pass over the tiles, then every probe set
    scored off the same embeddings — so a difference between rows is the
    wording and nothing else.
    """
    import torch
    from PIL import Image
    import rig

    e = rig.embedder(model_id)
    banks = {k: (rig.embed_probe_texts(e, pos), rig.embed_probe_texts(e, neg))
             for k, (pos, neg) in PROBES.items()}

    out = {k: {} for k in PROBES}
    for stem in order:
        rec = tiles_by_stem[stem]
        imgs = [Image.open(p).convert("RGB") for p in rec["tiles"]]
        emb = rig.embed(e, imgs)
        for k, (pos, neg) in banks.items():
            sig = pose.upright_scores(emb, pos, neg)
            out[k][stem] = {"sig": int(sig.argmax()),
                            "ens": int(pose.combine_up_scores(rec["geo"], sig))}
    del e
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=BASE, help="comma-separated HF SigLIP ids")
    ap.add_argument("--render-px", type=int, default=RENDER_PX)
    ap.add_argument("--set", default="all", help="all | orig | holdout | hard")
    args = ap.parse_args()
    model_ids = args.models.split(",")

    labels = load_labels(None if args.set == "all" else args.set)
    if not labels:
        raise SystemExit(f"no labels in set {args.set!r}")
    order = [l["stem"] for l in labels]
    gold = {l["stem"]: l["gold"] for l in labels}
    which = {l["stem"]: l["set"] for l in labels}
    counts = {s: sum(1 for l in labels if l["set"] == s)
              for s in dict.fromkeys(l["set"] for l in labels)}
    print("labels: " + ", ".join(f"{k} {v}" for k, v in counts.items())
          + f" (total {len(order)})")

    tiles = build_tiles(labels, args.render_px)      # render phase — GPU, then freed
    print("\nSigLIP phase")
    picks = {m: score_backbone(m, tiles, order) for m in model_ids}

    sets = [(n, [s for s in order if which[s] == n]) for n in ("orig", "holdout", "hard")]
    sets.insert(2, ("orig+hold", [s for s in order if which[s] in ("orig", "holdout")]))
    sets = [(n, sel) for n, sel in sets if sel]

    def acc(pick, sel):
        return f"{sum(pick[s] == gold[s] for s in sel)}/{len(sel)}"

    geo = {s: int(pose.rank_up_scores(tiles[s]["geo"])[0]) for s in order}
    print("\ngeometry alone (probe- and backbone-independent): "
          + "  ".join(f"{n} {acc(geo, sel)}" for n, sel in sets))

    for key, title in (("sig", "SigLIP alone"), ("ens", "ensembled with geometry")):
        print(f"\n{title} — up-axis accuracy by probe wording")
        for m in model_ids:
            hdr = f"{m.split('/')[-1][:26]:26} " + " ".join(f"{n:>12}" for n, _ in sets)
            print(hdr + "\n" + "-" * len(hdr))
            for k in PROBES:
                row = [acc({s: picks[m][k][s][key] for s in order}, sel)
                       for _, sel in sets]
                print(f"  {k:24} " + " ".join(f"{c:>12}" for c in row))

    # The production row is the one the pipeline actually ships; anything that
    # beats it on `holdout` is a candidate, anything that beats it only on
    # `orig` is the tuning set talking.
    print("\nagainst the production wording, on holdout + orig")
    hold = [s for s in order if which[s] in ("orig", "holdout")]
    if hold:
        for m in model_ids:
            base_ok = sum(picks[m]["production"][s]["ens"] == gold[s] for s in hold)
            for k in PROBES:
                if k == "production":
                    continue
                ok = sum(picks[m][k][s]["ens"] == gold[s] for s in hold)
                if ok != base_ok:
                    print(f"  {m.split('/')[-1][:26]:26} {k:24} "
                          f"{ok - base_ok:+d} ({ok}/{len(hold)} vs {base_ok})")

    out = OUT / "pose_probe_sweep.json"
    json.dump({"render_px": args.render_px, "models": model_ids,
               "gold": {s: AX[gold[s]] for s in order},
               "sets": {s: which[s] for s in order},
               "picks": {m: {k: {s: {kk: AX[vv] for kk, vv in v.items()}
                                 for s, v in per.items()}
                             for k, per in picks[m].items()} for m in model_ids}},
              open(out, "w"), indent=1)
    print(f"\nwrote {out}")
    import rig                      # build_tiles may have built a renderer
    rig.exit_without_teardown()


if __name__ == "__main__":
    main()
