"""The production pose tiers over any label set, as the arbiter harnesses' input.

    python pose_baseline.py                       # every label in the file
    python pose_baseline.py --set holdout

Writes `{stem: {geometry, ensemble, margin, needs_arbiter}}` — axis *names*,
the margin, and whether the gate fires — to `OUT/pose_baseline.json`, which
`gemini_vlm.py --baselines` reads in place of the published 2026-08-12 file.

It exists because that published file covers 44 models and the arbiter
harnesses filter to it (`if l["stem"] in base`), so on the 206-label set they
would have scored the original 44 and silently ignored 162. A number that
quietly changes what it is measured on is the failure this repo has already
paid for twice.

The three tiers are composed out of `src.pose` — `up_axis_scores` →
`rank_up_scores` → `combine_up` → `needs_arbiter_margin` — over
`Renderer.pose_tiles` pixels and `Embedder` embeddings, which is the same
arrangement `parser_gate.py` uses and the same one the Poser calls. The
harness owns the ordering and nothing else.
"""
import argparse, json

from PIL import Image

from common import AX, OUT, build_tiles, load_labels   # puts REPO on sys.path

from src import pose


def _production_model() -> str:
    """The backbone the newest cache was built with, not the Embedder default."""
    from src.cachedir import load_run_params
    from src.identity import DEFAULT_MODEL
    from common import REPO
    caches = sorted(REPO.glob("embed-cache*/run-params.json"),
                    key=lambda p: p.stat().st_mtime)
    for rp in reversed(caches):
        m = load_run_params(rp.parent).get("model")
        if m:
            return m
    return DEFAULT_MODEL


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", dest="which", default=None,
                    choices=sorted({l["set"] for l in load_labels()}),
                    help="restrict to one labelled set (default: all of them)")
    ap.add_argument("--render-px", type=int, default=512,
                    help="up-candidate tile size; 512 is what embed-cache512 built")
    ap.add_argument("--model", default=None,
                    help="SigLIP id (default: the model the cache was built "
                         "with, read from its run-params)")
    ap.add_argument("--up-margin", type=float, default=pose.MARGIN_THRESHOLD,
                    help=f"gate threshold (default {pose.MARGIN_THRESHOLD}, production's)")
    ap.add_argument("--out", default="pose_baseline.json")
    args = ap.parse_args()

    import rig
    # `rig.embedder()` defaults to `identity.DEFAULT_MODEL` (patch14-384), which
    # is **not** what embed-cache512 was built with (patch16-512). Taking that
    # default silently scored a different backbone than production runs and
    # moved 7 of 206 ensemble picks, all of them low-margin. Read the cache's
    # own run-params instead, the way every other entry point resolves its
    # cache identity.
    model = args.model or _production_model()
    labels = load_labels(args.which)
    print(f"{len(labels)} labels | tiles at {args.render_px}px | "
          f"gate margin {args.up_margin} | backbone {model}")

    tiles = build_tiles(labels, args.render_px)
    emb = rig.embedder(model)

    out, fired = {}, 0
    for l in labels:
        stem = l["stem"]
        geo_scores = tiles[stem]["geo"]
        imgs = [Image.open(t).convert("RGB") for t in tiles[stem]["tiles"]]
        sig = pose.upright_scores(rig.embed(emb, imgs), emb.up_T, emb.down_T)
        gi, _ratio, _best = pose.rank_up_scores(geo_scores)
        idx, margin = pose.combine_up(geo_scores, sig)
        gate = pose.needs_arbiter_margin(margin, args.up_margin)
        fired += gate
        out[stem] = {"geometry": AX[int(gi)], "ensemble": AX[int(idx)],
                     "siglip": AX[int(sig.argmax())],
                     "margin": round(float(margin), 4), "needs_arbiter": bool(gate),
                     "set": l["set"], "gold": l["up"]}

    path = OUT / args.out
    path.write_text(json.dumps({"render_px": args.render_px,
                                "up_margin": args.up_margin,
                                "model": model,
                                "baselines": out}, indent=1))
    hit = sum(v["ensemble"] == v["gold"] for v in out.values())
    print(f"ensemble {hit}/{len(out)}; the gate fires on {fired} "
          f"({100 * fired / max(len(out), 1):.0f}%)\nwrote {path}")
    rig.exit_without_teardown()


if __name__ == "__main__":
    main()
