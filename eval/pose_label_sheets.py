"""Contact sheets for models nobody has labelled yet — the input to widening
the labelled set.

  python pose_label_sheets.py                       # 24 models from the newest walk
  python pose_label_sheets.py -n 60 --seed 7
  python pose_label_sheets.py --walk embed-cache4/walk-6d68....json

Samples STLs from a cache's directory walk, renders each one's six
up-candidate tiles through `Renderer.pose_tiles` (the same call, and therefore
the same pixels, the VLM arbiter is shown), lays them out with
`pose.make_contact_sheet`, and writes one PNG per model plus an `index.json`
carrying geometry's pick, its confidence, and the cached pose if the model has
one. A human reads the sheet, writes the true axis into
`../up_axis_labels.json`, and the labelled set grows.

Serves OPEN_QUESTIONS' root bottleneck — "widen the labelled set". Every
accuracy number in this repo rests on 49 models, of which 5 were hand-picked
for being hard; the honest holdout is 20. Nothing else here can get better
error bars until this does.

**It does not call `common.load_labels()`, on purpose.** Its whole job is to
reach models the labels do not cover, so it samples the walk file — the
collection as the pipeline sees it — rather than the ground truth. That makes
it the one script in `eval/` where a seeded sample is the right instrument
instead of the mistake CLAUDE.md warns about; the rule is about *scoring*
against a re-derived sample, and nothing here scores anything. `index.json`
records the seed and the walk file so a sheet can always be traced back.

The half of the retired `siglip_up.py` that scored probe wordings is
`pose_probe_sweep.py`. This half is what it should always have been: no
hardcoded `embed-cache/walk-c6c430c8....json` (that path is why the original
stopped running at all), just arguments with defaults.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from common import AX, OUT, REPO   # puts REPO on sys.path

from src import pose

RENDER_PX = 384          # what run_classify.sh renders at


def newest_walk(repo=REPO):
    """The most recently scanned walk file under any embed-cache*/ — the
    default input, so this runs with no arguments right after a classify run."""
    walks = sorted(repo.glob("embed-cache*/walk-*.json"),
                   key=lambda p: json.loads(p.read_text()).get("scanned", 0))
    if not walks:
        raise SystemExit(f"no embed-cache*/walk-*.json under {repo} — pass --walk")
    return walks[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", default=None,
                    help="cache walk file to sample (default: newest under REPO)")
    ap.add_argument("--cache-dir", default=None,
                    help="cache holding pose-cache.json + run-params.json, for the "
                         "'already resolved' column (default: the walk's own dir)")
    ap.add_argument("-n", type=int, default=24, help="models to sample")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--render-px", type=int, default=RENDER_PX)
    ap.add_argument("--thumb", type=int, default=pose.SHEET_THUMB,
                    help="contact-sheet cell size; the sheet never enlarges a "
                         "tile, so keep --render-px at or above it")
    ap.add_argument("--out", default=None, help="output dir (default OUT/label_sheets)")
    args = ap.parse_args()

    from PIL import Image
    import rig
    from classify_stls import load_run_params

    walk = Path(args.walk) if args.walk else newest_walk()
    cache_dir = Path(args.cache_dir) if args.cache_dir else walk.parent
    out = Path(args.out) if args.out else (OUT / "label_sheets")
    out.mkdir(parents=True, exist_ok=True)

    files = [Path(p) for p in json.loads(walk.read_text())["files"]]
    rng = random.Random(args.seed)
    sample = rng.sample(files, min(args.n, len(files)))

    # The pose cache is keyed relative to the root the cache was built against,
    # not to anything on this command line (classify_stls.cache_root's rule).
    root = load_run_params(cache_dir).get("collection_root")
    cached = pose.load_pose_cache(cache_dir) if root else {}
    print(f"walk {walk} ({len(files)} files) | sample {len(sample)} seed {args.seed}\n"
          f"cache {cache_dir} | root {root or 'unknown — no cached-pose column'}\n"
          f"tiles at {args.render_px}px, sheets at {args.thumb}px -> {out}")
    if args.render_px < args.thumb:
        print(f"  warning: --render-px {args.render_px} is under the sheet cell "
              f"{args.thumb}; tiles will sit padded and the sheet reads smaller "
              f"than the number suggests (eval/README, 'Watch out')")

    r = rig.rig(args.render_px)
    rows = []
    for i, f in enumerate(sample, 1):
        try:
            lm = rig.load(f)
        except Exception as e:
            print(f"[{i}/{len(sample)}] {f.stem}: load failed ({e})")
            rows.append({"file": str(f), "stem": f.stem, "error": str(e)})
            continue
        geo = pose.up_axis_scores(lm.mesh)
        gi, ratio, best = pose.rank_up_scores(geo)
        tiles = [Image.fromarray(t) for t in rig.pose_sheet_tiles(r, lm)]
        sheet = out / f"{i:03d}_{f.stem}.png"
        pose.make_contact_sheet(tiles, args.thumb).save(sheet)

        entry = cached.get(pose.file_identity(f, Path(root))) if root else None
        ci = None
        if entry:
            ci = next((j for j, u in enumerate(pose.UP_CANDIDATES)
                       if np.allclose(u, entry["up"])), None)
        rows.append({"file": str(f), "stem": f.stem, "sheet": sheet.name,
                     "geometry": {"pick": AX[gi], "ratio": round(ratio, 4),
                                  "best": round(best, 5),
                                  "has_print_base": bool(best >= pose.ABS_SCORE_FLOOR)},
                     "cached": ({"pick": AX[ci] if ci is not None else None,
                                 "source": entry.get("source")} if entry else None),
                     "label": None})     # <- the human fills this in
        print(f"[{i}/{len(sample)}] {f.stem[:44]:44} geo={AX[gi]} "
              f"(best {best:.3f}) cached="
              f"{AX[ci] if ci is not None else '--':>2}"
              f"/{(entry['source'][:4] if entry else '----'):4}  {sheet.name}",
              flush=True)

    index = out / "index.json"
    index.write_text(json.dumps(
        {"walk": str(walk), "cache_dir": str(cache_dir), "collection_root": root,
         "seed": args.seed, "n_requested": args.n, "render_px": args.render_px,
         "thumb": args.thumb, "models": rows}, indent=1))
    print(f"\nwrote {len(rows)} sheets and {index}\n"
          f"fill in each model's \"label\" (one of {', '.join(AX)}), then move the "
          f"confirmed ones into ../up_axis_labels.json")
    rig.exit_without_teardown()


if __name__ == "__main__":
    main()
