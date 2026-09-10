"""How good is a labeller, measured against the labels we already trust?

Before widening the labelled set with anyone's proposals — a model's, a
colleague's — this asks what those proposals are worth on the 49 models whose
up axis is already settled. The answer decides whether pre-selecting a
proposal on the picking page saves the human work or quietly launders a
guess into ground truth.

It exists because the arbiter comparisons make the spread impossible to
ignore: on the same 44 sheets gemini-3.5-flash reads 43 and haiku reads 20
(LEARNINGS, 2026-08-30). "A model looked at it" is not a quality claim.

    python pose_label_calibration.py --render        # sheets + hidden truth
    ...the labeller writes picks.json: {"stem": "+Z", ...}
    python pose_label_calibration.py --score picks.json

**The two files are separate so the assessment can be honest.**
`manifest.json` carries what a labeller may see: the sheet, and nothing else.
`truth.json` carries the answers and is written once, read only by `--score`.
Geometry's pick is deliberately *not* in the manifest either — it scores 39/44
on its own, so showing it makes the labeller partly a reader of geometry, and
the agreement number would then be measuring the wrong thing.

The sheets are `rig.pose_sheet_tiles` through `pose.make_contact_sheet`: the
same six numbered tiles, in the same `AX` order, that the VLM arbiter is shown
and that the original 49 labels were read from. So a number here is comparable
to the arbiter tables, and the labeller sees no more than gemini did.
"""
import argparse
import json
from pathlib import Path

from common import AX, IDX, OUT, load_labels

from src import pose


def render(out: Path, render_px: int, thumb: int, which: str | None) -> None:
    from PIL import Image
    import rig

    labels = load_labels(which)
    out.mkdir(parents=True, exist_ok=True)
    r = rig.rig(render_px)
    manifest, truth = [], {}
    for i, lab in enumerate(labels, 1):
        stem = lab["stem"]
        try:
            lm = rig.load(lab["path"])
        except Exception as e:                      # a label whose file moved
            print(f"[{i}/{len(labels)}] {stem}: load failed ({e})")
            continue
        tiles = [Image.fromarray(t) for t in rig.pose_sheet_tiles(r, lm)]
        sheet = out / f"{i:03d}_{stem}.png"
        pose.make_contact_sheet(tiles, thumb).save(sheet)
        manifest.append({"i": i, "stem": stem, "sheet": sheet.name})
        truth[stem] = {"up": lab["up"], "set": lab["set"]}
        print(f"[{i}/{len(labels)}] {stem[:50]:50} {sheet.name}", flush=True)

    (out / "manifest.json").write_text(json.dumps(
        {"axes": AX, "render_px": render_px, "thumb": thumb,
         "note": "tile n is axes[n-1]; the sheet numbers tiles 1..6",
         "models": manifest}, indent=1))
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    print(f"\nwrote {len(manifest)} sheets, manifest.json and truth.json to {out}\n"
          f"the labeller reads the sheets and writes {{stem: axis}} to picks.json;\n"
          f"truth.json stays shut until --score")
    rig.exit_without_teardown()


def score(out: Path, picks_file: Path) -> None:
    truth = json.loads((out / "truth.json").read_text())
    picks = json.loads(picks_file.read_text())
    if isinstance(picks, dict) and "picks" in picks:
        picks = picks["picks"]

    rows, missing = [], []
    for stem, t in truth.items():
        got = picks.get(stem)
        if got is None:
            missing.append(stem)
            continue
        if got not in IDX:
            raise SystemExit(f"{stem}: {got!r} is not one of {', '.join(AX)}")
        rows.append((stem, t["set"], t["up"], got))

    by_set: dict[str, list[bool]] = {}
    for _, s, up, got in rows:
        by_set.setdefault(s, []).append(up == got)
    hits = sum(up == got for _, _, up, got in rows)

    print(f"agreement {hits}/{len(rows)}"
          f"{f' ({100 * hits / len(rows):.0f}%)' if rows else ''}")
    for s in sorted(by_set):
        got = by_set[s]
        print(f"  {s:8} {sum(got)}/{len(got)}")
    if missing:
        print(f"  no pick for {len(missing)}: {', '.join(sorted(missing))}")
    dis = [(stem, s, up, got) for stem, s, up, got in rows if up != got]
    if dis:
        print("\ndisagreements (label vs pick):")
        for stem, s, up, got in dis:
            print(f"  {stem[:52]:52} {s:8} {up:>2} vs {got:>2}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--render", action="store_true",
                    help="render the sheets and write manifest.json + truth.json")
    ap.add_argument("--score", metavar="PICKS.JSON", default=None,
                    help="compare a labeller's picks against truth.json")
    # the choices come from the file, not from here: a set added by
    # `pose_label_expand.py --merge` must be measurable the day it lands
    ap.add_argument("--set", dest="which", default=None,
                    choices=sorted({l["set"] for l in load_labels()}),
                    help="restrict to one labelled set (default: all of them)")
    ap.add_argument("--render-px", type=int, default=384,
                    help="tile size, as run_classify.sh renders")
    ap.add_argument("--thumb", type=int, default=pose.SHEET_THUMB,
                    help="contact-sheet cell size; 512 is what the arbiter sees")
    ap.add_argument("--out", default=None,
                    help="output dir (default OUT/label_calibration)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else (OUT / "label_calibration")
    if args.render:
        render(out, args.render_px, args.thumb, args.which)
    elif args.score:
        score(out, Path(args.score))
    else:
        ap.error("pass --render or --score PICKS.JSON")


if __name__ == "__main__":
    main()
