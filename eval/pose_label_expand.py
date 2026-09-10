"""Widen the labelled set: sample, render, propose, and let a human decide.

    python pose_label_expand.py --render -n 170          # sheets for new models
    ...a labeller writes proposals.json: {stem: {"up": "+Z", "risky": false}}
    python pose_label_expand.py --page proposals.json    # the picking page
    ...the human opens index.html, picks, exports confirmed.json
    python pose_label_expand.py --merge confirmed.json --set expand

Three properties this is built around, all of them lessons the repo already
paid for:

* **The sample excludes what is already labelled**, by path, because a widened
  set that re-draws the same models widens nothing. It samples the walk file —
  the collection as the pipeline sees it — for the same reason
  `pose_label_sheets.py` does: the job is to reach models the labels do not
  cover (CLAUDE.md's rule is about *scoring* against a re-derived sample, and
  nothing here scores).
* **The proposal is a proposal.** A label the pipeline wrote is not ground
  truth for the pipeline — it grades itself, and the models it gets wrong are
  exactly the ones it would label wrong. So proposals here come from a labeller
  measured first against the existing 49 (`pose_label_calibration.py`), the
  page shows the six candidate tiles rather than the proposal alone, and what
  lands in `up_axis_labels.json` is what the *human* pressed.
* **`undefined` is a first-class answer.** The original set excluded models
  whose upright is genuinely undefined — loose hands, wings, swords, pipes,
  pins. A page that only offers six axes forces a lie on exactly those.

The sheets are `rig.pose_sheet_tiles` through `pose.make_contact_sheet`: the
six numbered tiles in `AX` order that the arbiter is shown and that the first
49 labels were read from.
"""
import argparse
import html
import json
import os
import random
from pathlib import Path
from urllib.parse import quote

from common import AX, IDX, OUT, REPO, load_labels

from src import naming, pose

RENDER_PX = 384


def newest_walk(repo=REPO):
    """The most recently scanned walk under any embed-cache*/."""
    walks = sorted(repo.glob("embed-cache*/walk-*.json"),
                   key=lambda p: json.loads(p.read_text()).get("scanned", 0))
    if not walks:
        raise SystemExit(f"no embed-cache*/walk-*.json under {repo} — pass --walk")
    return walks[-1]


def render(out: Path, walk: Path, n: int, seed: int, render_px: int,
           thumb: int, exclude: list[Path] | None = None) -> None:
    from PIL import Image
    import rig

    labelled = {str(Path(l["path"]).resolve()) for l in load_labels()}
    # A top-up batch must also skip what an earlier batch already drew — those
    # models are not labelled *yet*, so `load_labels` cannot see them, and a
    # second draw that re-renders them widens nothing (the same reason the
    # first draw skips the labelled set).
    for m in exclude or []:
        root = json.loads(m.read_text()).get("collection_root", "")
        labelled |= {str((Path(root) / e["path"]).resolve())
                     for e in json.loads(m.read_text())["models"]}
    files = [Path(p) for p in json.loads(walk.read_text())["files"]]
    # A walk file is a record of what the vocabulary said *then*. Re-apply it
    # here, because a draw is for the collection as it is defined now — the
    # `sprue` cut (2026-09-09) landed after this walk was written, and without
    # this the sample offers models the next walk will not index.
    files = [f for f in files if not any(naming.skip(part) for part in f.parts)]
    fresh = [f for f in files if str(f.resolve()) not in labelled]
    if len(fresh) < n:
        raise SystemExit(f"walk holds {len(fresh)} unlabelled files, asked for {n}")
    sample = random.Random(seed).sample(fresh, n)

    out.mkdir(parents=True, exist_ok=True)
    print(f"walk {walk}: {len(files)} files, {len(files) - len(fresh)} already "
          f"labelled\nsampling {n} of the remaining {len(fresh)}, seed {seed}\n"
          f"tiles at {render_px}px, sheets at {thumb}px -> {out}")

    r = rig.rig(render_px)
    rows = []
    for i, f in enumerate(sample, 1):
        try:
            lm = rig.load(f)
        except Exception as e:                          # unreadable STL
            print(f"[{i}/{n}] {f.stem}: load failed ({e})")
            continue
        tiles = [Image.fromarray(t) for t in rig.pose_sheet_tiles(r, lm)]
        sheet = out / f"{i:03d}_{f.stem}.png"
        pose.make_contact_sheet(tiles, thumb).save(sheet)
        rows.append({"i": i, "stem": f.stem,
                     "path": str(f.relative_to(f.anchor)), "sheet": sheet.name})
        print(f"[{i}/{n}] {f.stem[:52]:52} {sheet.name}", flush=True)

    (out / "manifest.json").write_text(json.dumps(
        {"walk": str(walk), "seed": seed, "axes": AX, "render_px": render_px,
         "thumb": thumb, "collection_root": _root_of(sample[0]),
         "models": rows}, indent=1))
    print(f"\nwrote {len(rows)} sheets and manifest.json to {out}")
    rig.exit_without_teardown()


def _root_of(sample_path: Path) -> str:
    """The mount the sample came from, recorded so the set can be relocated —
    `up_axis_labels.json`'s own `collection_root` convention."""
    labels = json.loads((REPO / "up_axis_labels.json").read_text())
    return labels["collection_root"]


PAGE_CSS = """
:root { --bg:#f4f4f5; --card:#fff; --line:#d4d4d8; --ink:#18181b;
        --dim:#71717a; --pick:#2563eb; --warn:#b45309; --warnbg:#fef3c7; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:14px/1.5 ui-sans-serif,system-ui,sans-serif; }
header { position:sticky; top:0; z-index:5; background:var(--card);
         border-bottom:1px solid var(--line); padding:10px 16px;
         display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
header b { font-size:15px; }
.count { color:var(--dim); }
.keys { color:var(--dim); font-size:12px; margin-left:auto; }
main { padding:16px; display:flex; flex-direction:column; gap:14px; }
.m { background:var(--card); border:1px solid var(--line); border-radius:8px;
     padding:12px; }
.m.cur { border-color:var(--pick); box-shadow:0 0 0 2px rgba(37,99,235,.15); }
.m.done { opacity:.55; }
.hd { display:flex; gap:10px; align-items:baseline; margin-bottom:8px; }
.hd .n { color:var(--dim); font-variant-numeric:tabular-nums; }
.hd .stem { font-weight:600; word-break:break-all; }
.flag { background:var(--warnbg); color:var(--warn); border-radius:4px;
        padding:1px 7px; font-size:12px; white-space:nowrap; }
img { width:100%; max-width:900px; display:block; border:1px solid var(--line);
      border-radius:4px; background:#fff; }
.row { display:flex; gap:6px; flex-wrap:wrap; margin-top:10px; }
button { font:inherit; padding:6px 12px; border:1px solid var(--line);
         background:var(--card); border-radius:6px; cursor:pointer; }
button:hover { border-color:var(--pick); }
button.sel { background:var(--pick); border-color:var(--pick); color:#fff; }
button.und.sel { background:var(--warn); border-color:var(--warn); }
button .k { color:var(--dim); font-size:11px; margin-right:5px; }
button.sel .k { color:#dbeafe; }
.mine { color:var(--dim); font-size:12px; margin-top:6px; }
textarea { width:100%; height:150px; margin-top:12px; font-family:ui-monospace,
           monospace; font-size:12px; border:1px solid var(--line);
           border-radius:6px; padding:8px; }
"""

PAGE_JS = """
const KEYS = {"1":0,"2":1,"3":2,"4":3,"5":4,"6":5};
const store = "poselabels:" + location.pathname;
let picks = {};
try { picks = JSON.parse(localStorage.getItem(store) || "{}"); } catch (e) {}
let cur = 0;

function save() {
  try { localStorage.setItem(store, JSON.stringify(picks)); } catch (e) {}
  const done = Object.keys(picks).length;
  const und = Object.values(picks).filter(v => v === "undefined").length;
  document.getElementById("count").textContent =
    done + " of " + MODELS.length + " decided \\u00b7 " + und + " undefined";
}

function paint(i) {
  const m = MODELS[i], el = document.getElementById("m" + i);
  const got = picks[m.stem];
  el.classList.toggle("done", got !== undefined);
  el.querySelectorAll("button[data-ax]").forEach(b => {
    b.classList.toggle("sel", b.dataset.ax === got);
  });
}

function pick(i, ax) {
  picks[MODELS[i].stem] = ax;
  paint(i); save();
  if (i === cur) focus(Math.min(cur + 1, MODELS.length - 1));
}

function focus(i) {
  document.getElementById("m" + cur)?.classList.remove("cur");
  cur = i;
  const el = document.getElementById("m" + cur);
  el.classList.add("cur");
  el.scrollIntoView({block: "center", behavior: "smooth"});
}

document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA") return;
  if (!TRIAGE && e.key in KEYS) { pick(cur, AXES[KEYS[e.key]]); e.preventDefault(); }
  else if (TRIAGE && e.key === "k") { pick(cur, "keep"); e.preventDefault(); }
  else if (e.key === "u") { pick(cur, "undefined"); e.preventDefault(); }
  else if (e.key === "n" || e.key === "j" || e.key === "ArrowDown") {
    focus(Math.min(cur + 1, MODELS.length - 1)); e.preventDefault();
  } else if (e.key === "p" || e.key === "k" || e.key === "ArrowUp") {
    focus(Math.max(cur - 1, 0)); e.preventDefault();
  }
});

function exportJSON() {
  const out = JSON.stringify({picks: picks}, null, 1);
  document.getElementById("out").value = out;
  try {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([out], {type: "application/json"}));
    a.download = "confirmed.json";
    a.click();
  } catch (e) {}
}
"""


def page(dirs: list[Path], proposals_file: Path | None, page_dir: Path,
         triage: bool = False) -> None:
    """One page over one or more rendered batches.

    Two modes, because the two questions cost their own reader very different
    amounts. **Triage** asks only *is this a model or a loose part?* — a
    glance for a person, and the answer a labeller cannot skip past, since
    roughly 40% of this collection is hands, blades and wings whose upright is
    undefined by convention (`part_probe.py`, LEARNINGS 2026-09-03). **Pick**
    asks for the axis, over the survivors, with a proposal pre-selected.

    Splitting them is what keeps the expensive pass small: a proposal is only
    worth making for a model that can carry a label at all."""
    models, sheets, stale = [], {}, []
    for d in dirs:
        manifest = json.loads((d / "manifest.json").read_text())
        for m in manifest["models"]:
            # rendered before a vocabulary change that has since cut it: asking
            # for a label here would grow the set with a model the walk drops
            if any(naming.skip(part) for part in Path(m["path"]).parts):
                stale.append(m["stem"])
                continue
            models.append(m)
            # percent-encode, then HTML-escape: these filenames carry spaces,
            # apostrophes, ampersands and parentheses, and a bare `#` in one
            # would otherwise cut the URL short at a fragment
            rel = os.path.relpath(d / m["sheet"], page_dir)
            sheets[m["stem"]] = quote(rel)
    proposals = {}
    if proposals_file:
        raw = json.loads(proposals_file.read_text())
        proposals = raw.get("proposals", raw)
    if triage:
        # only the models nobody has ruled out yet, so a second triage pass
        # over a topped-up draw does not re-ask what was already answered
        models = [m for m in models if m["stem"] not in proposals]

    cards = []
    for k, m in enumerate(models):
        p = proposals.get(m["stem"], {})
        if isinstance(p, str):
            p = {"up": p}
        mine, risky = p.get("up"), bool(p.get("risky"))
        why = p.get("why", "")
        flag = (f'<span class="flag">look closely — {html.escape(why)}</span>'
                if risky else "")
        if triage:
            buttons = (
                f'<button data-ax="keep" onclick="pick({k},\'keep\')">'
                f'<span class="k">k</span>a model — label its up axis</button>')
        else:
            buttons = "".join(
                f'<button data-ax="{a}" onclick="pick({k},\'{a}\')">'
                f'<span class="k">{i + 1}</span>{a}</button>'
                for i, a in enumerate(AX))
        note = (f'<div class="mine">proposed: {html.escape(mine)}</div>'
                if mine and not risky else "")
        cards.append(f"""<section class="m" id="m{k}">
 <div class="hd"><span class="n">{m['i']:03d}/{len(models)}</span>
 <span class="stem">{html.escape(m['stem'])}</span>{flag}</div>
 <img loading="lazy" src="{html.escape(sheets[m['stem']])}" alt="">
 <div class="row">{buttons}
 <button class="und" data-ax="undefined" onclick="pick({k},'undefined')">
 <span class="k">u</span>{'a loose part — no upright' if triage
                          else 'undefined'}</button></div>{note}
</section>""")

    # the pre-selection: everything the labeller was confident about, so the
    # human confirms rather than re-derives — but never on a flagged model,
    # where a default is the thing most likely to be waved through
    seed = {m["stem"]: proposals[m["stem"]].get("up")
            for m in models
            if m["stem"] in proposals
            and isinstance(proposals[m["stem"]], dict)
            and not proposals[m["stem"]].get("risky")
            and proposals[m["stem"]].get("up")}

    title = ("triage — is it a model?" if triage
             else f"up-axis labelling — {len(models)} models")
    keys = ("k keep &middot; u loose part &middot; n/p move" if triage
            else "1-6 pick &middot; u undefined &middot; n/p move")
    doc = f"""<!doctype html>
<meta charset="utf-8"><title>{title}</title>
<style>{PAGE_CSS}</style>
<header>
 <b>{'model or part?' if triage else 'up axis'}</b>
 <span class="count" id="count">0 of {len(models)} decided</span>
 <button onclick="exportJSON()">Export JSON</button>
 <span class="keys">{keys}</span>
</header>
<main>
{''.join(cards)}
<textarea id="out" placeholder="Export JSON writes confirmed.json — and drops \
the same text here, in case the browser blocks a download from file://"></textarea>
</main>
<script>
const AXES = {json.dumps(AX)};
const TRIAGE = {json.dumps(bool(triage))};
const MODELS = {json.dumps([{'stem': m['stem']} for m in models])};
{PAGE_JS}
// pre-seed unflagged proposals on first open only; a later visit keeps
// whatever the human actually pressed
if (!localStorage.getItem(store)) {{ picks = {json.dumps(seed)}; }}
MODELS.forEach((_, i) => paint(i));
save(); focus(0);
</script>"""
    page_dir.mkdir(parents=True, exist_ok=True)
    index = page_dir / ("triage.html" if triage else "index.html")
    index.write_text(doc)
    if stale:
        print(f"dropped {len(stale)} the vocabulary now cuts: "
              + ", ".join(sorted(stale)))
    print(f"wrote {index} over {len(models)} models\n"
          + ("open it, mark the loose parts, Export JSON — then the axis pass "
             "runs over what survives" if triage else
             "open it, pick, Export JSON, then --merge"))


def merge(out: Path, confirmed_file: Path, which: str) -> None:
    manifest = json.loads((out / "manifest.json").read_text())
    by_stem = {m["stem"]: m for m in manifest["models"]}
    raw = json.loads(confirmed_file.read_text())
    confirmed = raw.get("picks", raw)

    labels_file = REPO / "up_axis_labels.json"
    labels = json.loads(labels_file.read_text())
    have = {l["path"] for l in labels["labels"]}
    root = labels["collection_root"]

    added, skipped, undefined = [], [], 0
    for stem, up in confirmed.items():
        if up == "undefined":
            undefined += 1
            continue
        if up not in IDX:
            raise SystemExit(f"{stem}: {up!r} is not one of {', '.join(AX)}")
        m = by_stem.get(stem)
        if m is None:
            raise SystemExit(f"{stem} is not in this run's manifest")
        rel = str(Path("/" + m["path"]).relative_to(root))
        if rel in have:
            skipped.append(stem)
            continue
        added.append({"stem": stem, "set": which, "up": up, "path": rel})

    labels["labels"].extend(added)
    labels.setdefault("sets", {})[which] = (
        f"random draw from the collection walk, seed {manifest['seed']}, "
        f"excluding every path already labelled; proposed by a labeller "
        f"measured on the first 49 and confirmed by hand from the same "
        f"six-tile sheets. Method frozen as of this draw.")
    labels_file.write_text(json.dumps(labels, indent=1, ensure_ascii=False) + "\n")
    print(f"added {len(added)} labels to {labels_file} as set {which!r}\n"
          f"  {undefined} marked undefined and left out"
          + (f"\n  {len(skipped)} already present: {', '.join(skipped)}"
             if skipped else "")
          + f"\n  the set now holds {len(labels['labels'])} labels")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--render", action="store_true", help="sample and render sheets")
    ap.add_argument("--page", metavar="PROPOSALS.JSON", nargs="?", const="",
                    default=None, help="build the picking page")
    ap.add_argument("--merge", metavar="CONFIRMED.JSON", default=None,
                    help="write confirmed picks into up_axis_labels.json")
    ap.add_argument("--triage", action="store_true",
                    help="with --page: ask only model-or-part, over every "
                         "model the proposals file does not already cover")
    ap.add_argument("--page-dir", default=None,
                    help="where index.html goes (default: the first --out dir)")
    ap.add_argument("--set", dest="which", default="expand",
                    help="set name for merged labels (default: expand)")
    ap.add_argument("--walk", default=None, help="walk file (default: newest)")
    ap.add_argument("-n", type=int, default=170, help="models to sample")
    ap.add_argument("--seed", type=int, default=909)
    ap.add_argument("--render-px", type=int, default=RENDER_PX)
    ap.add_argument("--thumb", type=int, default=pose.SHEET_THUMB)
    ap.add_argument("--exclude", action="append", default=[], metavar="MANIFEST",
                    help="a previous batch's manifest.json, whose models this "
                         "draw must not re-draw (repeatable)")
    ap.add_argument("--out", action="append", default=[],
                    help="batch dir; repeat to build one page over several "
                         "(default OUT/label_expand)")
    args = ap.parse_args()

    dirs = [Path(o) for o in args.out] or [OUT / "label_expand"]
    out = dirs[0]
    if args.render:
        render(out, Path(args.walk) if args.walk else newest_walk(),
               args.n, args.seed, args.render_px, args.thumb,
               [Path(e) for e in args.exclude])
    elif args.page is not None:
        page(dirs, Path(args.page) if args.page else None,
             Path(args.page_dir) if args.page_dir else dirs[0], args.triage)
    elif args.merge:
        merge(out, Path(args.merge), args.which)
    else:
        ap.error("pass --render, --page or --merge")


if __name__ == "__main__":
    main()
