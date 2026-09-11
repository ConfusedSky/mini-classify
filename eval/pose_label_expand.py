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
  lands in `labels/up_axis_labels.json` is what the *human* pressed.
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
import re
from pathlib import Path
from urllib.parse import quote

from common import AX, IDX, LABELS_FILE, OUT, REPO, load_labels

from src import naming, pose

RENDER_PX = 384

# A left/right marker, or a numbered "part N": the two name shapes that mean
# "this is a piece of something", measured against a 247-model hand triage on
# 2026-09-09. Between them they name 41% of what a person called a loose part
# (51 of 123) and **none** of what they called a model (0 of 124) — the only
# two signals tested that were free on that side. A weapon word costs 4 of 124
# and a body-part word costs 8, because `Body`, `Head` and `Torso` are what DM
# Stash and Artisan Guild call the *main sculpt*: `32_Unsupported_Tygrin_Body`
# is the character, not a piece of one.
#
# **This filters the labelling draw, not the collection.** It is deliberately
# not in `src.naming`: a paired weapon or a spell effect is a real thing
# someone searches for, and unlike a sprue or a `75_` twin it has no surviving
# sibling that carries it. It only means "unlikely to hold still for an up
# axis", which is a statement about ground truth, not about the library.
# 0 of 124 is not 0% — at that sample size the true rate could be ~3% — so
# `--all-shapes` draws without it when a run wants an unfiltered sample.
LIKELY_PART = re.compile(r"(?:^|[_\s(])[LR](?:$|[_\s)])|part[_\s-]?\d",
                         re.IGNORECASE)

# The second part signal, and the only one that is not in the name: a file
# whose stem ends in a bare capital letter *and* whose own directory holds
# siblings lettered the same way — `32mm_OdDracnesThrone_B` beside A, C and D,
# `32mm_DragonSkullManor_C` beside seven more. Those letters enumerate the
# components of one object, where `_L`/`_R` enumerate a left and a right.
#
# Measured against 247 models triaged by hand, scored against the labels that
# survived review: it catches 16% of parts on its own and **15 that the name
# alone misses** — nearly all modular terrain (Manor, Cave, Igloo, Nest,
# BunkerDoor) — for 0 of 119 models wrongly cut. Together with LIKELY_PART:
# 52% of parts, still 0 models. 0 of 119 is not 0%; at that sample the true
# rate could be ~2.5%.
#
# It survives the obvious objection — modular *character* kits letter their
# variants too (`Dragonpeak_Barbarian_B`) — only because this library gives
# each such variant its own directory, so the siblings are not beside it. That
# is a fact about this collection, not about naming, and a library that packed
# variants together would need this measured again.
LETTERED = re.compile(r"^(.*?)[_ ]([A-Z])$")


def lettered_family(stem: str, folder, index, n_min: int = 2) -> bool:
    """Is `stem` a lettered member of a family of `n_min`+ in its own folder?"""
    m = LETTERED.match(stem)
    if not m:
        return False
    sibs = re.compile(re.escape(m.group(1)) + r"[_ ][A-Z]$")
    return sum(bool(sibs.match(s)) for s in index.get(folder, ())) >= n_min


def folder_index(files):
    """{directory: [stem, ...]} for `lettered_family`."""
    index: dict = {}
    for f in files:
        index.setdefault(f.parent, []).append(f.stem)
    return index


def newest_walk(repo=REPO):
    """The most recently scanned walk under any embed-cache*/."""
    walks = sorted(repo.glob("embed-cache*/walk-*.json"),
                   key=lambda p: json.loads(p.read_text()).get("scanned", 0))
    if not walks:
        raise SystemExit(f"no embed-cache*/walk-*.json under {repo} — pass --walk")
    return walks[-1]


def render(out: Path, walk: Path, n: int, seed: int, render_px: int,
           thumb: int, exclude: list[Path] | None = None,
           skip_parts: bool = True) -> None:
    from PIL import Image
    import rig

    labelled = {str(Path(l["path"]).resolve()) for l in load_labels()}
    # A top-up batch must also skip what an earlier batch already drew — those
    # models are not labelled *yet*, so `load_labels` cannot see them, and a
    # second draw that re-renders them widens nothing (the same reason the
    # first draw skips the labelled set).
    for m in exclude or []:
        # `path` in a manifest is anchor-relative ("run/media/..."), so it is
        # rooted with the anchor, not with collection_root — joining it onto
        # the root built "/run/media/.../run/media/..." and matched nothing,
        # which is how batch 2 re-drew six of batch 1's models (2026-09-09).
        labelled |= {str(Path("/" + e["path"]).resolve())
                     for e in json.loads(m.read_text())["models"]}
    files = [Path(p) for p in json.loads(walk.read_text())["files"]]
    # A walk file is a record of what the vocabulary said *then*. Re-apply it
    # here, because a draw is for the collection as it is defined now — the
    # `sprue` cut (2026-09-09) landed after this walk was written, and without
    # this the sample offers models the next walk will not index.
    files = [f for f in files if not any(naming.skip(part) for part in f.parts)]
    fresh = [f for f in files if str(f.resolve()) not in labelled]
    drawn_from = len(fresh)
    if skip_parts:
        index = folder_index(files)
        fresh = [f for f in fresh
                 if not LIKELY_PART.search(f.stem)
                 and not lettered_family(f.stem, f.parent, index)]
    if len(fresh) < n:
        raise SystemExit(f"walk holds {len(fresh)} unlabelled files, asked for {n}")
    sample = random.Random(seed).sample(fresh, n)

    out.mkdir(parents=True, exist_ok=True)
    print(f"walk {walk}: {len(files)} files, {len(files) - drawn_from} already "
          f"labelled or cut\n"
          + (f"skipping {drawn_from - len(fresh)} likely parts (--all-shapes "
             f"keeps them)\n" if skip_parts else "")
          + f"sampling {n} of the remaining {len(fresh)}, seed {seed}\n"
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
        {"walk": str(walk), "seed": seed, "part_filter": bool(skip_parts),
         "axes": AX, "render_px": render_px,
         "thumb": thumb, "collection_root": _root_of(sample[0]),
         "models": rows}, indent=1))
    print(f"\nwrote {len(rows)} sheets and manifest.json to {out}")
    rig.exit_without_teardown()


def _root_of(sample_path: Path) -> str:
    """The mount the sample came from, recorded so the set can be relocated —
    `labels/up_axis_labels.json`'s own `collection_root` convention."""
    labels = json.loads(LABELS_FILE.read_text())
    return labels["collection_root"]


# The page asks one of three questions, and the answer vocabulary is the only
# thing that differs. `parts` exists because the axis benchmark was scored on a
# population production never faces: ground truth excluded parts by convention
# while the pipeline poses everything, so a headline accuracy described the
# easy half of the collection (2026-09-10). A part with a defensible upright —
# a backpack, a torso, a wall panel — is pose ground truth and belongs in the
# benchmark; only the genuinely undefined ones are skip-list ground truth.
CHOICES = {
    "axis": [(str(i + 1), a, a) for i, a in enumerate(AX)],
    "triage": [("k", "keep", "a model — label its up axis")],
    "parts": [("a", "accessory", "accessory — worn or carried"),
              ("c", "component", "component — part of one assembly")],
}
UNDEFINED_TEXT = {
    "axis": "undefined",
    "triage": "a loose part — no upright",
    "parts": "neither — not a part at all",
}

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
.note { margin:10px 16px 0; padding:10px 12px; border-radius:6px;
        background:#fff8e1; border:1px solid #e6d28a; color:#4a3c00;
        font-size:13px; line-height:1.5; white-space:pre-wrap; }
"""

PAGE_JS = """
const KEYS = {"1":0,"2":1,"3":2,"4":3,"5":4,"6":5};
// Keyed by *which models this page asks about*, not by its URL: successive
// pages are written to the same path, so a plain pathname key merged one
// batch's answers into the next one's page (seen live: a 67-model triage
// reporting "299 of 67 decided"). RUN is derived from the batch dirs.
const store = "poselabels:" + location.pathname + ":" + RUN;
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

// A stem is a display name, not an identity: "tail1", "head", "tile7" and
// "Gnomish_Miner_Backpack" each name different files in different folders.
// Keying answers by stem gave two distinct models one shared answer — 161
// models on the parts page produced 154 answers, and `merge` could only
// refuse them afterwards. The answer is keyed by path; the stem is shown.
function paint(i) {
  const m = MODELS[i], el = document.getElementById("m" + i);
  const got = picks[m.path];
  el.classList.toggle("done", got !== undefined);
  el.querySelectorAll("button[data-ax]").forEach(b => {
    b.classList.toggle("sel", b.dataset.ax === got);
  });
}

function pick(i, ax) {
  picks[MODELS[i].path] = ax;
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
  if (e.key in KEYMAP) { pick(cur, KEYMAP[e.key]); e.preventDefault(); }
  else if (e.key === "u") { pick(cur, "undefined"); e.preventDefault(); }
  else if (e.key === "n" || e.key === "j" || e.key === "ArrowDown") {
    focus(Math.min(cur + 1, MODELS.length - 1)); e.preventDefault();
  } else if (e.key === "p" || e.key === "k" || e.key === "ArrowUp") {
    focus(Math.max(cur - 1, 0)); e.preventDefault();
  }
});

function exportJSON() {
  const out = JSON.stringify({run: RUN, picks: picks}, null, 1);
  document.getElementById("out").value = out;
  try {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([out], {type: "application/json"}));
    // Name the file after the question it answers. Every page used to
    // download "confirmed.json", so four passes produced four files with one
    // name and the reader renamed them by hand — and a class export and an
    // axis export are indistinguishable until something opens them.
    a.download = "confirmed-" + RUN.replace(/[:,]/g, "-") + ".json";
    a.click();
  } catch (e) {}
}
"""


def page(dirs: list[Path], proposals_file: Path | None, page_dir: Path,
         mode: str = "axis", decided_file: Path | None = None,
         skip_parts: bool = False, note: str = "") -> None:
    """One page over one or more rendered batches.

    Two modes, because the two questions cost their own reader very different
    amounts. **Triage** asks only *is this a model or a loose part?* — a
    glance for a person, and the answer a labeller cannot skip past, since
    roughly 40% of this collection is hands, blades and wings whose upright is
    undefined by convention (`part_probe.py`, LEARNINGS 2026-09-03). **Pick**
    asks for the axis, over the survivors, with a proposal pre-selected.

    Splitting them is what keeps the expensive pass small: a proposal is only
    worth making for a model that can carry a label at all."""
    models, sheets, stale, seen, prefiltered = [], {}, [], set(), []
    part_index = None
    if skip_parts:
        walks = {json.loads((d / "manifest.json").read_text())["walk"] for d in dirs}
        part_index = folder_index(
            Path(p) for w in walks
            for p in json.loads(Path(w).read_text())["files"])
    for d in dirs:
        manifest = json.loads((d / "manifest.json").read_text())
        for m in manifest["models"]:
            # the same file drawn twice (see the exclude bug above) is one
            # model, not two questions
            if m["path"] in seen:
                continue
            seen.add(m["path"])
            # rendered before a vocabulary change that has since cut it: asking
            # for a label here would grow the set with a model the walk drops
            if any(naming.skip(part) for part in Path(m["path"]).parts):
                stale.append(m["stem"])
                continue
            # a part signal added after this batch was drawn (the lettered
            # family, 2026-09-09) still applies to it: asking a human to
            # triage what a measured rule already answers is wasted attention
            if part_index is not None:
                folder = Path("/" + m["path"]).parent
                if (LIKELY_PART.search(m["stem"])
                        or lettered_family(m["stem"], folder, part_index)):
                    prefiltered.append(m["stem"])
                    continue
            models.append(m)
            # percent-encode, then HTML-escape: these filenames carry spaces,
            # apostrophes, ampersands and parentheses, and a bare `#` in one
            # would otherwise cut the URL short at a fragment
            home = Path(m["sheet_dir"]) if m.get("sheet_dir") else d
            rel = os.path.relpath(home / m["sheet"], page_dir)
            sheets[m["path"]] = quote(rel)
    # A settled triage answer retires the model from the axis pass: a loose
    # part has no axis to pick, and re-asking is the one thing guaranteed to
    # make a second pass feel like the first one over again.
    # already in the labels file: answered, and answered by a human. The
    # exclude bug (see `render`) let two of batch 3 through as re-draws.
    known = json.loads(LABELS_FILE.read_text())
    root, done = known["collection_root"], {l["path"] for l in known["labels"]}
    already = {m["path"] for m in models
               if str(Path("/" + m["path"]).relative_to(root)) in done}
    if already:
        names = sorted(m["stem"] for m in models if m["path"] in already)
        models = [m for m in models if m["path"] not in already]
        print(f"dropped {len(already)} already labelled: " + ", ".join(names))
    settled = {}
    if decided_file:
        raw = json.loads(decided_file.read_text())
        settled = raw.get("picks", raw)
    if settled and mode != "triage":
        # keyed by path, with the stem read as a fallback for the two exports
        # written before the page keyed on path
        models = [m for m in models
                  if (settled.get(m["path"]) or settled.get(m["stem"], "keep"))
                  != "undefined"]
    proposals = {}
    if proposals_file:
        raw = json.loads(proposals_file.read_text())
        proposals = raw.get("proposals", raw)
    # A *proposal* is not an answer. Only a human decision retires a model
    # from triage, and that arrives through `--decided`; earlier this skipped
    # anything with a proposal, which emptied the page the moment the labeller
    # had judged every model — exactly when the page is most needed.

    cards = []
    for k, m in enumerate(models):
        p = proposals.get(m["path"], proposals.get(m["stem"], {}))
        if isinstance(p, str):
            p = {"up": p}
        mine, risky = p.get("up"), bool(p.get("risky"))
        why = p.get("why", "")
        flag = (f'<span class="flag">look closely — {html.escape(why)}</span>'
                if risky else "")
        buttons = "".join(
            f'<button data-ax="{val}" onclick="pick({k},\'{val}\')">'
            f'<span class="k">{key}</span>{text}</button>'
            for key, val, text in CHOICES[mode])
        bits = []
        if mine and not risky:
            bits.append("proposed: " + html.escape(
                ("a loose part" if mine == "undefined" else "a model")
                if mode == "triage" else mine))
        if why and not risky:
            bits.append(html.escape(why))
        if p.get("from"):
            bits.append("<b>" + html.escape(p["from"]) + "</b>")
        note = f'<div class="mine">{" &middot; ".join(bits)}</div>' if bits else ""
        cards.append(f"""<section class="m" id="m{k}">
 <div class="hd"><span class="n">{m['i']:03d}/{len(models)}</span>
 <span class="stem">{html.escape(m['stem'])}</span>{flag}</div>
 <img loading="lazy" src="{html.escape(sheets[m['path']])}" alt="">
 <div class="row">{buttons}
 <button class="und" data-ax="undefined" onclick="pick({k},'undefined')">
 <span class="k">u</span>{UNDEFINED_TEXT[mode]}</button></div>{note}
</section>""")

    # the pre-selection: everything the labeller was confident about, so the
    # human confirms rather than re-derives — but never on a flagged model,
    # where a default is the thing most likely to be waved through
    # In triage the question is model-or-part, so a proposed *axis* seeds
    # "keep" and only "undefined" seeds the part button. A flagged proposal
    # seeds nothing either way — that is what the flag is for.
    seed = {}
    for m in models:
        pr = proposals.get(m["path"], proposals.get(m["stem"]))
        if not isinstance(pr, dict) or pr.get("risky") or not pr.get("up"):
            continue
        seed[m["path"]] = (("undefined" if pr["up"] == "undefined" else "keep")
                           if mode == "triage" else pr["up"])

    # identity of this page's question: the batches it covers, and the mode
    run_id = mode + ":" + ",".join(sorted(d.name for d in dirs))
    title = ("triage — is it a model?" if mode == "triage"
             else f"up-axis labelling — {len(models)} models")
    keys = " &middot; ".join([f"{k} {t.split(' — ')[0]}" for k, _v, t in CHOICES[mode]]
                             + [f"u {UNDEFINED_TEXT[mode]}", "n/p move"])
    doc = f"""<!doctype html>
<meta charset="utf-8"><title>{title}</title>
<style>{PAGE_CSS}</style>
<header>
 <b>{ {'triage': 'model or part?', 'parts': 'which kind of part?',
        'axis': 'up axis'}[mode] }</b>
 <span class="count" id="count">0 of {len(models)} decided</span>
 <button onclick="exportJSON()">Export JSON</button>
 <span class="keys">{keys}</span>
</header>
{f'<div class="note">{html.escape(note)}</div>' if note else ''}
<main>
{''.join(cards)}
<textarea id="out" placeholder="Export JSON writes confirmed.json — and drops \
the same text here, in case the browser blocks a download from file://"></textarea>
</main>
<script>
const AXES = {json.dumps(AX)};
const MODE = {json.dumps(mode)};
const KEYMAP = {json.dumps({k: v for k, v, _t in CHOICES[mode]})};
const RUN = {json.dumps(run_id)};
const MODELS = {json.dumps([{'stem': m['stem'], 'path': m['path']} for m in models])};
{PAGE_JS}
// Seed a proposal only where the human has not already decided that model.
// Per-model rather than "first open only", so a page rebuilt with more
// proposals mid-pass seeds the new ones without touching a single answer
// already given — which is what lets the labeller and the reader work at
// the same time.
for (const [path, up] of Object.entries({json.dumps(seed)})) {{
  if (!(path in picks)) picks[path] = up;
}}
MODELS.forEach((_, i) => paint(i));
save(); focus(0);
</script>"""
    page_dir.mkdir(parents=True, exist_ok=True)
    index = page_dir / {"triage": "triage.html", "parts": "parts.html",
                        "axis": "index.html"}[mode]
    index.write_text(doc)
    if stale:
        print(f"dropped {len(stale)} the vocabulary now cuts: "
              + ", ".join(sorted(stale)))
    if prefiltered:
        print(f"dropped {len(prefiltered)} the part signals answer without "
              f"asking: " + ", ".join(sorted(prefiltered)[:6])
              + (" ..." if len(prefiltered) > 6 else ""))
    print(f"wrote {index} over {len(models)} models\n"
          + ("open it, mark the loose parts, Export JSON — then the axis pass "
             "runs over what survives" if mode == "triage" else
             "open it, sort them, Export JSON" if mode == "parts" else
             "open it, pick, Export JSON, then --merge"))


def merge(dirs: list[Path], confirmed_file: Path, which: str) -> None:
    """Write confirmed picks into `labels/up_axis_labels.json`.

    Takes every batch the page covered, not one: a page built over several
    draws exports one file, and looking its picks up in a single manifest
    would silently drop the models that came from the others."""
    manifests = [json.loads((d / "manifest.json").read_text()) for d in dirs]
    by_stem: dict[str, list] = {}
    by_path: dict[str, list] = {}
    for manifest in manifests:
        for m in manifest["models"]:
            by_stem.setdefault(m["stem"], []).append(m)
            by_path.setdefault(m["path"], []).append(m)
    raw = json.loads(confirmed_file.read_text())
    confirmed = raw.get("picks", raw)

    labels_file = LABELS_FILE
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
        # Pages key their answers by path (a stem is a display name, not an
        # identity: "tail1", "head" and "tile7" each name different files in
        # different directories). Stem keys are still read, for the files two
        # pages exported before that fix — but only where the stem is
        # unambiguous in this draw, because there is otherwise no way to tell
        # which file the human answered.
        hits = by_path.get(stem) or by_stem.get(stem, [])
        if not hits:
            raise SystemExit(f"{stem} is not in this run's manifest")
        paths = {h["path"] for h in hits}
        if len(paths) > 1:
            raise SystemExit(
                f"{stem!r} names {len(paths)} different files in this draw; "
                f"a pick keyed by stem cannot say which:\n  "
                + "\n  ".join(sorted(paths)))
        m = hits[0]
        rel = str(Path("/" + m["path"]).relative_to(root))
        if rel in have:
            skipped.append(stem)
            continue
        # the manifest's stem, never the confirmed file's key: that key is a
        # path now, and writing it here put a full path in every `stem` field
        # — which `common.build_tiles` then used as a *filename*.
        added.append({"stem": m["stem"], "set": which, "up": up, "path": rel})

    labels["labels"].extend(added)
    seeds = ", ".join(str(m["seed"]) for m in manifests)
    filtered = all(m.get("part_filter") for m in manifests)
    # A manifest that was not built by `render` says what it is instead. The
    # default sentence claims a random draw at a seed, and a set assembled some
    # other way — reclaimed from an earlier triage, hand-picked — would inherit
    # that claim and read as a sample of the collection when it is not. The
    # sets are pooled by other harnesses on exactly that basis.
    notes = [m["note"] for m in manifests if m.get("note")]
    labels.setdefault("sets", {})[which] = " ".join(notes) if notes else (
        f"random draw from the collection walk, seed {seeds}, "
        f"excluding every path already labelled"
        + (", and every name carrying a left/right marker or a numbered part "
           "— so this set is a sample of the collection's *whole* models, not "
           "of the collection" if filtered else "")
        + f"; proposed by a labeller measured on the first 49 and confirmed by "
        f"hand from the same six-tile sheets. Method frozen as of this draw.")
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
                    help="write confirmed picks into labels/up_axis_labels.json")
    ap.add_argument("--mode", default="axis",
                    choices=sorted(CHOICES), help=
                    "with --page: which question to ask — `axis` (the six "
                    "candidates), `triage` (model or loose part), or `parts` "
                    "(accessory or assembly component)")
    ap.add_argument("--decided", default=None, metavar="TRIAGE.JSON",
                    help="with --page: a triage export; models answered "
                         "\"undefined\" there are dropped from the axis pass")
    ap.add_argument("--note", default="", help=
                    "with --page: a banner above the cards. A pass that asks "
                    "the standing question differently — a convention, a "
                    "changed meaning for `u` — has to say so where the reader "
                    "is looking.")
    ap.add_argument("--page-dir", default=None,
                    help="where index.html goes (default: the first --out dir)")
    ap.add_argument("--set", dest="which", default="expand",
                    help="set name for merged labels (default: expand)")
    ap.add_argument("--walk", default=None, help="walk file (default: newest)")
    ap.add_argument("-n", type=int, default=170, help="models to sample")
    ap.add_argument("--seed", type=int, default=909)
    ap.add_argument("--all-shapes", action="store_true",
                    help="draw without the likely-part filter — an unfiltered "
                         "random sample of the collection")
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
               [Path(e) for e in args.exclude], not args.all_shapes)
    elif args.page is not None:
        page(dirs, Path(args.page) if args.page else None,
             Path(args.page_dir) if args.page_dir else dirs[0], args.mode,
             Path(args.decided) if args.decided else None, not args.all_shapes,
             args.note)
    elif args.merge:
        merge(dirs, Path(args.merge), args.which)
    else:
        ap.error("pass --render, --page or --merge")


if __name__ == "__main__":
    main()
