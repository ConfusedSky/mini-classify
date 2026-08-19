"""Render every hand-labelled model in its labelled up orientation.

The labels in `up_axis_labels.json` assert "this axis is up". This renders that
assertion: each mesh is rotated with `rotation_to_z_up(label)` — the same call
the pipeline makes — and photographed from a few azimuths, so a wrong label
shows up as a model lying on its side.

    python eval/gold_upright.py            # render, then write the HTML
    python eval/gold_upright.py --html     # rebuild the HTML from what exists

Output goes to OUT/gold_upright/ (gitignored, override with EVAL_OUT).
"""
import argparse
import base64
import html

from common import AX, OUT, load_labels

# `Renderer.views` orbits a full ring — `view_angles(n, [elev])` is
# 360/n-spaced — so three views *are* these azimuths. Named here because the
# filenames carry them.
AZIMUTHS = [0.0, 120.0, 240.0]
ELEVATION = 20.0
RENDER_PX = 768      # rendered big, saved small: antialiases the thin bits
SAVE_PX = 384
JPEG_QUALITY = 82

DIR = OUT / "gold_upright"


def render(labels):
    """One JPEG per (model, azimuth), skipping pairs already on disk."""
    import numpy as np
    from PIL import Image
    import rig
    from src import pose as P
    from src.pose import view_angles

    DIR.mkdir(parents=True, exist_ok=True)
    paths = {l["stem"]: [DIR / f"{l['stem']}_az{int(a)}.jpg" for a in AZIMUTHS]
             for l in labels}
    todo = [l for l in labels if any(not p.exists() for p in paths[l["stem"]])]
    if not todo:
        return paths

    print(f"rendering {len(todo)} models x {len(AZIMUTHS)} views -> {DIR}")
    r = rig.rig(RENDER_PX, views=len(AZIMUTHS), elevations=(ELEVATION,))
    # the rotation the pipeline makes: `Renderer.views` rotates a copy of the
    # mesh by rotation_to_z_up(up), which is exactly what this page asserts
    assert [round(a, 6) for a, _ in view_angles(len(AZIMUTHS), [ELEVATION])] \
        == [round(float(np.deg2rad(a)), 6) for a in AZIMUTHS]
    for n, l in enumerate(todo, 1):
        images = rig.views(r, rig.load(l["path"]), P.UP_CANDIDATES[l["gold"]])
        for p, arr in zip(paths[l["stem"]], images):
            im = Image.fromarray(arr)
            im.thumbnail((SAVE_PX, SAVE_PX))
            im.convert("RGB").save(p, "JPEG", quality=JPEG_QUALITY)
        print(f"  [{n}/{len(todo)}] {l['stem']} ({l['up']})", flush=True)
    return paths


def data_uri(path):
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def build_html(labels, paths, out_file):
    """One card per model: the three shots, the stem, the axis being asserted."""
    import json
    from common import LABELS_FILE
    notes = json.loads(LABELS_FILE.read_text()).get("sets", {})
    sections = []
    # sets in first-appearance order, so a set added to the labels file shows up
    # without editing this
    for s in dict.fromkeys(l["set"] for l in labels):
        rows = sorted((l for l in labels if l["set"] == s),
                      key=lambda l: (l["up"], l["stem"].lower()))
        if not rows:
            continue
        cards = []
        for l in rows:
            shots = "".join(
                f'<div class="shot"><img src="{data_uri(p)}" loading="lazy"'
                f' alt="{html.escape(l["stem"])} seen from azimuth {int(a)} degrees">'
                f'<span class="az">{int(a)}&deg;</span></div>'
                for p, a in zip(paths[l["stem"]], AZIMUTHS) if p.exists())
            cards.append(
                f'<figure class="card" data-up="{l["up"]}" data-set="{l["set"]}"'
                f' data-stem="{html.escape(l["stem"])}">\n'
                f'  <div class="strip">{shots}</div>\n'
                f'  <figcaption>\n'
                f'    <span class="stem" title="{html.escape(str(l["path"]))}">{html.escape(l["stem"])}</span>\n'
                f'    <span class="axis">{l["up"]}</span>\n'
                f'    <button class="flag" type="button" aria-pressed="false"'
                f' aria-label="Flag {html.escape(l["stem"])} as wrong">flag</button>\n'
                f'  </figcaption>\n'
                f'</figure>')
        note = notes.get(s, "")
        sections.append(
            f'<section data-set="{s}">\n'
            f'  <h2>{s}<span class="n">{len(rows)} models</span></h2>\n'
            f'  <p class="setnote">{html.escape(note)}</p>\n'
            f'  <div class="grid">\n' + "\n".join(cards) + '\n  </div>\n</section>')

    def chips(keys, counts):
        return "\n    ".join(
            f'<button data-f="{k}" aria-pressed="false">{k}'
            f'<span class="n">{counts[k]}</span></button>'
            for k in keys if counts[k])

    axis_n = {a: sum(l["up"] == a for l in labels) for a in AX}
    set_keys = list(dict.fromkeys(l["set"] for l in labels))
    set_n = {s: sum(l["set"] == s for l in labels) for s in set_keys}
    out_file.write_text(PAGE.replace("{{SECTIONS}}", "\n".join(sections))
                            .replace("{{AXIS_CHIPS}}", chips(AX, axis_n))
                            .replace("{{SET_CHIPS}}", chips(set_keys, set_n))
                            .replace("{{N}}", str(len(labels)))
                            .replace("{{VIEWS}}", str(len(AZIMUTHS)))
                            .replace("{{ELEV}}", str(int(ELEVATION))))
    print(f"wrote {out_file}  ({out_file.stat().st_size / 1e6:.1f} MB)")


PAGE = """<title>Gold Up-Axis Check</title>
<style>
/* Board and chrome theme; the cards keep a fixed light interior in both
   themes so every render sits on the same white it was rendered against. */
:root {
  --board: #e9ecf0;
  --board-2: #f4f6f8;
  --ink: #131820;
  --muted: #5c6673;
  --rule: #cdd4dc;
  --accent: #26418f;
  --accent-wash: #e2e7f6;
  --signal: #b5410c;
  --signal-wash: #fbe6da;
  --sheet: #ffffff;
  --sheet-ink: #141820;
  --sheet-muted: #6d7682;
  --sheet-rule: #dfe3e9;
  /* the --sheet-* set is deliberately NOT redefined per theme: the card
     interior stays the white the models were rendered against */
  --sheet-accent: #26418f;
  --sheet-accent-wash: #e2e7f6;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --board: #14181f;
    --board-2: #1b212b;
    --ink: #e8ecf3;
    --muted: #939eae;
    --rule: #2c3542;
    --accent: #93aaff;
    --accent-wash: #222c46;
    --signal: #ff9257;
    --signal-wash: #3a2013;
  }
}
:root[data-theme="dark"] {
  --board: #14181f;
  --board-2: #1b212b;
  --ink: #e8ecf3;
  --muted: #939eae;
  --rule: #2c3542;
  --accent: #93aaff;
  --accent-wash: #222c46;
  --signal: #ff9257;
  --signal-wash: #3a2013;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--board);
  color: var(--ink);
  font: 15px/1.55 var(--sans);
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1560px; margin: 0 auto; padding: 0 clamp(16px, 3vw, 40px) 96px; }

/* ---- masthead ---- */
header { padding: 44px 0 22px; }
.eyebrow {
  font: 500 11px/1 var(--mono);
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 0 0 14px;
}
h1 {
  font: 600 clamp(24px, 3.4vw, 34px)/1.1 var(--mono);
  letter-spacing: -0.02em;
  margin: 0 0 14px;
  text-wrap: balance;
}
.lede { color: var(--muted); max-width: 68ch; margin: 0; }
.lede code {
  font: 0.88em var(--mono);
  color: var(--ink);
  background: var(--board-2);
  border: 1px solid var(--rule);
  border-radius: 3px;
  padding: 1px 5px;
}

/* ---- utility bar ---- */
.bar {
  position: sticky;
  top: 0;
  z-index: 5;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  padding: 12px 0;
  margin-bottom: 26px;
  background: var(--board);
  border-bottom: 1px solid var(--rule);
}
.bar .sep { width: 1px; align-self: stretch; background: var(--rule); margin: 2px 4px; }
button {
  font: 500 12px/1 var(--mono);
  color: var(--ink);
  background: var(--board-2);
  border: 1px solid var(--rule);
  border-radius: 3px;
  padding: 7px 11px;
  cursor: pointer;
  display: inline-flex;
  gap: 7px;
  align-items: center;
}
button:hover { border-color: var(--accent); }
button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.bar button[aria-pressed="true"] {
  color: var(--accent);
  background: var(--accent-wash);
  border-color: var(--accent);
}
button .n {
  font-size: 11px;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.bar button[aria-pressed="true"] .n { color: var(--accent); }
.tally {
  margin-left: auto;
  font: 12px/1 var(--mono);
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}
#copy[data-state="done"] { color: var(--accent); border-color: var(--accent); }

/* ---- sections ---- */
section { margin-bottom: 40px; }
section.empty { display: none; }
h2 {
  display: flex;
  align-items: baseline;
  gap: 12px;
  font: 500 12px/1 var(--mono);
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 0 0 14px;
  padding-bottom: 9px;
  border-bottom: 1px solid var(--rule);
}
h2 .n { font-size: 11px; letter-spacing: 0.08em; font-variant-numeric: tabular-nums; }
.setnote {
  color: var(--muted);
  font-size: 13px;
  max-width: 78ch;
  margin: -4px 0 16px;
}

.grid {
  display: grid;
  gap: 16px;
  grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
}

/* ---- card ---- */
.card {
  margin: 0;
  background: var(--sheet);
  border: 1px solid var(--sheet-rule);
  border-radius: 4px;
  overflow: hidden;
  box-shadow: 0 1px 2px rgb(0 0 0 / 0.10);
}
.card.hide { display: none; }
.card[data-flagged="true"] { outline: 2px solid var(--signal); outline-offset: -2px; }
.strip { display: grid; grid-template-columns: repeat(3, 1fr); }
.shot { position: relative; }
.shot + .shot { border-left: 1px solid var(--sheet-rule); }
.shot img { width: 100%; display: block; aspect-ratio: 1; object-fit: contain; }
.az {
  position: absolute;
  left: 5px;
  bottom: 4px;
  font: 10px/1 var(--mono);
  color: var(--sheet-muted);
  font-variant-numeric: tabular-nums;
}
figcaption {
  display: flex;
  gap: 10px;
  align-items: center;
  padding: 8px 10px;
  border-top: 1px solid var(--sheet-rule);
  color: var(--sheet-ink);
}
.stem {
  flex: 1;
  min-width: 0;
  font: 12px/1.35 var(--mono);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.axis {
  flex: none;
  font: 600 12px/1 var(--mono);
  color: var(--sheet-accent);
  background: var(--sheet-accent-wash);
  border-radius: 3px;
  padding: 5px 7px;
}
.flag {
  flex: none;
  font: 500 11px/1 var(--mono);
  letter-spacing: 0.06em;
  color: var(--sheet-muted);
  background: transparent;
  border-color: var(--sheet-rule);
  padding: 5px 8px;
}
.flag:hover { color: var(--signal); border-color: var(--signal); }
.flag[aria-pressed="true"] {
  color: var(--signal);
  background: var(--signal-wash);
  border-color: var(--signal);
}
@media (max-width: 420px) { .grid { grid-template-columns: 1fr; } }
</style>

<div class="wrap">
<header>
  <p class="eyebrow">up_axis_labels.json &middot; ground truth</p>
  <h1>Gold up-axis check</h1>
  <p class="lede">All {{N}} hand-labelled models, each rotated so the axis it was
  labelled with points up &mdash; <code>rotation_to_z_up(label)</code>, the same call the
  pipeline makes. {{VIEWS}} azimuths at {{ELEV}}&deg; elevation. Every model here should
  read as standing upright; a mislabel shows up lying on its side or upside&nbsp;down.
  Flag the ones that look wrong and copy the list out.</p>
</header>

<div class="bar">
  <button data-f="all" aria-pressed="true">all<span class="n">{{N}}</span></button>
  <span class="sep"></span>
  {{SET_CHIPS}}
  <span class="sep"></span>
  {{AXIS_CHIPS}}
  <span class="sep"></span>
  <button data-f="flagged" aria-pressed="false">flagged<span class="n" id="fn">0</span></button>
  <button id="copy" type="button">copy flagged</button>
  <span class="tally" id="tally"></span>
</div>

{{SECTIONS}}
</div>

<script>
const cards = [...document.querySelectorAll('.card')];
const sections = [...document.querySelectorAll('section')];
const filters = [...document.querySelectorAll('.bar button[data-f]')];
const tally = document.getElementById('tally');
const fn = document.getElementById('fn');
const copy = document.getElementById('copy');
let active = 'all';

const flagged = () => cards.filter(c => c.dataset.flagged === 'true');

function apply() {
  filters.forEach(b => b.setAttribute('aria-pressed', String(b.dataset.f === active)));
  cards.forEach(c => {
    const show = active === 'all'
      || (active === 'flagged'
          ? c.dataset.flagged === 'true'
          : c.dataset.up === active || c.dataset.set === active);
    c.classList.toggle('hide', !show);
  });
  sections.forEach(s => s.classList.toggle('empty',
    ![...s.querySelectorAll('.card')].some(c => !c.classList.contains('hide'))));
  const shown = cards.filter(c => !c.classList.contains('hide')).length;
  tally.textContent = shown + ' of ' + cards.length + ' shown';
  fn.textContent = flagged().length;
}

filters.forEach(b => b.addEventListener('click', () => { active = b.dataset.f; apply(); }));

document.querySelectorAll('.flag').forEach(btn => btn.addEventListener('click', () => {
  const card = btn.closest('.card');
  const on = card.dataset.flagged !== 'true';
  card.dataset.flagged = String(on);
  btn.setAttribute('aria-pressed', String(on));
  btn.textContent = on ? 'flagged' : 'flag';
  apply();
}));

copy.addEventListener('click', async () => {
  const list = flagged().map(c => c.dataset.stem).join('\\n');
  try { await navigator.clipboard.writeText(list); } catch (e) { /* clipboard blocked */ }
  copy.textContent = list ? 'copied ' + flagged().length : 'none flagged';
  copy.dataset.state = 'done';
  setTimeout(() => { copy.textContent = 'copy flagged'; copy.dataset.state = ''; }, 1600);
});

apply();
</script>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", action="store_true", help="skip rendering; rebuild the page")
    ap.add_argument("-o", "--out", default=str(DIR / "gold_upright.html"))
    a = ap.parse_args()

    import pathlib
    labels = load_labels()
    if a.html:
        paths = {l["stem"]: [DIR / f"{l['stem']}_az{int(z)}.jpg" for z in AZIMUTHS]
                 for l in labels}
    else:
        paths = render(labels)
    build_html(labels, paths, pathlib.Path(a.out))
    if not a.html:                 # a renderer is live; teardown would abort
        import rig
        rig.exit_without_teardown()
