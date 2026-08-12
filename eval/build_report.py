"""Build the up-axis failure report as a self-contained HTML page.

Every verdict is derived from the score dumps, never typed in — geometry's
unseeded sampler moves picks between runs, so hardcoded values go stale.
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from common import OUT, AX, IDX, load_labels, mark, score

D = OUT / "siglip_up"
OUT = D.parent / "up_axis_failures.html"
AX = ["+Z", "-Z", "+Y", "-Y", "+X", "-X"]
IDX = {a: i for i, a in enumerate(AX)}
T = 256

SCORES = {r["stem"]: r for r in json.load(open(D / "results.json"))}
SCORES.update(json.load(open(D / "targets.json")))

GOLD = {l["stem"]: l["gold"] for l in load_labels()}  # from up_axis_labels.json
mm = lambda v: (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else np.zeros_like(v)


def picks(stem):
    """(geometry, object_generic, upright_toppled, ensemble) as axis strings."""
    r = SCORES[stem]
    g = np.array(r["geo"]["scores"])
    og = np.array(r["siglip"]["object_generic"]["scores"])
    ut = np.array(r["siglip"]["upright_toppled"]["scores"])
    return (AX[int(g.argmax())], AX[int(og.argmax())], AX[int(ut.argmax())],
            AX[int((mm(g) + mm(og)).argmax())])


def headline_rates():
    res = json.load(open(D / "results.json"))
    rows = [(i, r) for i, r in enumerate(res) if i in GOLD]
    out = {}
    for name, fn in (
        ("geometry", lambda r: np.array(r["geo"]["scores"])),
        ("object_generic", lambda r: np.array(r["siglip"]["object_generic"]["scores"])),
        ("upright_toppled", lambda r: np.array(r["siglip"]["upright_toppled"]["scores"])),
        ("ensemble", lambda r: mm(np.array(r["geo"]["scores"]))
                               + mm(np.array(r["siglip"]["object_generic"]["scores"]))),
    ):
        out[name] = sum(int(fn(r).argmax() == IDX[GOLD[i]]) for i, r in rows)
    return out, len(rows)


def tile_uri(sheet, axis):
    """One 256px candidate tile, cropped clear of the red index numeral."""
    i = IDX[axis]
    im = Image.open(D / sheet).convert("RGB")
    x, y = (i % 3) * T, (i // 3) * T
    im = im.crop((x, y + 20, x + T, y + T))
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# stem, sheet, truth, note
MODELS = [
    ("Right axis, wrong sign",
     "The vertical axis is identified correctly and only up-versus-down is inverted. Every "
     "model here tapers &mdash; wide at the top, narrow at the bottom &mdash; so upside down it "
     "still reads as a solid object resting on something. On an untextured render there is no "
     "gravity cue, no ground contact and no texture to break the tie.", [
        ("tile9", "002_tile9.png", "+Y", "The socket recess ends up underneath."),
        ("Grass Tufts Tuner (3)", "038_Grass Tufts Tuner (3).png", "+Y",
         "Blades hang down instead of splaying up; inverted it reads as a claw or tripod."),
        ("Propane_Tank", "018_Propane_Tank.png", "+Y", "Stood on its valve."),
        ("Bunker_MiniV2_Roof_", "005_Bunker_MiniV2_Roof_.png", "+Z",
         "Picks the view into the hollow underside."),
        ("PitFiend_Bust", "TARGET_PitFiend_Bust.png", "+Z",
         "A bust is a wedge &mdash; wide at the head, narrow at the plinth. The margin between "
         "right and upside down was 0.0066."),
     ]),
    ("Flat slabs stood on edge",
     "A rectangular slab presents a large convincing silhouette in any orientation, so "
     "&ldquo;resting stably on the ground&rdquo; cannot separate lying flat from standing on "
     "a long edge.", [
        ("Floor", "025_Floor.png", "-Z",
         "Rears a floor panel up like a wall. Both probe sets agree on the wrong answer."),
        ("Concrete Chunk (6)", "019_Concrete Chunk (6).png", "+Z",
         "Stands a piece of rubble on edge."),
        ("32mm_Gate_L", "034_32mm_Gate_L.png", "+Y", "Lays the gate down flat."),
     ]),
    ("Tipped onto the back",
     "Machined props with a large flat rear panel: that panel is a more convincing "
     "&ldquo;base&rdquo; than the actual footprint.", [
        ("Bedienkonsole", "026_Bedienkonsole.png", "+Z",
         "Every method agrees the console lies on its back &mdash; the one model in the "
         "labelled set that no combination recovers."),
        ("Body", "024_Body.png", "+Y",
         "Only upright_toppled fails, standing the console on its end."),
     ]),
]

COUNTER = ("32mm_PitFiend", "TARGET_32mm_PitFiend.png", "+Z",
           "No print base at all: geometry scores 0.0045 with a near-total tie between its top "
           "two candidates, and lays the demon on its side. Both probe sets recover the upright.")


def cell(sheet, axis, truth, label):
    ok = axis == truth
    return (f'<div class="cell {"ok" if ok else "bad"}">'
            f'<img src="{tile_uri(sheet, axis)}" alt="candidate orientation, up {axis}">'
            f'<div class="cap"><span class="ax">{axis.replace("-", "&minus;")}</span>'
            f'<span class="who">{label}</span></div></div>')


def card(stem, sheet, truth, note, extra_class=""):
    geo, og, ut, ens = picks(stem)
    methods = (("geometry", geo), ("object_generic", og),
               ("upright_toppled", ut), ("ensemble", ens))
    cells = [cell(sheet, truth, truth, "ground truth")]
    grouped = {}
    for name, p in methods[1:]:            # geometry's tile is not shown; its chip carries it
        grouped.setdefault(p, []).append(name)
    for p, names in grouped.items():
        if p != truth:
            cells.append(cell(sheet, p, truth, " + ".join(names)))
    verdicts = "".join(
        f'<span class="v {"ok" if p == truth else "bad"}'
        f'{" lead" if n == "ensemble" else ""}">{n} {p.replace("-", "&minus;")}</span>'
        for n, p in methods)
    if ens == truth:
        wrong = [n for n, p in methods[:3] if p != truth]
        verdict = (f'<p class="also rescue">The average recovers this: '
                   f'{", ".join(wrong)} wrong alone, ensemble correct.</p>') if wrong else ""
    else:
        verdict = '<p class="also miss">The average does not recover this one.</p>'
    return f"""      <article class="card{extra_class}">
        <header class="card-head">
          <h3>{stem}</h3>
          <div class="verdicts">{verdicts}</div>
        </header>
        <div class="tiles">{"".join(cells)}</div>
        <p class="note">{note}</p>{verdict}
      </article>"""


rates, n = headline_rates()
sections = []
for title, blurb, models in MODELS:
    cards = "\n".join(card(*m) for m in models)
    sections.append(f"""    <section class="group">
      <h2>{title}</h2>
      <p class="lede">{blurb}</p>
{cards}
    </section>""")

HTML = f"""<title>Up-axis detection: where the SigLIP probes fail</title>
<style>
  :root {{
    --paper:#EDEFF2; --card:#FFFFFF; --ink:#14181D; --ink-2:#4A535E;
    --rule:#D3D8DE; --ok:#0E6E63; --bad:#A93B27; --ok-bg:#E4F0EE; --bad-bg:#F7E6E2;
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
    --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --paper:#101316; --card:#191D22; --ink:#E6E9EC; --ink-2:#98A2AE;
      --rule:#2A3037; --ok:#4FBFAE; --bad:#E0705A; --ok-bg:#14302C; --bad-bg:#331914;
    }}
  }}
  :root[data-theme="dark"] {{
    --paper:#101316; --card:#191D22; --ink:#E6E9EC; --ink-2:#98A2AE;
    --rule:#2A3037; --ok:#4FBFAE; --bad:#E0705A; --ok-bg:#14302C; --bad-bg:#331914;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; background:var(--paper); color:var(--ink);
    font-family:var(--sans); font-size:15.5px; line-height:1.65;
    -webkit-font-smoothing:antialiased;
  }}
  .wrap {{ max-width:1040px; margin:0 auto; padding:56px 24px 96px; }}
  .eyebrow {{
    font-family:var(--mono); font-size:11px; letter-spacing:.14em;
    text-transform:uppercase; color:var(--ink-2); margin:0 0 14px;
  }}
  h1 {{
    font-family:var(--mono); font-size:clamp(25px,3.4vw,34px); font-weight:600;
    letter-spacing:-.015em; line-height:1.18; margin:0 0 18px; text-wrap:balance;
  }}
  .standfirst {{ max-width:68ch; color:var(--ink-2); margin:0 0 34px; font-size:16.5px; }}
  .standfirst strong {{ color:var(--ink); font-weight:600; }}
  .scores {{
    display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:1px;
    background:var(--rule); border:1px solid var(--rule); border-radius:3px;
    overflow:hidden; margin:0 0 14px;
  }}
  .score {{ background:var(--card); padding:16px 18px; }}
  .score.hero dt {{ color:var(--ok); }}
  .score dt {{
    font-family:var(--mono); font-size:11px; letter-spacing:.1em; text-transform:uppercase;
    color:var(--ink-2); margin:0 0 8px;
  }}
  .score dd {{
    margin:0; font-family:var(--mono); font-size:22px; font-weight:600;
    font-variant-numeric:tabular-nums;
  }}
  .score.hero dd {{ color:var(--ok); }}
  .score dd small {{ font-size:12px; font-weight:400; color:var(--ink-2); margin-left:7px; }}
  .caveat {{
    max-width:68ch; font-size:14px; color:var(--ink-2); margin:0 0 52px;
    padding-left:14px; border-left:2px solid var(--rule);
  }}
  .group {{ margin:0 0 12px; }}
  h2 {{
    font-family:var(--mono); font-size:13px; font-weight:600; letter-spacing:.1em;
    text-transform:uppercase; color:var(--ink); margin:44px 0 0;
    padding-bottom:10px; border-bottom:1px solid var(--rule);
  }}
  .lede {{ max-width:68ch; color:var(--ink-2); margin:16px 0 26px; }}
  .card {{
    background:var(--card); border:1px solid var(--rule); border-radius:3px;
    padding:18px 20px 20px; margin:0 0 16px;
  }}
  .card-head {{
    display:flex; flex-wrap:wrap; align-items:baseline; gap:10px 16px; margin-bottom:16px;
  }}
  .card h3 {{ font-family:var(--mono); font-size:15px; font-weight:600; margin:0; }}
  .verdicts {{ display:flex; flex-wrap:wrap; gap:6px; }}
  .v {{
    font-family:var(--mono); font-size:11px; letter-spacing:.03em; padding:3px 8px;
    border-radius:2px; white-space:nowrap;
  }}
  .v.ok  {{ color:var(--ok);  background:var(--ok-bg); }}
  .v.bad {{ color:var(--bad); background:var(--bad-bg); }}
  .v.lead {{ box-shadow:inset 0 0 0 1px currentColor; }}
  .tiles {{ display:flex; gap:14px; overflow-x:auto; padding-bottom:4px; }}
  .cell {{ flex:0 0 210px; }}
  .cell img {{
    display:block; width:100%; height:auto; border-radius:2px;
    border:1px solid var(--rule); border-top-width:3px;
  }}
  .cell.ok  img {{ border-top-color:var(--ok); }}
  .cell.bad img {{ border-top-color:var(--bad); }}
  .cap {{ display:flex; align-items:baseline; gap:8px; margin-top:8px; }}
  .ax {{ font-family:var(--mono); font-size:14px; font-weight:600;
         font-variant-numeric:tabular-nums; }}
  .cell.ok  .ax {{ color:var(--ok); }}
  .cell.bad .ax {{ color:var(--bad); }}
  .who {{ font-family:var(--mono); font-size:10.5px; letter-spacing:.05em;
          text-transform:uppercase; color:var(--ink-2); }}
  .note {{ max-width:68ch; margin:16px 0 0; font-size:14.5px; }}
  .also {{ max-width:68ch; margin:8px 0 0; font-size:13.5px; font-family:var(--mono);
           font-size:12px; letter-spacing:.02em; }}
  .rescue {{ color:var(--ok); }}
  .miss {{ color:var(--bad); }}
  .counter {{ border-color:var(--ok); }}
  footer {{
    margin-top:54px; padding-top:20px; border-top:1px solid var(--rule);
    font-size:13.5px; color:var(--ink-2); max-width:68ch;
  }}
  footer code {{ font-family:var(--mono); font-size:12.5px; }}
  @media (max-width:560px) {{ .cell {{ flex-basis:180px; }} }}
</style>

<div class="wrap">
  <p class="eyebrow">Pose pipeline &middot; up-axis detection</p>
  <h1>Where the SigLIP orientation probes fail</h1>
  <p class="standfirst">Every failure of both probe sets is a <strong>non-figure</strong>. On
  characters and creatures they were perfect; all the errors below are terrain, scatter and
  props. Each card shows the true upright beside the orientation each method chose &mdash;
  including the <strong>ensemble</strong>, a plain mean of the two after min&ndash;max
  normalising each, which recovers all but one of them.</p>

  <dl class="scores">
    <div class="score"><dt>geometry (current)</dt><dd>{rates['geometry']}&thinsp;/&thinsp;{n}</dd></div>
    <div class="score"><dt>upright_toppled</dt><dd>{rates['upright_toppled']}&thinsp;/&thinsp;{n}</dd></div>
    <div class="score"><dt>object_generic</dt><dd>{rates['object_generic']}&thinsp;/&thinsp;{n}</dd></div>
    <div class="score hero"><dt>ensemble (mean)</dt><dd>{rates['ensemble']}&thinsp;/&thinsp;{n} <small>ceiling</small></dd></div>
  </dl>
  <p class="caveat">Ground truth is {n} hand-labelled models from a 40-model random sample; 17 were
  excluded because their upright is genuinely undefined &mdash; a moustache, a gate pin, a flat gear
  disc, a dragon in flight. <strong>These rates are optimistic.</strong> object_generic was written
  after watching earlier probes fail on some of these same models, and the ensemble is the best of
  six normalisation schemes measured against the same labels, with only 8 disagreements to
  arbitrate. The failure modes are the trustworthy part; the rates need a clean holdout.
  Geometry is also not reproducible run to run &mdash; its point sampler is unseeded, and
  <code>Propane_Tank</code> and <code>32mm_PitFiend</code> both changed pick between two runs on
  identical input.</p>

{chr(10).join(sections)}

    <section class="group">
      <h2>Counterpoint &mdash; where the probes earn their place</h2>
      <p class="lede">The same taxonomy read forwards. This is the population that motivated the
      experiment: models with no print base, where geometry has nothing to measure.</p>
{card(*COUNTER, extra_class=" counter")}
    </section>

  <footer>Renders are the six axis-aligned up candidates from
  <code>render_up_candidate_tiles</code>, scored by SigLIP&nbsp;2 so400m against upright/toppled
  text probes and picked by argmax. Geometry is <code>detect_up_axis</code> &mdash; flat
  down-facing surface in the bottom 2% height slab &mdash; averaged over three draws to suppress
  its sampling noise. The ensemble min&ndash;max normalises each score vector per model and takes
  the mean; because geometry's weakest candidate is almost always 0, that normalisation preserves
  its <code>runner-up / best</code> ratio, so geometry votes hardest exactly when it has real base
  evidence and abstains when it is guessing.</footer>
</div>
"""

OUT.write_text(HTML)
print(f"wrote {OUT}  ({len(HTML)/1024:.0f} KB)")
print("rates:", rates, f"n={n}")
