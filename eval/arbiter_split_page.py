"""Blind labelling page for the gemini-against-fused-Jev disagreements that
Jev calls neither a part nor ambiguous.

    .venv/bin/python eval/arbiter_split_page.py
    .venv/bin/python eval/arbiter_split_page.py --score    # after labelling

Writes `eval/out/arbiter_split.html` beside `sheets512/`. The page carries no
arbiter answer and a shuffled order, so a label cannot lean on either one. The
cutoffs select on Jev's answers, so the set is a sample of nothing: never pool it.
"""
import argparse, html, json, random
from pathlib import Path
from urllib.parse import quote

from common import ARBITER_SPLIT_FILE, AX, OUT
from pose_label_expand import PAGE_CSS, PAGE_JS

PART_CUT, AMBIGUOUS_CUT = 0.35, 0.75
RUN = "arbiter-split"


def score():
    picks = json.loads(ARBITER_SPLIT_FILE.read_text())["picks"]
    runs = {i["rel"]: i for i in json.loads((OUT / "arbiter_agreement_1000_s0.json").read_text())["items"]}
    traits = {i["rel"]: i["gemini"] for i in json.loads((OUT / "model_traits.json").read_text())["items"]
              if i.get("gemini")}
    axes = {r: a for r, a in picks.items() if a != "undefined"}
    print(f"{len(picks)} labelled: {len(axes)} with an upright, {len(picks) - len(axes)} without")
    for k in ("gemini", "fused", "jev", "ens"):
        print(f"  {k:6} right on {sum(AX[runs[r][k]] == a for r, a in axes.items())}/{len(axes)}")
    flagged = lambda r: traits[r]["part"] or traits[r]["ambiguous"]
    print(f"gemini flags part or ambiguous: {sum(flagged(r) for r in picks if r not in axes)}"
          f"/{len(picks) - len(axes)} without an upright, {sum(map(flagged, axes))}/{len(axes)} with one")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--score", action="store_true")
    if ap.parse_args().score:
        return score()
    items = json.loads((OUT / "model_traits.json").read_text())["items"]
    models = sorted((r for r in items if r["disagree"]
                     and r["jev_name"]["part"] < PART_CUT
                     and r["jev_name"]["ambiguous"] < AMBIGUOUS_CUT), key=lambda r: r["rel"])
    random.Random(0).shuffle(models)
    cards = []
    for k, m in enumerate(models):
        assert (OUT / "sheets512" / f"{m['stem']}.png").exists(), m["stem"]
        buttons = "".join(f'<button data-ax="{a}" onclick="pick({k},\'{a}\')"><span class="k">{i + 1}'
                          f'</span>tile {i + 1}</button>' for i, a in enumerate(AX))
        cards.append(f"""<section class="m" id="m{k}">
 <div class="hd"><span class="n">{k + 1:02d}/{len(models)}</span>
 <span class="stem">{html.escape(Path(m['rel']).stem)}</span></div>
 <img src="sheets512/{quote(m['stem'])}.png" alt="">
 <div class="row">{buttons}
 <button class="und" data-ax="undefined" onclick="pick({k},'undefined')">
 <span class="k">u</span>no upright</button></div>
</section>""")
    keymap = {str(i + 1): a for i, a in enumerate(AX)}
    doc = f"""<!doctype html>
<meta charset="utf-8"><title>arbiter split: {len(models)} models</title>
<style>{PAGE_CSS}</style>
<header>
 <b>which tile is upright?</b>
 <span class="count" id="count">0 of {len(models)} decided</span>
 <button onclick="exportJSON()">Export JSON</button>
 <span class="keys">1-6 tile &middot; u no upright &middot; n/p move</span>
</header>
<main>
{''.join(cards)}
<textarea id="out" placeholder="Export JSON downloads the answers and drops the same text here, in case the browser blocks a download from file://"></textarea>
</main>
<script>
const KEYMAP = {json.dumps(keymap)};
const RUN = {json.dumps(RUN)};
const MODELS = {json.dumps([{'stem': Path(m['rel']).stem, 'path': m['rel']} for m in models])};
{PAGE_JS}
MODELS.forEach((_, i) => paint(i));
save(); focus(0);
</script>"""
    path = OUT / "arbiter_split.html"
    path.write_text(doc)
    print(f"wrote {path} over {len(models)} models")


if __name__ == "__main__":
    main()
