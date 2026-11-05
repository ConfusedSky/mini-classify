"""Ask Jev and gemini two extra questions per model: is its up axis ambiguous,
and is it a part of something larger. Measurement only; nothing in `src/` reads
the answers.

    .venv/bin/python eval/model_traits.py --limit 8     # smoke, prints the cost
    .venv/bin/python eval/model_traits.py               # Jev on all, gemini on GEMINI_SET
    .venv/bin/python eval/model_traits.py --report-only

Models: `arbiter_agreement.py`'s gated sample, plus the labelled models the
triage called whole, so the part question has hand truth on both sides. Every
triaged model in that sample is a part, because whole ones became labels and
the sample excludes labels. The ambiguity question has no hand truth at all.

Jev is asked twice, on the painted-out tiles alone and with the file name, to
show how much of its part answer is the name. Gemini sees the production sheet
and the name, so the like-for-like column is `jev+name`.
"""
import argparse, itertools, json, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from common import OUT, collection_root, load_labels, load_triage

import decisions_vlm, gemini_vlm
from part_judge import PART_FALSE, PART_TRUE
from pose_label_expand import LIKELY_PART
from src import pose

SAMPLE = OUT / "arbiter_agreement_1000_s0.json"
RESULT = OUT / "model_traits.json"
AMBIGUOUS_Q = "Is the up axis ambiguous for this model?"
PART_Q = "Is this a part of a larger model?"
AMBIGUOUS_TRUE = ("No one orientation is clearly the way up: it could reasonably "
                  "rest or be displayed several ways, like a loose weapon, a "
                  "limb or a shapeless rock.")
AMBIGUOUS_FALSE = ("One orientation is clearly the way up, like a standing "
                   "figure, a building or a vehicle on its wheels.")
SCHEMA = {"type": "object", "required": ["ambiguous", "part"],
          "properties": {"ambiguous": {"type": "boolean"}, "part": {"type": "boolean"}}}
PROMPT = ("These six tiles show one 3D-printable tabletop miniature or terrain "
          "piece, each in a different orientation. The file is named `{name}`.\n"
          f"ambiguous: {AMBIGUOUS_Q} True means: {AMBIGUOUS_TRUE} False means: "
          f"{AMBIGUOUS_FALSE}\npart: {PART_Q} True means: {PART_TRUE} False "
          f"means: {PART_FALSE}")


def load_items():
    root = collection_root()
    truth = {str(r["path"].relative_to(root)): r["part"] for r in load_triage()}
    items = []
    for it in json.loads(SAMPLE.read_text())["items"]:
        if it["gated"]:
            items.append({"stem": it["stem"], "rel": it["rel"], "group": "sample",
                          "disagree": it["gemini"] != it["fused"], "part_truth": truth.get(it["rel"])})
    for l in load_labels():
        rel = str(l["path"].relative_to(root))
        if truth.get(rel) is False:
            items.append({"stem": l["stem"], "rel": rel, "group": "labelled",
                          "disagree": None, "part_truth": False})
    return items


def gemini_set(it):
    """Gemini costs ~100x Jev per call, so it only sees the models an answer
    could change a decision on, or that have hand truth."""
    return bool(it["disagree"]) or it["part_truth"] is not None


def ask_jev(sheet, name, usage):
    refs = [decisions_vlm.ref() for _ in range(6)]
    state = {"renders": refs, "about": "Six renders of one 3D-printable tabletop "
             "miniature or terrain piece, each in a different orientation."}
    if name:
        state["file"] = name
    a = decisions_vlm.jev(dict(zip(refs, decisions_vlm.tiles(sheet, blank=True))), state, {
        "ambiguous": {"type": "noul", "instructions": AMBIGUOUS_Q,
                      "criteria": {"true": AMBIGUOUS_TRUE, "false": AMBIGUOUS_FALSE}},
        "part": {"type": "noul", "instructions": PART_Q,
                 "criteria": {"true": PART_TRUE, "false": PART_FALSE}}}, usage)
    return None if a is None else {k: a[k]["noul"] for k in ("ambiguous", "part")}


def ask_gemini(sheet, name, tok, usage):
    """A copy of `gemini_vlm.ask`'s request: that one parses a tile number."""
    import base64, urllib.error, urllib.request
    url = (f"https://{gemini_vlm.HOST}/v1/projects/{gemini_vlm.PROJECT}/locations/"
           f"{gemini_vlm.LOCATION}/publishers/google/models/{pose.GEMINI_MODEL}:generateContent")
    body = json.dumps({
        "contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": "image/png",
                            "data": base64.b64encode(sheet.read_bytes()).decode()}},
            {"text": PROMPT.format(name=name)}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                             "responseSchema": SCHEMA}}).encode()
    for attempt in range(6):
        try:
            t0 = time.time()
            req = urllib.request.Request(url, body, {"Authorization": f"Bearer {tok}",
                                                     "Content-Type": "application/json"})
            d = json.loads(urllib.request.urlopen(req, timeout=300).read())
            u = d.get("usageMetadata", {})
            usage.append({"secs": round(time.time() - t0, 2), "prompt": u.get("promptTokenCount", 0),
                          "output": u.get("candidatesTokenCount", 0),
                          "thoughts": u.get("thoughtsTokenCount", 0)})
            return json.loads("".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"]))
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 503):
                print(f"  gemini HTTP {e.code}: {e.read()[:200]}", flush=True)
                return None
            time.sleep(min(60, 4 * 2 ** attempt))
        except Exception as e:
            print(f"  gemini error: {e}", flush=True)
    return None


def costs(usage):
    p = gemini_vlm.PRICES[pose.GEMINI_MODEL]
    g = sum(x["prompt"] * p["in"] + (x["thoughts"] + x["output"]) * p["out"] for x in usage["gemini"]) / 1e6
    j = sum(x["prompt"] * 0.042 for x in usage["jev"]) / 1e6
    return g, j


def binary(rows, key, q, cut=0.5):
    """Hand-truth confusion for one answer column."""
    have = [r for r in rows if r.get(key) is not None]
    val = lambda r: r[key][q] >= cut
    tp = sum(val(r) and r["part_truth"] for r in have)
    fp = sum(val(r) and not r["part_truth"] for r in have)
    pos = sum(r["part_truth"] for r in have)
    return f"{tp}/{pos} parts caught, {fp}/{len(have) - pos} whole models called parts"


def report(d):
    items = d["items"]
    print("part question against hand truth:")
    truth = [r for r in items if r["part_truth"] is not None]
    for k in ("jev", "jev_name", "gemini"):
        print(f"  {k:9} {binary(truth, k, 'part')}")
    sample = [r for r in items if r["group"] == "sample"]
    print(f"\nshare answering yes, on {len(sample)} gated sample models:")
    for k in ("jev", "jev_name", "gemini"):
        have = [r for r in sample if r.get(k)]
        for q in ("ambiguous", "part"):
            for name, sel in (("disagree", [r for r in have if r["disagree"]]),
                              ("agree", [r for r in have if not r["disagree"]])):
                if sel:
                    print(f"  {k:9} {q:9} {name:8} {sum(r[k][q] >= .5 for r in sel):3}/{len(sel)}")
    both = [r for r in items if r.get("jev_name") and r.get("gemini")]
    print(f"\nJev+name against gemini, {len(both)} models:")
    for q in ("ambiguous", "part"):
        print(f"  {q:9} agree {sum((r['jev_name'][q] >= .5) == bool(r['gemini'][q]) for r in both)}/{len(both)}")
    print(f"\nname regex alone on the same hand truth: "
          f"{sum(bool(LIKELY_PART.search(Path(r['rel']).stem)) and r['part_truth'] for r in truth)}"
          f"/{sum(r['part_truth'] for r in truth)} parts, "
          f"{sum(bool(LIKELY_PART.search(Path(r['rel']).stem)) and not r['part_truth'] for r in truth)} whole")
    dis = [r for r in sample if r["disagree"] and r.get("jev_name")]
    keep = [r for r in dis if r["jev_name"]["part"] < .5 and r["jev_name"]["ambiguous"] < .5]
    print(f"\ndisagreements Jev+name calls neither part nor ambiguous: {len(keep)}/{len(dis)}")
    g, j = costs(d["usage"])
    for k, c in (("gemini", g), ("jev", j)):
        u = d["usage"][k]
        if u:
            print(f"{k}: {len(u)} calls, ${c:.2f}, median {np.median([x['secs'] for x in u]):.1f}s"
                  + (f", mean thoughts {np.mean([x['thoughts'] for x in u]):.0f}" if k == "gemini" else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=0, help="first N of each group, for a smoke run")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    out = RESULT.with_name("model_traits_smoke.json") if args.limit else RESULT
    if args.report_only:
        return report(json.loads(out.read_text()))

    items = load_items()
    if args.limit:
        items = ([r for r in items if r["group"] == "sample" and r["disagree"]][:args.limit]
                 + [r for r in items if r["group"] == "labelled"][:args.limit])
    print(f"{len(items)} models, gemini on {sum(map(gemini_set, items))}")
    tok, usage = gemini_vlm.token(), {"gemini": [], "jev": []}
    t0, done, lock = time.time(), itertools.count(1), threading.Lock()

    def one(it):
        sheet, name = OUT / "sheets512" / f"{it['stem']}.png", Path(it["rel"]).stem
        it["jev"] = ask_jev(sheet, None, usage["jev"])
        it["jev_name"] = ask_jev(sheet, name, usage["jev"])
        if gemini_set(it):
            it["gemini"] = ask_gemini(sheet, name, tok, usage["gemini"])
        with lock:
            print(f"  [{next(done):>3}/{len(items)}] {name[:40]:40} jev {it['jev']} "
                  f"+name {it['jev_name']} gemini {it.get('gemini')} {time.time()-t0:5.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(one, items))
    out.write_text(json.dumps({"items": items, "usage": usage}, indent=1))
    report(json.loads(out.read_text()))


if __name__ == "__main__":
    main()
