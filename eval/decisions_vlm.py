"""Run the pose-arbiter question through a decision model — OpenAI's Decisions
API (gpt-6-luna) or TypeSafe's Jev — on the same hand-labelled models and
cached contact sheets as eval/gemini_vlm.py, and print the same tables — so
the column is directly comparable with the published gemini/glm numbers.

Question shapes:
  choice      Luna, one call per model: a `choice` over the sheet's tile numerals.
  predicate   Luna, one call per tile: "is this upright?" on the tile cropped
              out of the sheet; the answer is the most probable tile.
  jev-sheet   Jev, the sheet in `state` and a Choice over its tile numerals.
  jev-noul    Jev, one request per model holding six Nouls, one cropped tile
              in each question's instructions; the answer is the most probable.
  jev-tiles   Jev, a Choice whose six options are the cropped tiles themselves,
              so no numeral has to be read.
  jev-tiles-clean  jev-tiles with the red numerals painted out of each tile.
Each model's confidence is kept, and a calibration table shows whether gating
the arbiter on it would help.

  .venv/bin/python eval/decisions_vlm.py                       # choice @512
  .venv/bin/python eval/decisions_vlm.py --mode predicate --thumb 256,512
  .venv/bin/python eval/decisions_vlm.py --mode jev-tiles --baselines out/pose_baseline.json
  .venv/bin/python eval/decisions_vlm.py --mode predicate --report-only

Auth: Luna needs a Platform API key in ~/.config/openai/key (OPENAI_KEY_FILE
overrides); a ChatGPT/Codex login is not accepted by api.openai.com. Jev needs
a TypeSafe key in ~/.config/typesafe/key (TYPESAFE_KEY_FILE overrides): as of
2026-10-20 OpenRouter accepts Jev image requests but drops the `assets`, so
the model answers without the image. Jev rejects images under 280 px a side,
so its tile modes need sheet 512 or larger.
"""
import argparse, base64, io, itertools, json, os, threading, time, uuid
import urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import AX, OUT, build_sheets, load_baselines, load_labels  # REPO on sys.path

import gemini_vlm                       # report(), cost_report(), PRICES table

URL = "https://api.openai.com/v1/decisions"
MODEL = "gpt-6-luna"
KEY = Path(os.environ.get("OPENAI_KEY_FILE", Path.home() / ".config/openai/key"))
JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-1.14.0"
JEV_KEY = Path(os.environ.get("TYPESAFE_KEY_FILE", Path.home() / ".config/typesafe/key"))
TAGS = {"choice": "luna-dec", "predicate": "luna-pred",
        "jev-sheet": "jev-sheet", "jev-noul": "jev-noul", "jev-tiles": "jev-tiles",
        "jev-tiles-clean": "jev-tiles-clean"}
# Input-only billing: OpenAI Decisions guide as read 2026-10-10, TypeSafe
# models page as read 2026-10-20.
for _m, _t in TAGS.items():
    gemini_vlm.PRICES[_t] = {"in": 0.042 if _m.startswith("jev") else 0.10, "out": 0}

CHOICE_INSTRUCTIONS = (
    "Each numbered tile shows the same 3D model in a different orientation. "
    "Which tile shows the model standing upright, the way it would sit on a table?")
PREDICATE_INSTRUCTIONS = (
    "Is this 3D model shown standing upright, the way it would sit on a table? "
    "Ignore the red number in the corner.")


def png_url(png):
    return "data:image/png;base64," + base64.b64encode(png).decode()


def decide(png, question, usage):
    """One /v1/decisions call on one image. The answer dict, or None."""
    body = json.dumps({
        "model": MODEL,
        "input": [{"role": "user", "content": [{"type": "input_image", "image_url": png_url(png)}]}],
        "questions": [question],
    }).encode()
    req = urllib.request.Request(URL, body, {"Authorization": f"Bearer {KEY.read_text().strip()}",
                                             "Content-Type": "application/json"})
    for attempt in range(6):
        t0 = time.time()
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=120).read())
        except urllib.error.HTTPError as e:
            msg = e.read()[:200]
            if e.code in (429, 500, 502, 503):
                print(f"  HTTP {e.code} (attempt {attempt+1}): {msg}", flush=True)
                time.sleep(min(60, 4 * 2 ** attempt)); continue
            print(f"  HTTP {e.code}: {msg}")
            return None
        except Exception as e:
            print(f"  error: {e}"); time.sleep(4); continue
        usage.append({"prompt": d["usage"]["input_tokens"], "thoughts": 0, "output": 0,
                      "secs": time.time() - t0, "attempts": attempt + 1})
        return d["answers"][0]
    return None


def tiles(sheet, n_tiles=6, cols=3, blank=False):
    """The sheet's tiles as PNG bytes, in candidate order. `blank` paints the
    red numeral out with the background, so a tile carries only the render."""
    from PIL import Image
    im = Image.open(sheet).convert("RGB")
    w = im.width // cols
    out = []
    for i in range(n_tiles):
        t = im.crop(((i % cols) * w, (i // cols) * w, (i % cols + 1) * w, (i // cols + 1) * w))
        if blank:
            bg, px = t.getpixel((1, 1)), t.load()
            for x in range(w // 8):              # the numeral sits in this corner (pose.make_contact_sheet)
                for y in range(w // 8):
                    r, g, b = px[x, y]
                    if r - max(g, b) > 6:       # anti-aliased edges too; renders are grey
                        px[x, y] = bg
        buf = io.BytesIO()
        t.save(buf, "PNG")
        out.append(buf.getvalue())
    return out


def ask_choice(sheet, usage, n_tiles=6):
    """(0-based candidate, confidence, per-candidate probabilities) or Nones."""
    a = decide(sheet.read_bytes(), {
        "type": "choice", "name": "tile", "instructions": CHOICE_INSTRUCTIONS,
        "choices": [{"value": str(i), "description": f"Tile {i}"} for i in range(1, n_tiles + 1)]},
        usage)
    if a is None or a["type"] != "choice":     # a refusal is an answer, not a retry
        return None, None, None
    p = {x["value"]: x["probability"] for x in a["probabilities"]}
    return int(a["choice"]) - 1, a["confidence"], [p[str(i)] for i in range(1, n_tiles + 1)]


def ask_predicate(sheet, usage, n_tiles=6):
    """Argmax over per-tile upright probabilities; confidence is that maximum."""
    q = {"type": "predicate", "name": "upright", "instructions": PREDICATE_INSTRUCTIONS}
    ps, calls = [], []
    for png in tiles(sheet, n_tiles):
        a = decide(png, q, calls)
        if a is None or a["type"] != "predicate":
            return None, None, None
        ps.append(a["probability"])
    # cost_report prices per arbiter call, and one arbitration here is n_tiles requests
    usage.append({k: sum(c[k] for c in calls) for k in calls[0]})
    best = max(range(n_tiles), key=ps.__getitem__)
    return best, ps[best], ps


def jev(images, state, questions, usage):
    """One Jev request. `images` maps each reference used in `state` or
    `questions` to PNG bytes. The answers dict, or None."""
    body = json.dumps({
        "model": JEV_MODEL, "state": state, "questions": questions,
        "assets": {ref: {"type": "image", "format": "png",
                         "data_base64": base64.b64encode(png).decode()}
                   for ref, png in images.items()},
    }).encode()
    req = urllib.request.Request(JEV_URL, body, {
        "Authorization": f"Bearer {JEV_KEY.read_text().strip()}",
        "Content-Type": "application/json", "TypeSafe-Danger-Zone-Images": "true"})
    for attempt in range(6):
        t0 = time.time()
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=120).read())
        except urllib.error.HTTPError as e:
            msg = e.read()[:200]
            if e.code in (429, 500, 502, 503, 504):
                print(f"  HTTP {e.code} (attempt {attempt+1}): {msg}", flush=True)
                time.sleep(min(60, 4 * 2 ** attempt)); continue
            print(f"  HTTP {e.code}: {msg}")
            return None
        except Exception as e:
            print(f"  error: {e}"); time.sleep(4); continue
        usage.append({"prompt": d["usage"]["input_tokens"], "thoughts": 0, "output": 0,
                      "secs": time.time() - t0, "attempts": attempt + 1})
        return d["answers"]
    return None


def ref():
    # References match as plain text anywhere in the request, so they must not
    # be able to occur in prose by accident (TypeSafe images docs).
    return f"img-{uuid.uuid4()}"


def from_choice(a, n_tiles):
    p = a["probabilities"]
    return int(a["choice"]) - 1, a["confidence"], [p[str(i)] for i in range(1, n_tiles + 1)]


def ask_jev_sheet(sheet, usage, n_tiles=6):
    r = ref()
    a = jev({r: sheet.read_bytes()}, {"sheet": r}, {"tile": {
        "type": "choice",
        "instructions": "`sheet` shows six numbered tiles of the same 3D model in "
                        "different orientations. Which tile shows the model standing "
                        "upright, the way it would sit on a table?",
        "criteria": {str(i): f"Tile {i}" for i in range(1, n_tiles + 1)}}}, usage)
    return (None, None, None) if a is None else from_choice(a["tile"], n_tiles)


def ask_jev_noul(sheet, usage, n_tiles=6):
    """Six Nouls in one request; they are evaluated independently."""
    refs = [ref() for _ in range(n_tiles)]
    a = jev(dict(zip(refs, tiles(sheet, n_tiles))), "A render of a 3D-printable "
            "tabletop miniature or terrain piece.", {
        f"t{i}": {"type": "noul",
                  "instructions": f"Is the 3D model in this render standing upright, "
                                  f"the way it would sit on a table? {r} "
                                  f"Ignore the red number in the corner."}
        for i, r in enumerate(refs)}, usage)
    if a is None:
        return None, None, None
    ps = [a[f"t{i}"]["noul"] for i in range(n_tiles)]
    best = max(range(n_tiles), key=ps.__getitem__)
    return best, ps[best], ps


def ask_jev_tiles(sheet, usage, n_tiles=6, blank=False):
    refs = [ref() for _ in range(n_tiles)]
    a = jev(dict(zip(refs, tiles(sheet, n_tiles, blank=blank))), "Six renders of the "
            "same 3D-printable tabletop miniature or terrain piece, each in a different "
            "orientation.", {"tile": {
                "type": "choice",
                "instructions": "Which render shows the model standing upright, the "
                                "way it would sit on a table?"
                                + ("" if blank else " Ignore the red numbers."),
                "criteria": {str(i + 1): r for i, r in enumerate(refs)}}}, usage)
    return (None, None, None) if a is None else from_choice(a["tile"], n_tiles)


ASK = {"choice": ask_choice, "predicate": ask_predicate, "jev-sheet": ask_jev_sheet,
       "jev-noul": ask_jev_noul, "jev-tiles": ask_jev_tiles,
       "jev-tiles-clean": lambda s, u: ask_jev_tiles(s, u, blank=True)}


def calibration(items, key):
    """Does confidence separate right answers from wrong ones, and would gating
    the arbiter on it beat always trusting it?"""
    rows = [it for it in items if it.get(key) is not None]
    print(f"\n{key}: confidence vs correctness")
    right = sorted(it[f"{key}_conf"] for it in rows if it[key] == it["up"])
    wrong = sorted(it[f"{key}_conf"] for it in rows if it[key] != it["up"])
    med = lambda s: s[len(s) // 2] if s else float("nan")
    print(f"  right n={len(right)} median {med(right):.2f} | wrong n={len(wrong)} median {med(wrong):.2f}")
    # AUROC: chance a random right answer outranks a random wrong one
    if right and wrong:
        auc = sum((r > w) + 0.5 * (r == w) for r in right for w in wrong) / (len(right) * len(wrong))
        print(f"  AUROC {auc:.2f}  (0.5 = confidence says nothing)")
    print(f"  {'gate':>6} {'override':>9} {'rescued':>8} {'broke':>6} {'pipeline':>9}")
    for t in [0, .3, .5, .7, .8, .9, .95]:
        res = brk = n = 0
        for it in rows:
            if it["arb"] and it[f"{key}_conf"] >= t:
                n += 1
                res += it[key] == it["up"] and it["ens"] != it["up"]
                brk += it[key] != it["up"] and it["ens"] == it["up"]
        pipe = sum(it["ens"] == it["up"] for it in rows) + res - brk
        print(f"  {t:>6.2f} {n:>9} {res:>8} {brk:>6} {pipe:>6}/{len(rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=TAGS, default="choice")
    ap.add_argument("--thumb", default="512")
    ap.add_argument("--baselines", default=None, metavar="JSON",
                    help="pose_baseline.py output to score against, instead of "
                         "the published 2026-08-12 predictions (44 models)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None, help="default decisions_<mode>.json")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    thumbs = [int(t) for t in args.thumb.split(",")]
    tag, out = TAGS[args.mode], args.out or f"decisions_{args.mode}.json"
    keys = [f"{tag}_sheet{t}" for t in thumbs]

    if args.report_only:
        saved = json.load(open(OUT / out))
        gemini_vlm.report(saved["predictions"], thumbs, [tag])
        gemini_vlm.cost_report(saved["usage"])
        for k in keys:
            calibration(saved["predictions"], k)
        return

    labels = load_labels()
    if args.baselines:
        raw = json.loads(Path(args.baselines).read_text())["baselines"]
        base = {k: {"needs_arbiter": v["needs_arbiter"], "geometry": v["geometry"],
                    "ensemble_2048": v["ensemble"]} for k, v in raw.items()}
    else:
        base = load_baselines()
    items = [dict(l, **{"arb": base[l["stem"]]["needs_arbiter"],
                        "geo": base[l["stem"]]["geometry"],
                        "ens": base[l["stem"]]["ensemble_2048"]})
             for l in labels if l["stem"] in base]
    print(f"{len(items)} labelled models | {args.mode} | sheets {thumbs}\n")
    sheets = build_sheets(thumbs, [l for l in labels if l["stem"] in base])
    ask = ASK[args.mode]
    usage = {}
    for t, key in zip(thumbs, keys):
        t0, u, done, lock = time.time(), [], itertools.count(1), threading.Lock()

        def one(it, t=t):
            r = ask(sheets[t][it["stem"]], u)
            with lock:
                print(f"  [{next(done):>3}/{len(items)}] {it['stem'][:38]:38} "
                      f"{'--' if r[0] is None else AX[r[0]]:>3}  {time.time()-t0:5.0f}s", flush=True)
            return r

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(one, items))
        for it, (v, conf, probs) in zip(items, res):
            it[key] = None if v is None else AX[v]
            it[f"{key}_conf"], it[f"{key}_probs"] = conf, probs
        usage[key] = u
        print(f"{key:20} {time.time()-t0:5.0f}s "
              f"({sum(r[0] is not None for r in res)}/{len(items)} answered)", flush=True)

    json.dump({"predictions": items, "usage": usage}, open(OUT / out, "w"),
              indent=1, default=str)
    gemini_vlm.report(items, thumbs, [tag])
    gemini_vlm.cost_report(usage)
    for k in keys:
        calibration(items, k)


if __name__ == "__main__":
    main()
