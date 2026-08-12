"""Run the pose-arbiter prompt through Gemini models on Vertex AI and compare
with the gemma/haiku/sonnet numbers already in LEARNINGS.

  python gemini_vlm.py                 # both sheet sizes, all three models
  python gemini_vlm.py --thumb 512     # one sheet size
  python gemini_vlm.py --models gemini-2.5-flash

Same 44 hand-labelled models, same UP_PROMPT, same sheets. Auth is gcloud ADC
(`gcloud auth application-default login`); no SDK dependency — one HTTPS POST
per model per sheet, so this contends with nothing on the GPU.
"""
import argparse, base64, json, subprocess, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import AX, OUT, build_sheets, load_baselines, load_labels  # puts REPO on sys.path

import pose

PROJECT = "mini-classify"
LOCATION = "global"
HOST = "aiplatform.googleapis.com"
MODELS = ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-pro"]
# {"tile": n} is forced by the response schema rather than asked for in prose,
# the same contract the ollama backend uses.
SCHEMA = {"type": "object", "properties": {"tile": {"type": "integer"}},
          "required": ["tile"]}

# Production shape the cost table is scaled to: the arbiter fired on 354 of the
# 602 models in the last full run.
COLLECTION, ARBITER_CALLS = 602, 354
# $/M tokens, list price. Gemini: ai.google.dev/gemini-api/docs/pricing (Vertex
# bills the same rates for these models). Claude: platform.claude.com/docs/pricing
# — the harness drives them through the `claude` CLI, which may bill against a
# subscription rather than per token; API list price is the comparable basis.
PRICES_AS_OF = "2026-08-12"
PRICES = {
    "gemini-3.5-flash": {"in": 1.50, "out": 9.00},
    "gemini-2.5-flash": {"in": 0.30, "out": 2.50},
    "gemini-2.5-pro":   {"in": 1.25, "out": 10.00},   # <=200k-token prompts
    "haiku":            {"in": 1.00, "out": 5.00},    # claude-haiku-4-5
    "sonnet":           {"in": 3.00, "out": 15.00},   # claude-sonnet-5 list
}


def token():
    return subprocess.run(["gcloud", "auth", "application-default", "print-access-token"],
                          capture_output=True, text=True, check=True).stdout.strip()


def ask(model, sheet, tok, n_tiles=6, usage=None):
    """One arbiter call. Returns a 0-based candidate index, or None — the
    pipeline never hard-fails because of the VLM, so neither does the harness.

    Appends the reported token counts to `usage` when given; thinking tokens
    are billed as output and are most of what a reasoning model costs here,
    so cost has to be measured per call rather than assumed."""
    url = (f"https://{HOST}/v1/projects/{PROJECT}/locations/{LOCATION}"
           f"/publishers/google/models/{model}:generateContent")
    body = json.dumps({
        "contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": "image/png",
                            "data": base64.b64encode(sheet.read_bytes()).decode()}},
            {"text": pose.UP_PROMPT}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                             "responseSchema": SCHEMA},
    }).encode()
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, body, {"Authorization": f"Bearer {tok}",
                                                     "Content-Type": "application/json"})
            t0 = time.time()
            d = json.loads(urllib.request.urlopen(req, timeout=300).read())
            u = d.get("usageMetadata", {})
            if usage is not None:
                usage.append({"model": model, "secs": round(time.time() - t0, 2),
                              "prompt": u.get("promptTokenCount", 0),
                              "output": u.get("candidatesTokenCount", 0),
                              "thoughts": u.get("thoughtsTokenCount", 0)})
            parts = d["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
            v = pose.parse_tile_answer(text, n_tiles)
            if v is not None:
                return v
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503):        # transient — back off and retry
                time.sleep(2 * (attempt + 1))
            elif e.code != 400:
                print(f"  {model} HTTP {e.code}: {e.read()[:200]}")
                return None
        except Exception as e:
            print(f"  {model} error: {e}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thumb", default="256,512",
                    help="contact-sheet tile size(s); the knob that mattered")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report-only", action="store_true",
                    help="re-print the tables from out/gemini_vlm.json without calling the API")
    args = ap.parse_args()
    thumbs = [int(t) for t in args.thumb.split(",")]
    models = args.models.split(",")

    if args.report_only:
        saved = json.load(open(OUT / "gemini_vlm.json"))
        report(saved["predictions"], thumbs, models)
        cost_report(saved["usage"])
        return

    labels = load_labels()
    base = load_baselines()
    items = [dict(l, **{"arb": base[l["stem"]]["needs_arbiter"],
                        "geo": base[l["stem"]]["geometry"],
                        "ens": base[l["stem"]]["ensemble_2048"]})
             for l in labels if l["stem"] in base]
    print(f"{len(items)} labelled models ({sum(i['set']=='orig' for i in items)} orig + "
          f"{sum(i['set']=='holdout' for i in items)} holdout) | sheets {thumbs} | "
          f"{len(models)} models\n")

    sheets = build_sheets(thumbs, labels)
    tok = token()
    usage = {}
    for t in thumbs:
        for m in models:
            t0, u = time.time(), []
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                res = list(ex.map(lambda it: ask(m, sheets[t][it["stem"]], tok, usage=u), items))
            for it, v in zip(items, res):
                it[f"{m}_sheet{t}"] = None if v is None else AX[v]
            usage[f"{m}_sheet{t}"] = u
            n_ok = sum(v is not None for v in res)
            print(f"{m:20} @{t:<5} {time.time()-t0:5.0f}s  ({n_ok}/{len(items)} answered)")

    json.dump({"predictions": items, "usage": usage},
              open(OUT / "gemini_vlm.json", "w"), indent=1, default=str)
    report(items, thumbs, models)
    cost_report(usage)


def report(items, thumbs, models):
    keys = [f"{m}_sheet{t}" for t in thumbs for m in models]
    ref = [f"{m}_sheet{t}" for t in thumbs
           for m in ["gemma", "haiku", "sonnet"]]
    base = load_baselines()
    for it in items:
        for k in ref:
            it.setdefault(k, base[it["stem"]].get(k))

    print(f"\n{'model':30} {'gold':>5} {'geo':>5} {'ens':>5} "
          + " ".join(f"{k.replace('gemini-','')[:12]:>13}" for k in keys))
    for it in items:
        g = it["up"]
        mk = lambda p: "--" if p is None else p + ("" if p == g else "*")
        print(f"{it['stem'][:30]:30} {g:>5} {mk(it['geo']):>5} {mk(it['ens']):>5} "
              + " ".join(f"{mk(it[k]):>13}" for k in keys))

    o = [x for x in items if x["set"] == "orig"]
    h = [x for x in items if x["set"] == "holdout"]
    def acc(key, sel):
        s = [x for x in sel if x.get(key) is not None]
        return f"{sum(x[key]==x['up'] for x in s)}/{len(s)}" if s else "--"
    print(f"\nstandalone accuracy\n{'method':30} {'orig':>8} {'holdout':>9} {'pooled':>8}")
    for key in ["geo", "ens"] + keys + ref:
        print(f"{key:30} {acc(key,o):>8} {acc(key,h):>9} {acc(key,items):>8}")

    print("\nanswer distribution (a repeated pick is a positional prior, not a judgement)")
    print(f"{'method':30} " + " ".join(f"{a:>4}" for a in AX))
    for key in ["truth"] + keys + ref:
        c = [sum((x["up"] if key == "truth" else x.get(key)) == a for x in items) for a in AX]
        print(f"{key:30} " + " ".join(f"{n:>4}" for n in c))

    print("\nas the arbiter tier (overrides the ensemble when needs_arbiter fires)")
    arb = [x for x in items if x["arb"]]
    for key in keys + ref:
        s = [x for x in arb if x.get(key) is not None]
        if not s:
            continue
        resc = [x["stem"] for x in s if x[key] == x["up"] != x["ens"]]
        brk = [x["stem"] for x in s if x["ens"] == x["up"] != x[key]]
        pipe = sum((x[key] if x["arb"] else x["ens"]) == x["up"]
                   for x in items if x.get(key) is not None)
        print(f"  {key:28} on {len(s):>2} arbiter models: rescued {len(resc)}, "
              f"broke {len(brk)} -> net {len(resc)-len(brk):+d}   pipeline {pipe}/{len(items)}")
        if resc: print(f"       rescued: {resc}")
        if brk:  print(f"       broke:   {brk}")


def cost_report(usage):
    """Measured tokens per arbiter call, priced out at the rate table above.

    The unit that matters is not a call but a *full-collection run*: the
    arbiter fired on 354 of 602 models in production.
    """
    print(f"\nmeasured cost per arbiter call ({ARBITER_CALLS} calls on a "
          f"{COLLECTION} model collection)")
    hdr = (f"{'method':28} {'in tok':>7} {'think':>6} {'out':>5} {'secs':>6} "
           f"{'$/1k calls':>11} {'$/run':>8}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for key, calls in usage.items():
        if not calls:
            continue
        n = len(calls)
        pin = sum(c["prompt"] for c in calls) / n
        pth = sum(c["thoughts"] for c in calls) / n
        pou = sum(c["output"] for c in calls) / n
        sec = sum(c["secs"] for c in calls) / n
        model = key.rsplit("_sheet", 1)[0]
        p = PRICES.get(model)
        if p:
            per = (pin * p["in"] + (pth + pou) * p["out"]) / 1e6
            rows.append((key, pin, pth, pou, sec, per * 1000, per * ARBITER_CALLS))
        else:
            rows.append((key, pin, pth, pou, sec, None, None))
    for key, pin, pth, pou, sec, k, run in sorted(rows, key=lambda r: (r[6] is None, r[6] or 0)):
        money = f"{k:>11.2f} {run:>8.2f}" if k is not None else f"{'?':>11} {'?':>8}"
        print(f"{key:28} {pin:>7.0f} {pth:>6.0f} {pou:>5.0f} {sec:>6.1f} {money}")
    print(f"\nprices $/M tokens as of {PRICES_AS_OF}; thinking tokens bill as output.")
    print("gemma4:26b is local — no token cost, but it holds 17 GB of an 8 GB card "
          "and\nevicts SigLIP between calls (10.1 s reload against 0.49 s of inference).")


if __name__ == "__main__":
    main()
