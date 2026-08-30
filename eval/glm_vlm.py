"""Run the pose-arbiter prompt through GLM-5.3-Flash (OpenRouter) on the same
44 hand-labelled models, same UP_PROMPT, same cached contact sheets as
eval/gemini_vlm.py, and print the same tables — so the GLM column is directly
comparable with the published gemini/gemma/haiku/sonnet numbers.

  .venv/bin/python eval/glm_vlm.py                 # sheets 256+512, efforts low+high
  .venv/bin/python eval/glm_vlm.py --efforts low --thumb 512
  .venv/bin/python eval/glm_vlm.py --report-only

Auth: ~/.config/openrouter/key. One POST per call; contends with nothing on
the GPU. Reasoning cannot be disabled on this endpoint, so effort is the knob.
"""
import argparse, base64, itertools, json, threading, time
import urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import AX, OUT, build_sheets, load_baselines, load_labels  # REPO on sys.path

import gemini_vlm                       # report(), cost_report(), PRICES table
from src import pose

import os
MODEL = os.environ.get("OR_MODEL", "z-ai/glm-5.3-flash")     # any OpenRouter vision model
TAG = os.environ.get("OR_TAG", "glm")                           # column prefix in the tables
URL = "https://openrouter.ai/api/v1/chat/completions"
OR_PRICES = {"z-ai/glm-5.3-flash": {"in": 0.075, "out": 0.25},
             "openai/gpt-5.6-luna": {"in": 0.20, "out": 1.20}}   # OpenRouter list, 2026-08-30

KEY = Path.home() / ".config/openrouter/key"
SCHEMA = {"type": "object", "properties": {"tile": {"type": "integer"}},
          "required": ["tile"], "additionalProperties": False}
# $/M as listed on OpenRouter 2026-08-30; keyed by the column names below so
# gemini_vlm.cost_report can price the rows.
for _e in ("low", "high"):
    gemini_vlm.PRICES[f"{TAG}-{_e}"] = OR_PRICES.get(MODEL, {"in": 0, "out": 0})


# A request that OpenRouter keeps alive with heartbeat bytes while the upstream
# provider hangs never trips urlopen's socket timeout (measured 2026-08-30:
# workers idle 46 min on connections receiving ~1 packet/150 ms). So the wall
# clock is enforced from outside: the read runs on a helper thread and is
# abandoned at the deadline; the thread and its socket die with the process.
_DEADLINE_S = float(__import__("os").environ.get("OR_DEADLINE", 120))
_deadline_pool = __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=64)


def fetch_with_deadline(req, deadline=None):
    fut = _deadline_pool.submit(lambda: urllib.request.urlopen(req, timeout=300).read())
    return fut.result(timeout=deadline or _DEADLINE_S)   # raises futures.TimeoutError


def ask(effort, sheet, n_tiles=6, usage=None):
    """One arbiter call. 0-based candidate index or None, like gemini_vlm.ask."""
    body = {
        "model": MODEL, "temperature": 0, "reasoning": {"effort": effort},
        "usage": {"include": True},
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "tile", "strict": True, "schema": SCHEMA}},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,"
                                                + base64.b64encode(sheet.read_bytes()).decode()}},
            {"type": "text", "text": pose.UP_PROMPT}]}],
    }
    if os.environ.get("OR_PROVIDER"):          # pin, e.g. OR_PROVIDER=Z.AI; auto-routed otherwise
        body["provider"] = {"order": [os.environ["OR_PROVIDER"]], "allow_fallbacks": False}
    body = json.dumps(body).encode()
    req = urllib.request.Request(URL, body, {"Authorization": f"Bearer {KEY.read_text().strip()}",
                                             "Content-Type": "application/json"})
    for attempt in range(6):
        t0 = time.time()
        try:
            d = json.loads(fetch_with_deadline(req))
            if "error" in d:
                time.sleep(min(60, 4 * 2 ** attempt)); continue
            u = d.get("usage", {})
            det = u.get("completion_tokens_details") or {}
            if usage is not None:
                usage.append({"prompt": u.get("prompt_tokens", 0),
                              "thoughts": det.get("reasoning_tokens", 0),
                              "output": u.get("completion_tokens", 0) - det.get("reasoning_tokens", 0),
                              "secs": time.time() - t0, "provider": d.get("provider"),
                              "attempts": attempt + 1})
            v = pose.parse_tile_answer(d["choices"][0]["message"]["content"], n_tiles)
            if v is not None:
                return v
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                print(f"  {MODEL} HTTP {e.code} (attempt {attempt+1}): {e.read()[:120]}", flush=True)
                time.sleep(min(60, 4 * 2 ** attempt))
            else:
                print(f"  {MODEL} HTTP {e.code}: {e.read()[:200]}")
                return None
        except __import__("concurrent.futures").futures.TimeoutError:
            # A reasoning loop on an ambiguous shape is reproducible: the retry
            # re-runs the same loop server-side and bills it again (measured
            # 2026-08-30: Floor at max effort, 6 abandoned attempts, 17k output
            # tokens each). One deadline is the verdict — unanswered.
            print(f"  deadline {_DEADLINE_S:.0f}s exceeded — unanswered, not retried", flush=True)
            return None
        except Exception as e:
            print(f"  glm error: {e}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thumb", default="256,512")
    ap.add_argument("--efforts", default="low,high")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="glm_vlm.json")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    thumbs = [int(t) for t in args.thumb.split(",")]
    cols = [f"{TAG}-{e}" for e in args.efforts.split(",")]

    if args.report_only:
        saved = json.load(open(OUT / args.out))
        gemini_vlm.report(saved["predictions"], thumbs, cols)
        gemini_vlm.cost_report(saved["usage"])
        return

    labels = load_labels()
    base = load_baselines()
    items = [dict(l, **{"arb": base[l["stem"]]["needs_arbiter"],
                        "geo": base[l["stem"]]["geometry"],
                        "ens": base[l["stem"]]["ensemble_2048"]})
             for l in labels if l["stem"] in base]
    print(f"{len(items)} labelled models | sheets {thumbs} | efforts {args.efforts}\n")
    sheets = build_sheets(thumbs, labels)
    usage = {}
    for t in thumbs:
        for col in cols:
            effort = col.split("-", 1)[1]
            t0, u, done, lock = time.time(), [], itertools.count(1), threading.Lock()

            def one(it, effort=effort, t=t):
                v = ask(effort, sheets[t][it["stem"]], usage=u)
                with lock:
                    print(f"  [{next(done):>3}/{len(items)}] {it['stem'][:38]:38} "
                          f"{'--' if v is None else AX[v]:>3}  {time.time()-t0:5.0f}s", flush=True)
                return v

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                res = list(ex.map(one, items))
            for it, v in zip(items, res):
                it[f"{col}_sheet{t}"] = None if v is None else AX[v]
            usage[f"{col}_sheet{t}"] = u
            print(f"{col:20} @{t:<5} {time.time()-t0:5.0f}s "
                  f"({sum(v is not None for v in res)}/{len(items)} answered)", flush=True)

    json.dump({"predictions": items, "usage": usage}, open(OUT / args.out, "w"),
              indent=1, default=str)
    gemini_vlm.report(items, thumbs, cols)
    gemini_vlm.cost_report(usage)


if __name__ == "__main__":
    main()
