"""The arbiter task on f3d renders instead of the production Open3D tiles,
crossed with how the tiles are presented: one numbered contact sheet ("grid",
the production shape) vs six separately numbered images ("solo").

  .venv/bin/python eval/f3d_arbiter.py             # sizes 256/512/1024, both
                                                   # presentations, both models
  .venv/bin/python eval/f3d_arbiter.py --report-only

Tiles: per labelled model, one f3d render per up candidate (--up=<axis>,
--config=none or the system config overrides it, az 0 / el 20 like
pose_sheet_tiles, AO+AA, white bg) at 1024, downsized per sheet size. The
UP_PROMPT is verbatim in both presentations; "solo" precedes each image with
a text part "Tile N:". Same 44 labelled models, same report as gemini_vlm.
"""
import argparse, base64, io, itertools, json, subprocess, threading, time
import urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from common import AX, OUT, load_baselines, load_labels

import gemini_vlm                     # report(), cost_report(), PRICES, token()
from src import pose

import os
GLM = os.environ.get("OR_MODEL", "z-ai/glm-5.3-flash")          # the OpenRouter model under test
TAG = os.environ.get("OR_TAG", "glm")
OR_PRICES = {"z-ai/glm-5.3-flash": {"in": 0.075, "out": 0.25},
             "openai/gpt-5.6-luna": {"in": 0.20, "out": 1.20}}   # OpenRouter list, 2026-08-30
ORKEY = Path.home() / ".config/openrouter/key"
SCHEMA = {"type": "object", "properties": {"tile": {"type": "integer"}}, "required": ["tile"]}
TILE_DIR = OUT / "f3d-tiles1024"
UP_TOKEN = {"+Z": "+Z", "-Z": "-Z", "+Y": "+Y", "-Y": "-Y", "+X": "+X", "-X": "-X"}
for col in ("g35-f3d-grid", "g35-f3d-solo"):
    gemini_vlm.PRICES[col] = gemini_vlm.PRICES["gemini-3.5-flash"]
for col in (f"{TAG}-f3d-grid", f"{TAG}-f3d-solo"):
    gemini_vlm.PRICES[col] = OR_PRICES.get(GLM, {"in": 0, "out": 0})


def resolve(labels):
    """The 2026-08-30 Orconspiracy reshuffle: labels whose file moved are found
    again by name under the set's folder (Enemies -> Enemies_Part1/2/4_V2)."""
    orc = Path("/run/media/masa/STLLibrary/Loot Studios/Orconspiracy")
    for l in labels:
        if not l["path"].exists():
            hits = [h for h in orc.rglob(l["path"].name) if "Supported" not in str(h.parent)]
            assert hits, f"missing labelled file: {l['path']}"
            l["path"] = hits[0]
    return labels


def build_tiles(labels):
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    todo = [(l, ax) for l in labels for ax in AX
            if not (TILE_DIR / f"{l['stem']}_{ax}.png").exists()]
    if todo:
        print(f"rendering {len(todo)} f3d tiles")

        def one(la):
            l, ax = la
            out = TILE_DIR / f"{l['stem']}_{ax}.png"
            r = subprocess.run(["f3d", str(l["path"]), "--config=none", f"--up={UP_TOKEN[ax]}",
                                "--camera-azimuth-angle=0", "--camera-elevation-angle=20",
                                "--resolution=1024,1024", f"--output={out}",
                                "--ambient-occlusion", "--anti-aliasing",
                                "--background-color=1,1,1", "--grid=false", "--axis=false",
                                "--filename=false"], capture_output=True, timeout=300)
            if r.returncode != 0 or not out.exists():
                print(f"  FAIL {l['stem']} {ax}: {r.stderr[-120:]}", flush=True)

        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(one, todo))


def has_tiles(stem):
    """f3d cannot load a few STLs at all ("No files loaded"); those models are
    scored as unanswered rather than crashing the sheet builder."""
    return all((TILE_DIR / f"{stem}_{ax}.png").exists() for ax in AX)


def tile_bytes(stem, ax, size):
    im = Image.open(TILE_DIR / f"{stem}_{ax}.png")
    im.thumbnail((size, size))
    buf = io.BytesIO(); im.save(buf, "PNG")
    return buf.getvalue()


def sheet_bytes(stem, size):
    tiles = [Image.open(TILE_DIR / f"{stem}_{ax}.png") for ax in AX]
    buf = io.BytesIO(); pose.make_contact_sheet(tiles, thumb=size).save(buf, "PNG")
    return buf.getvalue()


def gemini_parts(stem, size, solo):
    if solo:
        parts = []
        for i, ax in enumerate(AX, 1):
            parts += [{"text": f"Tile {i}:"},
                      {"inlineData": {"mimeType": "image/png",
                                      "data": base64.b64encode(tile_bytes(stem, ax, size)).decode()}}]
        return parts + [{"text": pose.UP_PROMPT}]
    return [{"inlineData": {"mimeType": "image/png",
                            "data": base64.b64encode(sheet_bytes(stem, size)).decode()}},
            {"text": pose.UP_PROMPT}]


def glm_content(stem, size, solo):
    if solo:
        parts = []
        for i, ax in enumerate(AX, 1):
            parts += [{"type": "text", "text": f"Tile {i}:"},
                      {"type": "image_url", "image_url": {"url": "data:image/png;base64,"
                       + base64.b64encode(tile_bytes(stem, ax, size)).decode()}}]
        return parts + [{"type": "text", "text": pose.UP_PROMPT}]
    return [{"type": "image_url", "image_url": {"url": "data:image/png;base64,"
             + base64.b64encode(sheet_bytes(stem, size)).decode()}},
            {"type": "text", "text": pose.UP_PROMPT}]


def ask_gemini(parts, tok, usage):
    url = (f"https://{gemini_vlm.HOST}/v1/projects/{gemini_vlm.PROJECT}/locations/"
           f"{gemini_vlm.LOCATION}/publishers/google/models/gemini-3.5-flash:generateContent")
    body = json.dumps({"contents": [{"role": "user", "parts": parts}],
                       "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                            "responseSchema": SCHEMA}}).encode()
    req = urllib.request.Request(url, body, {"Authorization": f"Bearer {tok}",
                                             "Content-Type": "application/json"})
    for attempt in range(6):
        t0 = time.time()
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=300).read())
            u = d.get("usageMetadata", {})
            usage.append({"prompt": u.get("promptTokenCount", 0),
                          "thoughts": u.get("thoughtsTokenCount", 0),
                          "output": u.get("candidatesTokenCount", 0), "secs": time.time() - t0})
            text = "".join(p.get("text", "") for p in d["candidates"][0]["content"].get("parts", []))
            return pose.parse_tile_answer(text, 6)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503):
                time.sleep(min(60, 4 * 2 ** attempt))
            else:
                print(f"  gemini HTTP {e.code}: {e.read()[:150]}"); return None
        except Exception as e:
            print(f"  gemini error: {e}")
    return None


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


def ask_glm(content, usage):
    body = {"model": GLM, "temperature": 0,
            "reasoning": {"effort": os.environ.get("OR_EFFORT", "low")},
            "usage": {"include": True},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "tile", "strict": True,
                "schema": dict(SCHEMA, additionalProperties=False)}},
            "messages": [{"role": "user", "content": content}]}
    if os.environ.get("OR_PROVIDER"):
        body["provider"] = {"order": [os.environ["OR_PROVIDER"]], "allow_fallbacks": False}
    body = json.dumps(body).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", body,
                                 {"Authorization": f"Bearer {ORKEY.read_text().strip()}",
                                  "Content-Type": "application/json"})
    for attempt in range(6):
        t0 = time.time()
        try:
            d = json.loads(fetch_with_deadline(req))
            if "error" in d:
                time.sleep(min(60, 4 * 2 ** attempt)); continue
            u = d.get("usage", {}); det = u.get("completion_tokens_details") or {}
            usage.append({"prompt": u.get("prompt_tokens", 0),
                          "thoughts": det.get("reasoning_tokens", 0),
                          "output": u.get("completion_tokens", 0) - det.get("reasoning_tokens", 0),
                          "secs": time.time() - t0, "provider": d.get("provider"),
                          "attempts": attempt + 1})
            return pose.parse_tile_answer(d["choices"][0]["message"]["content"], 6)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                print(f"  {GLM} HTTP {e.code} (attempt {attempt+1}): {e.read()[:120]}", flush=True)
                time.sleep(min(60, 4 * 2 ** attempt))
            else:
                print(f"  glm HTTP {e.code}: {e.read()[:150]}"); return None
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


COLS = ["g35-f3d-grid", "g35-f3d-solo", f"{TAG}-f3d-grid", f"{TAG}-f3d-solo"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thumb", default="256,512,1024")
    ap.add_argument("--cols", default=",".join(COLS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="f3d_arbiter.json")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    thumbs = [int(t) for t in args.thumb.split(",")]
    cols = args.cols.split(",")

    if args.report_only:
        saved = json.load(open(OUT / args.out))
        gemini_vlm.report(saved["predictions"], thumbs, cols)
        gemini_vlm.cost_report(saved["usage"])
        return

    labels = resolve(load_labels())
    base = load_baselines()
    items = [dict(l, **{"arb": base[l["stem"]]["needs_arbiter"],
                        "geo": base[l["stem"]]["geometry"],
                        "ens": base[l["stem"]]["ensemble_2048"]})
             for l in labels if l["stem"] in base]
    build_tiles(labels)
    tok = gemini_vlm.token()
    usage = {}
    for t in thumbs:
        for col in cols:
            solo = col.endswith("solo")
            t0, u, done, lock = time.time(), [], itertools.count(1), threading.Lock()

            def one(it, col=col, t=t, solo=solo):
                if not has_tiles(it["stem"]):
                    return None
                if col.startswith("g35"):
                    v = ask_gemini(gemini_parts(it["stem"], t, solo), tok, u)
                else:
                    v = ask_glm(glm_content(it["stem"], t, solo), u)
                with lock:
                    print(f"  [{next(done):>3}/{len(items)}] {col}@{t} {it['stem'][:32]:32} "
                          f"{'--' if v is None else AX[v]:>3}  {time.time()-t0:4.0f}s", flush=True)
                return v

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                res = list(ex.map(one, items))
            for it, v in zip(items, res):
                it[f"{col}_sheet{t}"] = None if v is None else AX[v]
            usage[f"{col}_sheet{t}"] = u
            print(f"{col:16} @{t:<5} {time.time()-t0:5.0f}s "
                  f"({sum(v is not None for v in res)}/{len(items)})", flush=True)

    json.dump({"predictions": items, "usage": usage}, open(OUT / args.out, "w"),
              indent=1, default=str)
    gemini_vlm.report(items, thumbs, cols)
    gemini_vlm.cost_report(usage)


if __name__ == "__main__":
    main()
