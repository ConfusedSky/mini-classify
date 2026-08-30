"""Description-panel matrix on the Proto-Osteotron.

4 renderers x 3 conditions (control / temp panel / viewset panel) x 2 efforts,
describers = GLM-5.3-Flash; each panel synthesized by BOTH GLM (cell effort)
and gemini-3.6-flash from the same three descriptions. -> panel_matrix.json
"""
import base64, json, os, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path.home() / "Documents/tests/mini-classify"))
from src import pose

HERE = Path(__file__).parent
PILOT = Path.home() / "Documents/tests/mini-classify/eval/renderer_pilot/renders"
F3D_DIR = HERE / "f3d1024-renders"; F3D_DIR.mkdir(exist_ok=True)   # gitignored (*renders*/)
STL = ("/run/media/masa/STLLibrary/Loot Studios/Spacecraft-Crasher/All_Aliens_Part1/"
       "JuvenileProtoOsteotron1/32mm/No Supports/32mm_JuvenileProtoOsteotron1.stl")
BB_MIN, BB_MAX = (76.23, 71.32, 0.0), (103.77, 108.68, 13.43)   # up = -Y (pose cache)


def render(azimuth=0, elevation=10, zoom=1.0, vertical_focus=0.5):
    """f3d 1024px, explicit camera (azimuth 0 = front), AO+AA, --config=none."""
    import math, subprocess
    zoom = max(0.5, min(4.0, float(zoom))); vf = max(0.0, min(1.0, float(vertical_focus)))
    out = F3D_DIR / f"r_{azimuth:.0f}_{elevation:.0f}_{zoom:.2f}_{vf:.2f}.png"
    if not out.exists():
        h = BB_MAX[1] - BB_MIN[1]
        fx = (BB_MIN[0] + BB_MAX[0]) / 2
        fy = BB_MAX[1] - vf * h
        fz = (BB_MIN[2] + BB_MAX[2]) / 2
        dist = (h / 2) / math.tan(math.radians(15)) * 1.15 / zoom
        a, e = math.radians(azimuth), math.radians(elevation)
        dx = -math.sin(a) * math.cos(e); dz = -math.cos(a) * math.cos(e); dy = -math.sin(e)
        subprocess.run(["f3d", STL, "--config=none", "--up=-Y",
                        f"--camera-position={fx+dist*dx},{fy+dist*dy},{fz+dist*dz}",
                        f"--camera-focal-point={fx},{fy},{fz}", "--camera-view-up=0,-1,0",
                        "--camera-view-angle=30", "--resolution=1024,1024", f"--output={out}",
                        "--ambient-occlusion", "--anti-aliasing", "--background-color=0.92,0.92,0.92",
                        "--grid=false", "--axis=false", "--filename=false"],
                       capture_output=True, timeout=180, check=True)
    return out
KEY_ = "32mm_JuvenileProtoOsteotron1_e71033"
GLM, GEMINI = "z-ai/glm-5.3-flash", "gemini-3.6-flash"
ORKEY = Path.home() / ".config/openrouter/key"
EFFORTS = ["low", "high"]

Q = ("These are several renders of one 3D-printable miniature STL, viewed from "
     "different angles. What is this a model of? Answer in 2-4 sentences: what the "
     "figure is, its pose, notable features, and likely genre/game context.")
SYNTH = ("Three independent observers each described the same 3D-printable miniature "
         "after viewing renders of it from different angles. Their descriptions:\n\n"
         "1) {d0}\n\n2) {d1}\n\n3) {d2}\n\n"
         "Produce a consensus. Answer as JSON with fields: agreements (one sentence), "
         "disagreements (one sentence naming any conflicting readings), final (the "
         "consensus description of what the model is, 2-4 sentences; where observers "
         "conflict, prefer the reading best supported by the majority and by anatomical "
         "coherence).")
SCHEMA = {"type": "object", "properties": {"agreements": {"type": "string"},
                                           "disagreements": {"type": "string"},
                                           "final": {"type": "string"}},
          "required": ["agreements", "disagreements", "final"], "additionalProperties": False}


def views_512(arm, idxs):
    return [PILOT / arm / f"{KEY_}_v{i}.png" for i in idxs]


def viewpacks(renderer):
    if renderer == "f3d-1024":
        v1 = [render(az, 10) for az in (0, 90, 180, 270)]
        v2 = [render(az, 25) for az in (45, 135, 225, 315)]
        v3 = [render(90, 10), render(270, 10), render(0, 15, 2.5, 0.85)]
        return v1, v2, v3
    arm = {"open3d_tight": "open3d_tight", "three_ao": "three_ao", "f3d": "f3d"}[renderer]
    return (views_512(arm, [0, 2, 4, 6]),        # el +20, az 0/90/180/270
            views_512(arm, [1, 3, 5, 7]),        # el +20, az 45/135/225/315
            views_512(arm, [8, 10, 12, 14]))     # el -20 ring


def img_part_glm(p):
    from PIL import Image
    import io
    im = Image.open(p)
    if im.mode == "RGBA":                        # three_ao: composite over zinc-900
        bg = Image.new("RGB", im.size, (0x18, 0x18, 0x1b)); bg.paste(im, mask=im.getchannel("A")); im = bg
    buf = io.BytesIO(); im.convert("RGB").save(buf, "PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


TELEMETRY = []      # one row per API call: model, effort, secs, tokens, cost
_tlock = __import__("threading").Lock()


def _record(**row):
    with _tlock:
        TELEMETRY.append(row)


def call_glm(messages, effort, temp, schema=None, model=None):
    import os
    body = {"model": model or GLM, "messages": messages, "temperature": temp,
            "reasoning": {"effort": effort}, "usage": {"include": True}}
    if os.environ.get("GLM_PROVIDER") and (model or GLM) == GLM:
        body["provider"] = {"order": [os.environ["GLM_PROVIDER"]], "allow_fallbacks": False}
    if schema:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "consensus", "strict": True, "schema": schema}}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Authorization": f"Bearer {ORKEY.read_text().strip()}",
                                  "Content-Type": "application/json"})
    err = "no attempt made"
    for attempt in range(8):
        try:
            t0 = time.time()
            d = json.loads(urllib.request.urlopen(req, timeout=300).read())
            if "error" in d:
                err = json.dumps(d["error"])[:300]
                time.sleep(4 * 2 ** attempt); continue
            u = d.get("usage", {}) or {}; det = u.get("completion_tokens_details") or {}
            _record(model=body["model"], provider=d.get("provider"), effort=effort, temp=temp,
                    kind="synth" if schema else "describe", secs=round(time.time() - t0, 2),
                    prompt=u.get("prompt_tokens", 0), completion=u.get("completion_tokens", 0),
                    reasoning=det.get("reasoning_tokens", 0), cost=u.get("cost", 0) or 0,
                    attempts=attempt + 1)
            return d["choices"][0]["message"]["content"].strip(), u.get("cost", 0) or 0
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"
            if e.code in (429, 500, 502, 503):
                time.sleep(min(90, 6 * 2 ** attempt)); continue
            raise RuntimeError(err)
        except OSError as e:
            err = str(e); time.sleep(4 * 2 ** attempt)
    raise RuntimeError(f"retries exhausted: {err}")  # err set by every failure path


def describe(imgs, effort, temp, seed_hint=""):
    text = Q + (f"\n\n(observer {seed_hint})" if seed_hint else "")
    import os
    msgs = [{"role": "user", "content": [*map(img_part_glm, imgs), {"type": "text", "text": text}]}]
    return call_glm(msgs, effort, temp, model=os.environ.get("DESCRIBE_MODEL"))  # default GLM


def synth_glm(descs, effort):
    msgs = [{"role": "user", "content": SYNTH.format(d0=descs[0], d1=descs[1], d2=descs[2])}]
    out, cost = call_glm(msgs, effort, 0, SCHEMA)
    return json.loads(out), cost


def synth_gemini(descs, tok, project):
    url = (f"https://{pose.GEMINI_HOST}/v1/projects/{project}/locations/{pose.GEMINI_LOCATION}"
           f"/publishers/google/models/{GEMINI}:generateContent")
    body = json.dumps({"contents": [{"role": "user", "parts": [
        {"text": SYNTH.format(d0=descs[0], d1=descs[1], d2=descs[2])}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                             "responseSchema": {k: v for k, v in SCHEMA.items() if k != "additionalProperties"}}}).encode()
    req = urllib.request.Request(url, body, {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            t0 = time.time()
            d = json.loads(urllib.request.urlopen(req, timeout=300).read())
            parts = d["candidates"][0]["content"]["parts"]
            u = d.get("usageMetadata", {})
            _record(model=GEMINI, provider="vertex", effort="default", temp=0, kind="synth",
                    secs=round(time.time() - t0, 2), prompt=u.get("promptTokenCount", 0),
                    completion=u.get("candidatesTokenCount", 0), reasoning=u.get("thoughtsTokenCount", 0),
                    cost=None, attempts=attempt + 1)
            return json.loads("".join(p.get("text", "") for p in parts))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503):
                time.sleep(min(60, 4 * 2 ** attempt)); continue
            raise RuntimeError(f"HTTP {e.code}")


RENDERERS = ["open3d_tight", "three_ao", "f3d", "f3d-1024"]


def main():
    import os
    (HERE / "out").mkdir(exist_ok=True)
    out_p = HERE / "out" / os.environ.get("PANEL_OUT", "panel_matrix.json")
    R = json.load(open(out_p)) if out_p.exists() else {}
    tok, project = pose.gcloud_token(), pose.gcloud_project()
    t0, cost = time.time(), [0.0]

    def cell_jobs():
        for r in RENDERERS:
            v1, v2, v3 = viewpacks(r)
            for e in EFFORTS:
                yield f"{r}|control|{e}", [(v1, 0, "")]
                yield f"{r}|temp-panel|{e}", [(v1, 0.8, f"{i+1} of 3") for i in range(3)]
                yield f"{r}|view-panel|{e}", [(v, 0, "") for v in (v1, v2, v3)]

    def run_describer(args):
        cid, i, (imgs, temp, hint), effort = args
        d, c = describe(imgs, effort, temp, hint)
        cost[0] += c
        return cid, i, d

    jobs = []
    for cid, packs in cell_jobs():
        effort = cid.rsplit("|", 1)[1]
        if cid not in R:
            R[cid] = {"descriptions": [None] * len(packs)}
        for i, pk in enumerate(packs):
            if R[cid]["descriptions"][i] is None:
                jobs.append((cid, i, pk, effort))
    print(f"{len(jobs)} describer calls")
    with ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", 6))) as ex:
        for cid, i, d in ex.map(run_describer, jobs):
            R[cid]["descriptions"][i] = d
            print(f"  {cid} [{i}] done {time.time()-t0:.0f}s", flush=True)
    json.dump(R, open(out_p, "w"), indent=1)

    def run_synth(cid):
        effort = cid.rsplit("|", 1)[1]
        ds = R[cid]["descriptions"]
        if "synth_glm" not in R[cid]:
            R[cid]["synth_glm"], c = synth_glm(ds, effort); cost[0] += c
        if "synth_gemini" not in R[cid]:
            R[cid]["synth_gemini"] = synth_gemini(ds, tok, project)
        return cid

    panel_ids = [cid for cid in R if "panel" in cid]
    with ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", 6))) as ex:
        for cid in ex.map(run_synth, panel_ids):
            print(f"  synth {cid} done {time.time()-t0:.0f}s", flush=True)
    json.dump(R, open(out_p, "w"), indent=1)
    json.dump(TELEMETRY, open(out_p.with_suffix(".telemetry.json"), "w"), indent=0)
    print(f"all done {time.time()-t0:.0f}s, GLM cost ${cost[0]:.3f}, {len(TELEMETRY)} calls logged")


if __name__ == "__main__":
    main()
