"""Relevance judgments for candidates.json: gemini-3.6-flash sees two 1024px
f3d views of the model and the query, answers {relevance: 0|1|2}. Judged once
per (query, model), blind to which arm retrieved it. -> judgments.json"""
import base64, json, math, subprocess, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path.home() / "Documents/tests/mini-classify"))
from src import pose

HERE = Path(__file__).parent
JR = HERE / "renders/judge"; JR.mkdir(parents=True, exist_ok=True)
S = json.load(open(HERE / "sample.json"))
BY_N = {m["n"]: m for m in S["sample"]}
MODEL = "gemini-3.6-flash"
UP = {(0, 0, 1): "+Z", (0, 0, -1): "-Z", (0, 1, 0): "+Y", (0, -1, 0): "-Y", (1, 0, 0): "+X", (-1, 0, 0): "-X"}
VIEWS = [(30, 15), (210, 15)]
SCHEMA = {"type": "object", "properties": {"relevance": {"type": "integer"}, "reason": {"type": "string"}},
          "required": ["relevance", "reason"]}
PROMPT = ("These are two views of one 3D-printable miniature/model. A user searched a model "
          "library for: \"{q}\".\n\nRate how well this model matches that search: 2 = clearly a "
          "match (this is what the user was looking for), 1 = partial or arguable (related "
          "subject, or a component/accessory of one), 0 = not a match. Judge the subject of the "
          "model, not render quality. Answer as JSON with fields relevance and reason.")
_lock = __import__("threading").Lock()


def views(m):
    out = []
    for az, el in VIEWS:
        p = JR / f"{m['key']}_{az}_{el}.png"
        if not p.exists():
            up = UP[tuple(int(round(c)) for c in m["up"])]
            subprocess.run(["f3d", m["path"], "--config=none", f"--up={up}",
                            f"--camera-azimuth-angle={az}", f"--camera-elevation-angle={el}",
                            "--resolution=1024,1024", f"--output={p}", "--ambient-occlusion",
                            "--anti-aliasing", "--background-color=1,1,1", "--grid=false",
                            "--axis=false", "--filename=false"], capture_output=True, timeout=300)
        if not p.exists():   # f3d cannot load a few STLs; fall back to the Open3D tight renders
            p = HERE / "renders/open3d_tight" / f"{m['key']}_v{1 if az == 30 else 5}.png"
        out.append(p)
    return out


def ask(q, m, tok, project):
    url = (f"https://{pose.GEMINI_HOST}/v1/projects/{project}/locations/{pose.GEMINI_LOCATION}"
           f"/publishers/google/models/{MODEL}:generateContent")
    parts = [{"inlineData": {"mimeType": "image/png", "data": base64.b64encode(p.read_bytes()).decode()}}
             for p in views(m)] + [{"text": PROMPT.format(q=q)}]
    body = json.dumps({"contents": [{"role": "user", "parts": parts}],
                       "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                            "responseSchema": SCHEMA}}).encode()
    req = urllib.request.Request(url, body, {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=300).read())
            text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
            j = json.loads(text)
            u = d.get("usageMetadata", {})
            return {"relevance": int(j["relevance"]), "reason": j.get("reason", "")[:200],
                    "tok": u.get("promptTokenCount", 0) + u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)}
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503):
                time.sleep(min(60, 4 * 2 ** attempt)); continue
            return {"error": f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"}
        except Exception as e:
            err = str(e)[:200]
    return {"error": err if 'err' in dir() else "retries exhausted"}


def main():
    cands = json.load(open(HERE / "candidates.json"))
    out_p = HERE / "judgments.json"
    judg = json.load(open(out_p)) if out_p.exists() else {}
    todo = [(q, n) for q, ns in cands.items() for n in ns if f"{q}|{n}" not in judg or "error" in judg[f"{q}|{n}"]]
    print(f"{len(todo)} judgments to make ({len(judg)} cached)", flush=True)
    project, tok = pose.gcloud_project(), pose.gcloud_token()
    t0 = time.time()

    def one(qn):
        q, n = qn
        r = ask(q, BY_N[n], tok, project)
        with _lock:
            judg[f"{q}|{n}"] = r
            if len(judg) % 25 == 0:
                json.dump(judg, open(out_p, "w"), indent=0)
                print(f"  {len(judg)} done, {time.time()-t0:.0f}s", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(one, todo))
    json.dump(judg, open(out_p, "w"), indent=0)
    errs = sum("error" in v for v in judg.values())
    tok_total = sum(v.get("tok", 0) for v in judg.values())
    print(f"done: {len(judg)} judgments, {errs} errors, ~{tok_total} tokens, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
