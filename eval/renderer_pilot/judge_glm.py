"""Second judge: z-ai/glm-5.3-flash via OpenRouter on the identical
candidates, views and prompt as judge.py -> out/judgments_glm.json.
Reasoning cannot be disabled on this endpoint; effort is 'low' (the level
that read the Osteotron correctly at 7 s). Then `python judge_glm.py --agree`
reports agreement with the Gemini judge and re-scores every arm under it."""
import json, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import judge  # candidates, views(), PROMPT, S

HERE = Path(__file__).parent
OUT = HERE / "out"
MODEL = "z-ai/glm-5.3-flash"
URL = "https://openrouter.ai/api/v1/chat/completions"
KEY = Path.home() / ".config/openrouter/key"
SCHEMA = {"type": "object", "properties": {"relevance": {"type": "integer", "enum": [0, 1, 2]},
                                           "reason": {"type": "string"}},
          "required": ["relevance", "reason"], "additionalProperties": False}
_lock = threading.Lock()


def img(p):
    import base64
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()}}


def ask(q, m):
    body = {"model": MODEL, "temperature": 0, "reasoning": {"effort": "low"},
            "usage": {"include": True},
            "response_format": {"type": "json_schema", "json_schema": {"name": "relevance", "strict": True, "schema": SCHEMA}},
            "messages": [{"role": "user", "content": [*map(img, judge.views(m)),
                                                      {"type": "text", "text": judge.PROMPT.format(q=q)}]}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {
        "Authorization": f"Bearer {KEY.read_text().strip()}", "Content-Type": "application/json"})
    err = "retries exhausted"
    for attempt in range(6):
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=300).read())
            if "error" in d:
                err = json.dumps(d["error"])[:200]; time.sleep(4 * 2 ** attempt); continue
            j = json.loads(d["choices"][0]["message"]["content"])
            u = d.get("usage", {})
            return {"relevance": int(j["relevance"]), "reason": str(j.get("reason", ""))[:200],
                    "tok": u.get("total_tokens", 0), "usd": u.get("cost", 0)}
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"
            if e.code in (429, 500, 502, 503):
                time.sleep(min(60, 4 * 2 ** attempt)); continue
            return {"error": err}
        except Exception as e:  # malformed JSON, timeouts
            err = str(e)[:200]; time.sleep(2)
    return {"error": err}


def run():
    cands = json.load(open(OUT / "candidates.json"))
    out_p = OUT / "judgments_glm.json"
    judg = json.load(open(out_p)) if out_p.exists() else {}
    todo = [(q, n) for q, ns in cands.items() for n in ns if f"{q}|{n}" not in judg or "error" in judg[f"{q}|{n}"]]
    print(f"{len(todo)} to judge ({len(judg)} cached)", flush=True)
    t0 = time.time()

    def one(qn):
        q, n = qn
        r = ask(q, judge.BY_N[n])
        with _lock:
            judg[f"{q}|{n}"] = r
            if len(judg) % 25 == 0:
                json.dump(judg, open(out_p, "w"), indent=0); print(f"  {len(judg)} {time.time()-t0:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(one, todo))
    json.dump(judg, open(out_p, "w"), indent=0)
    errs = sum("error" in v for v in judg.values())
    print(f"done: {len(judg)} judgments, {errs} errors, {sum(v.get('tok', 0) for v in judg.values())} tokens, "
          f"${sum(v.get('usd', 0) or 0 for v in judg.values()):.3f}, {time.time()-t0:.0f}s")


def ndcg(q, order, J, k=20):
    g = [J.get(f"{q}|{n}", {}).get("relevance", 0) for n in order[:k]]
    dcg = sum(x / np.log2(i + 2) for i, x in enumerate(g))
    ideal = sorted([v.get("relevance", 0) for kk, v in J.items() if kk.startswith(q + "|")], reverse=True)[:k]
    return dcg / (sum(x / np.log2(i + 2) for i, x in enumerate(ideal)) or 1)


def agree():
    G = json.load(open(OUT / "judgments.json")); L = json.load(open(OUT / "judgments_glm.json"))
    keys = [k for k in G if k in L and "relevance" in G[k] and "relevance" in L[k]]
    g = np.array([G[k]["relevance"] for k in keys]); l = np.array([L[k]["relevance"] for k in keys])
    conf = np.zeros((3, 3), int)
    for a, b in zip(g, l):
        conf[a, b] += 1
    exact = (g == l).mean(); within1 = (np.abs(g - l) <= 1).mean()
    # quadratic-weighted Cohen's kappa
    n = len(g); w = np.array([[(i - j) ** 2 for j in range(3)] for i in range(3)]) / 4
    exp = np.outer(conf.sum(1), conf.sum(0)) / n
    kappa = 1 - (w * conf).sum() / (w * exp).sum()
    binG, binL = g == 2, l == 2
    print(f"{n} pairs judged by both. exact agreement {exact:.3f}, within one level {within1:.3f}, "
          f"quadratic kappa {kappa:.3f}, agreement on 'clearly relevant' {(binG == binL).mean():.3f}")
    print("confusion (rows gemini 0/1/2, cols glm 0/1/2):\n", conf)
    print("gemini marginal", conf.sum(1), " glm marginal", conf.sum(0))
    R = json.load(open(OUT / "rankings.json")); Q = list(judge.S["queries"])
    ARMS = ["open3d", "open3d_tight", "f3d", "three_ao", "three_noao", "three_white"]
    print(f"\n{'arm':13}{'prompt':12}{'nDCG gemini':>13}{'nDCG glm':>10}{'P@10 gem':>10}{'P@10 glm':>10}")
    rows = []
    for a in ARMS:
        for kind in ("production", "described"):
            ng = np.mean([ndcg(q, R[f"{a}|{kind}|{q}"], G) for q in Q]); nl = np.mean([ndcg(q, R[f"{a}|{kind}|{q}"], L) for q in Q])
            pg = np.mean([np.mean([G.get(f"{q}|{n}", {}).get("relevance") == 2 for n in R[f"{a}|{kind}|{q}"][:10]]) for q in Q])
            pl = np.mean([np.mean([L.get(f"{q}|{n}", {}).get("relevance") == 2 for n in R[f"{a}|{kind}|{q}"][:10]]) for q in Q])
            rows.append((a, kind, ng, nl)); print(f"{a:13}{kind:12}{ng:13.3f}{nl:10.3f}{pg:10.3f}{pl:10.3f}")
    from scipy.stats import spearmanr
    print("\nSpearman between judges over the 12 arm×prompt nDCG means:",
          round(spearmanr([r[2] for r in rows], [r[3] for r in rows]).statistic, 3))
    print("disagreements by 2 levels:")
    for k in keys:
        if abs(G[k]["relevance"] - L[k]["relevance"]) == 2:
            q, n = k.rsplit("|", 1)
            print(f"  {q[:24]:24} {judge.BY_N[int(n)]['key'][:36]:36} gemini {G[k]['relevance']} / glm {L[k]['relevance']}: {L[k]['reason'][:80]}")


if __name__ == "__main__":
    agree() if "--agree" in sys.argv else run()
