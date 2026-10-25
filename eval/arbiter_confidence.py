"""Does the arbiter's confidence say when its answer is wrong? Two kinds of
confidence, three presentations, three vendors, on the production 512 px pose
sheets.

Confidence arms:
  logprob  the production prompt and schema, unchanged; confidence is the
           probability of the answer token. Only where the endpoint returns
           logprobs: gemini-2.5-flash/-pro (Vertex refuses them for 3.5/3.6
           flash) and GLM pinned to Parasail (DeepInfra, production's pin,
           does not return them; Wafer returns a broken top-k list).
  self     the schema gains a `confidence` field the model fills in. Every
           model. Claude only has this arm: the API exposes no logprobs.
Presentations:
  sheet    the production contact sheet, one call, pick a numbered tile.
  solo     the six tiles as separate captioned images, numerals painted out,
           one call, pick one.
  tile     one call per tile, "is this upright?"; the pick is the tile most
           likely upright and the confidence is that probability.

  .venv/bin/python eval/arbiter_confidence.py --labels published
  .venv/bin/python eval/arbiter_confidence.py --labels gated --models claude-opus-5-5 --present tile
  .venv/bin/python eval/arbiter_confidence.py --report-only

Answers accumulate in out/arbiter_confidence-<tag>.json, keyed by model, arm
and presentation per stem; a rerun skips any cell answered under any tag, and
the report merges every tag. Claude goes through the `claude` CLI on the
subscription (OAuth, no API key), isolated from settings, tools and CLAUDE.md;
it alone gets a one-line system prompt (without one the CLI sends its own
agentic one) and runs at the CLI's temperature, which cannot be set. Gemini needs gcloud ADC,
GLM ~/.config/openrouter/key. Labels and the ensemble baseline come from
out/pose_baseline.json (`pose_baseline.py`, embed-cache512's backbone).
"""
import argparse, base64, itertools, json, math, os, subprocess, threading, time
import urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import AX, OUT, load_baselines, load_labels  # REPO on sys.path

import decisions_vlm                    # tiles(), calibration()
import gemini_vlm                       # PROJECT/HOST/LOCATION, PRICES
from src import pose

SHEETS = OUT / "sheets512"
GEMINI = ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.5-pro"]
GLM = "z-ai/glm-5.3-flash"
GLM_PROVIDER = "Parasail"
CLAUDE = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]
LOGPROB_MODELS = {"gemini-2.5-flash", "gemini-2.5-pro", GLM}
N = 6

PICK_SELF = (
    "Each numbered tile shows the same 3D model in a different orientation. "
    "Which tile shows the model standing upright, the way it would sit on a "
    "table? Also give your confidence, from 0 to 1, that your answer is correct. "
    'Answer with JSON only: {"tile": <number>, "confidence": <number>}')
TILE_LOGPROB = (
    "This image shows a 3D model. Is it standing upright, the way it would sit "
    'on a table? Answer with JSON only: {"upright": <true or false>}')
TILE_SELF = (
    "This image shows a 3D model. Is it standing upright, the way it would sit "
    "on a table? Also give your confidence, from 0 to 1, that your answer is "
    'correct. Answer with JSON only: {"upright": <true or false>, "confidence": <number>}')


def schema(present, arm):
    key = {"upright": {"type": "boolean"}} if present == "tile" else {"tile": {"type": "integer"}}
    if arm == "self":
        key["confidence"] = {"type": "number"}
    return {"type": "object", "properties": key, "required": list(key),
            "additionalProperties": False}


def prompt(present, arm):
    if present == "tile":
        return TILE_SELF if arm == "self" else TILE_LOGPROB
    return PICK_SELF if arm == "self" else pose.UP_PROMPT


def images(stem, present):
    """[(caption or None, png bytes)] in the order the request sends them."""
    sheet = SHEETS / f"{stem}.png"
    if present == "sheet":
        return [(None, sheet.read_bytes())]
    tiles = decisions_vlm.tiles(sheet, N, blank=True)
    if present == "solo":
        return [(f"Tile {i}:", png) for i, png in enumerate(tiles, 1)]
    return [(None, png) for png in tiles]


# --- reading a confidence out of token logprobs -----------------------------

def answer_dist(positions, values):
    """{value: probability} for the answer token, from [(chosen, {token:
    logprob})] per position. The answer often follows a lone space token
    whose own top-k already carries the answer with the space dropped, so
    that mass is added to the conditional one rather than lost.

    The *last* matching token is the answer: the logprob arm's JSON holds
    nothing after it, while reasoning text left before it can mention a tile."""
    for i in reversed(range(len(positions))):
        tok, top = positions[i]
        if tok.strip().lower() in values:
            p = {v: 0.0 for v in values}
            for t, lp in top.items():
                if t.strip().lower() in p:
                    p[t.strip().lower()] += math.exp(lp)
            if i and not positions[i - 1][0].strip():
                prev = positions[i - 1][1]
                lead = math.exp(prev.get(positions[i - 1][0], 0.0))
                p = {v: q * lead for v, q in p.items()}
                for t, lp in prev.items():
                    if t.strip().lower() in p and t.strip():
                        p[t.strip().lower()] += math.exp(lp)
            z = sum(p.values())
            return {v: q / z for v, q in p.items()} if z else None
    return None


def gemini_positions(cand):
    lp = cand.get("logprobsResult") or {}
    return [(c["token"], {t["token"]: t["logProbability"] for t in top.get("candidates", [])})
            for c, top in zip(lp.get("chosenCandidates", []), lp.get("topCandidates", []))]


def openrouter_positions(choice):
    content = (choice.get("logprobs") or {}).get("content") or []
    # Parasail streams the reasoning's tail first; only the JSON after it counts
    start = max((i + 1 for i, t in enumerate(content) if t["token"] == "</think>"), default=0)
    return [(t["token"], {x["token"]: x["logprob"] for x in t.get("top_logprobs", [])})
            for t in content[start:]]


# --- backends: each returns (text, positions or None, usage) ----------------

_tok = {"t": 0.0, "v": None}
_tok_lock = threading.Lock()


def gcloud_token():
    with _tok_lock:
        if time.time() - _tok["t"] > 1500:
            _tok["v"], _tok["t"] = gemini_vlm.token(), time.time()
        return _tok["v"]


def call_gemini(model, imgs, text, sch, logprobs):
    parts = []
    for cap, png in imgs:
        if cap:
            parts.append({"text": cap})
        parts.append({"inlineData": {"mimeType": "image/png", "data": base64.b64encode(png).decode()}})
    cfg = {"temperature": 0, "responseMimeType": "application/json", "responseSchema":
           {k: v for k, v in sch.items() if k != "additionalProperties"}}
    if logprobs:
        cfg.update(responseLogprobs=True, logprobs=10)
    body = json.dumps({"contents": [{"role": "user", "parts": parts + [{"text": text}]}],
                       "generationConfig": cfg}).encode()
    url = (f"https://{gemini_vlm.HOST}/v1/projects/{gemini_vlm.PROJECT}/locations/"
           f"{gemini_vlm.LOCATION}/publishers/google/models/{model}:generateContent")
    t0 = time.time()
    req = urllib.request.Request(url, body, {"Authorization": f"Bearer {gcloud_token()}",
                                             "Content-Type": "application/json"})
    # a normal call returns in 5-30 s; past 120 s it has hung, and the retry is
    # the same request, so waiting longer buys nothing
    d = json.loads(urllib.request.urlopen(req, timeout=120).read())
    u = d.get("usageMetadata", {})
    cand = (d.get("candidates") or [{}])[0]
    out = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []))
    return out, gemini_positions(cand) if logprobs else None, {
        "prompt": u.get("promptTokenCount", 0), "thoughts": u.get("thoughtsTokenCount", 0),
        "output": u.get("candidatesTokenCount", 0), "secs": time.time() - t0}


def call_glm(imgs, text, sch, logprobs):
    content = []
    for cap, png in imgs:
        if cap:
            content.append({"type": "text", "text": cap})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(png).decode()}})
    body = {"model": GLM, "temperature": 0, "reasoning": {"effort": "low"},
            "usage": {"include": True},
            "provider": {"order": [GLM_PROVIDER], "allow_fallbacks": False},
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "answer", "strict": True, "schema": sch}},
            "messages": [{"role": "user", "content": content + [{"type": "text", "text": text}]}]}
    if logprobs:
        body.update(logprobs=True, top_logprobs=10)
    key = (Path.home() / ".config/openrouter/key").read_text().strip()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=300).read())
    if "error" in d:
        raise RuntimeError(f"openrouter: {d['error']}")
    u, ch = d.get("usage", {}), d["choices"][0]
    think = (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    return ch["message"]["content"], openrouter_positions(ch) if logprobs else None, {
        "prompt": u.get("prompt_tokens", 0), "thoughts": think,
        "output": u.get("completion_tokens", 0) - think, "secs": time.time() - t0,
        "cost": u.get("cost")}


def call_claude(model, imgs, text, sch):
    """One `claude -p` turn with the images inline. `--setting-sources ''` and
    no tools keep the user's hooks, CLAUDE.md and plugins' context out of the
    judgment; `--bare` would too, but it refuses OAuth, i.e. the subscription."""
    content = []
    for cap, png in imgs:
        if cap:
            content.append({"type": "text", "text": cap})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                    "data": base64.b64encode(png).decode()}})
    msg = {"type": "user", "message": {"role": "user", "content": content + [{"type": "text", "text": text}]}}
    t0 = time.time()
    out = subprocess.run(
        ["claude", "-p", "--model", model, "--input-format", "stream-json",
         "--output-format", "stream-json", "--verbose", "--json-schema", json.dumps(sch),
         "--tools", "", "--setting-sources", "", "--strict-mcp-config",
         "--no-session-persistence", "--system-prompt",
         "You judge the orientation of renders of 3D-printable models."],
        input=json.dumps(msg) + "\n", capture_output=True, text=True, timeout=300,
        cwd=OUT)
    lines = [json.loads(l) for l in out.stdout.splitlines() if l.startswith("{")]
    res = next((x for x in lines if x.get("type") == "result"), None)
    if out.returncode or res is None or res.get("is_error"):
        raise RuntimeError(f"claude exit {out.returncode}: {(out.stderr or out.stdout)[-200:]}")
    u = res.get("usage", {})
    think = (u.get("output_tokens_details") or {}).get("thinking_tokens", 0)
    return json.dumps(res.get("structured_output") or {}), None, {
        "prompt": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                  + u.get("cache_read_input_tokens", 0),
        "thoughts": think, "output": u.get("output_tokens", 0) - think,
        "secs": time.time() - t0, "cost": res.get("total_cost_usd")}


def call(model, imgs, text, sch, logprobs):
    if model.startswith("gemini"):
        return call_gemini(model, imgs, text, sch, logprobs)
    if model == GLM:
        return call_glm(imgs, text, sch, logprobs)
    return call_claude(model, imgs, text, sch)


def ask_once(model, arm, present, imgs, usage):
    """One request, retried on transient failure and on an unparseable answer.
    Returns the parsed answer dict plus the answer-token distribution, or None."""
    sch, text, lp = schema(present, arm), prompt(present, arm), arm == "logprob"
    values = ["true", "false"] if present == "tile" else [str(i) for i in range(1, N + 1)]
    bad_answers = 0
    for attempt in range(6):
        try:
            out, positions, u = call(model, imgs, text, sch, lp)
        except urllib.error.HTTPError as e:
            msg = e.read()[:160]
            print(f"  {model} HTTP {e.code} (attempt {attempt + 1}): {msg}", flush=True)
            if e.code not in (429, 500, 502, 503, 504):
                return None
            time.sleep(min(60, 4 * 2 ** attempt))
            continue
        except Exception as e:      # network, CLI failure, a 200 carrying an error
            print(f"  {model} {arm}/{present} (attempt {attempt + 1}): {str(e)[:160]}", flush=True)
            time.sleep(min(60, 4 * 2 ** attempt))
            continue
        usage.append(u)
        try:
            a = json.loads(out[out.index("{"):out.rindex("}") + 1])
            if arm == "self" and not 0 <= float(a["confidence"]) <= 1:
                raise ValueError(f"confidence off the 0-1 scale: {a['confidence']!r}")
            dist = answer_dist(positions, values) if lp else None
            if lp and dist is None:
                raise ValueError("no answer-token logprobs")
            return a, dist
        except (ValueError, KeyError, TypeError) as e:
            print(f"  {model} {arm}/{present} unusable answer: {str(e)[:160]}", flush=True)
            bad_answers += 1
            if bad_answers == 2:
                return None
    print(f"  {model} {arm}/{present}: gave up after 6 attempts", flush=True)
    return None


def ask(model, arm, present, stem, usage):
    """{"pick": axis, "conf": float, "probs": [6]} or None."""
    imgs = images(stem, present)
    if present == "tile":
        ps = []
        for img in imgs:
            r = ask_once(model, arm, present, [img], usage)
            if r is None or not isinstance(r[0].get("upright"), bool):
                return None
            a, dist = r
            if arm == "logprob":
                ps.append(dist["true"])
            else:
                c = min(1.0, max(0.0, float(a["confidence"])))
                ps.append(c if a["upright"] else 1 - c)
        best = max(range(N), key=ps.__getitem__)
        return {"pick": AX[best], "conf": ps[best], "probs": ps}
    r = ask_once(model, arm, present, imgs, usage)
    if r is None:
        return None
    a, dist = r
    tile = a.get("tile")
    if not isinstance(tile, int) or isinstance(tile, bool) or not 1 <= tile <= N:
        return None
    if arm == "logprob":
        probs = [dist[str(i)] for i in range(1, N + 1)]
        return {"pick": AX[tile - 1], "conf": probs[tile - 1], "probs": probs}
    return {"pick": AX[tile - 1], "conf": min(1.0, max(0.0, float(a["confidence"]))), "probs": None}


# --- the run ---------------------------------------------------------------

def load_items(which):
    base = json.loads((OUT / "pose_baseline.json").read_text())["baselines"]
    published = load_baselines()
    items = []
    for l in load_labels():
        b = base.get(l["stem"])
        if b is None or l["set"] in ("hard", "reclaim"):     # labels/README: never pooled
            continue
        if which == "published" and l["stem"] not in published:
            continue
        if which == "gated" and not b["needs_arbiter"]:
            continue
        items.append({"stem": l["stem"], "set": l["set"], "up": l["up"],
                      "geo": b["geometry"], "ens": b["ensemble"], "arb": b["needs_arbiter"]})
    missing = [it["stem"] for it in items if not (SHEETS / f"{it['stem']}.png").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} sheets missing from {SHEETS} (run common.build_sheets): {missing[:5]}")
    return items


def cells(models, arms, presents):
    return [(m, a, p) for m in models for a in arms for p in presents
            if a == "self" or m in LOGPROB_MODELS]


def results_file(tag):
    return OUT / f"arbiter_confidence-{tag}.json"


def load_saved(paths):
    saved = {"answers": {}, "usage": {}}
    for p in paths:
        d = json.loads(p.read_text())
        for stem, per in d["answers"].items():
            saved["answers"].setdefault(stem, {}).update(per)
        for cell, calls in d["usage"].items():
            saved["usage"].setdefault(cell, []).extend(calls)
    return saved


def save(saved, path):
    # atomic, so an interrupt mid-write cannot cost the answers already paid for
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(saved, indent=1))
    os.replace(tmp, path)


def run(items, todo, workers, saved, path, answered):
    """`answered` is every tag's answers, so a cell paid for under another
    tag is not paid for again; `saved` is this tag's file alone."""
    lock = threading.Lock()
    for model, arm, present in todo:
        cell = f"{model}|{arm}|{present}"
        pending = [it for it in items if cell not in answered.get(it["stem"], {})]
        if not pending:
            continue
        t0, done, u = time.time(), itertools.count(1), saved["usage"].setdefault(cell, [])
        print(f"\n{cell}: {len(pending)} to ask", flush=True)

        def one(it):
            r = ask(model, arm, present, it["stem"], u)
            with lock:
                if r is not None:
                    saved["answers"].setdefault(it["stem"], {})[cell] = r
                    if next(done) % 10 == 0:            # cheap checkpoint against a crash
                        save(saved, path)
                print(f"  {it['stem'][:34]:34} {(r or {}).get('pick', '--'):>3} "
                      f"{'' if r is None else format(r['conf'], '.2f'):>5} "
                      f"gold {it['up']}  {time.time() - t0:5.0f}s", flush=True)

        w = min(workers, 4) if model.startswith("claude") else workers
        try:
            with ThreadPoolExecutor(max_workers=w) as ex:
                list(ex.map(one, pending))
        finally:
            with lock:
                save(saved, path)


def auroc(right, wrong):
    if not right or not wrong:
        return None
    return sum((r > w) + 0.5 * (r == w) for r in right for w in wrong) / (len(right) * len(wrong))


def report(items, saved, detail):
    stems = {it["stem"] for it in items}
    cols = sorted({c for s in stems for c in saved["answers"].get(s, {})})
    print(f"\n{len(items)} models, {sum(it['arb'] for it in items)} gate-fired | "
          f"ensemble alone {sum(it['ens'] == it['up'] for it in items)}/{len(items)}")
    hdr = (f"{'cell':40} {'n':>4} {'acc':>8} {'net':>4} {'pipeline':>9} "
           f"{'med R':>6} {'med W':>6} {'AUROC':>6} {'ties':>5} {'think':>6} {'secs':>5}")
    print(hdr); print("-" * len(hdr))
    for c in cols:
        rows, ties = [], 0
        for it in items:
            a = saved["answers"].get(it["stem"], {}).get(c)
            if a is not None:
                rows.append(dict(it, **{c: a["pick"], f"{c}_conf": a["conf"]}))
                # a tied top goes to the first candidate, +Z, joint-commonest label with +Y;
                # sheet and solo picks are the sampled token, so they cannot tie
                ties += c.endswith("|tile") and sorted(a["probs"])[-2] >= a["conf"] - 1e-6
        if not rows:
            continue
        right = sorted(r[f"{c}_conf"] for r in rows if r[c] == r["up"])
        wrong = sorted(r[f"{c}_conf"] for r in rows if r[c] != r["up"])
        net = sum((r[c] == r["up"]) - (r["ens"] == r["up"]) for r in rows if r["arb"])
        pipe = sum((r[c] if r["arb"] else r["ens"]) == r["up"] for r in rows)
        med = lambda s: f"{s[len(s) // 2]:.2f}" if s else "--"
        auc = auroc(right, wrong)
        us = saved["usage"].get(c) or [{"thoughts": 0, "secs": 0}]
        calls = 6 if c.endswith("|tile") else 1
        print(f"{c:40} {len(rows):>4} {len(right):>4}/{len(rows):<3} {net:>+4} "
              f"{pipe:>5}/{len(rows):<3} {med(right):>6} {med(wrong):>6} "
              f"{'--' if auc is None else format(auc, '.2f'):>6} {ties:>5} "
              f"{calls * sum(u['thoughts'] for u in us) / len(us):>6.0f} "
              f"{calls * sum(u['secs'] for u in us) / len(us):>5.1f}")
        if detail:
            decisions_vlm.calibration(rows, c)
    print("\nacc: standalone on models it answered. net: rescued - broke as the arbiter "
          "tier, on gate-fired models.\nmed R/W: median confidence when right/wrong. "
          "ties: answers whose top probability was shared, so the pick was list order.\n"
          "think and secs: per arbitration (a tile cell is 6 calls).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", choices=["published", "poolable", "gated"], default="published")
    ap.add_argument("--models", default=",".join(GEMINI + [GLM] + CLAUDE))
    ap.add_argument("--arms", default="logprob,self")
    ap.add_argument("--present", default="sheet,solo,tile")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="main",
                    help="results file suffix; run backends in parallel under different tags, "
                         "the report merges every arbiter_confidence-*.json")
    ap.add_argument("--limit", type=int, default=None, help="first N models, for a smoke run")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--detail", action="store_true", help="per-cell gate tables")
    args = ap.parse_args()
    items = load_items(args.labels)[:args.limit]
    if not args.report_only:
        path = results_file(args.tag)
        saved = load_saved([path] if path.exists() else [])
        answered = load_saved(sorted(OUT.glob("arbiter_confidence-*.json")))["answers"]
        todo = cells(args.models.split(","), args.arms.split(","), args.present.split(","))
        print(f"{len(items)} models ({args.labels}) x {len(todo)} cells -> {path.name}")
        run(items, todo, args.workers, saved, path, answered)
    report(items, load_saved(sorted(OUT.glob("arbiter_confidence-*.json"))), args.detail)


if __name__ == "__main__":
    main()
