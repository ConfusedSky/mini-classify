"""Model or loose part, judged from the name alone: the draw-time regexes
against Jev and against a chat LLM, on the hand triage.

    python eval/part_judge.py --backend regex
    python eval/part_judge.py --backend jev
    python eval/part_judge.py --backend openrouter --model google/gemini-3.5-flash
    python eval/part_judge.py --backend jev --reuse     # rescore the last dump

Ground truth is `common.load_triage()`. Every backend sees the same thing a
person reading a directory listing would: the file's path under the
collection root and the other names in its folder, from the walk the draws
were taken from. No pixels, and no geometry.

The regex backend is `pose_label_expand`'s `LIKELY_PART` or `lettered_family`,
imported rather than copied, so its row is the filter as it runs today. The
report splits the unfiltered draws from `label_expand3`, which was drawn with
that filter on: its parts are the ones the regex missed by construction, so it
is where a judge can only add, never tie.

Jev and the chat model both go through OpenRouter, keyed by
`OPENROUTER_API_KEY` from the environment or the repo's `.env`. Answers land
in `OUT/part_judge-<tag>.json` with the raw reply, so threshold sweeps and new
metrics rescore with `--reuse` instead of paying again.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import (OUT, PART_CLASSES_FILE, REPO, TRIAGE_FILE, collection_root,
                    load_triage)
from pose_label_expand import LIKELY_PART, folder_index, lettered_family

from src import naming

OPENROUTER = "https://openrouter.ai/api/v1"
JEV_MODEL = "~typesafe/jev-latest"
MAX_SIBLINGS = 80

# The definitions the triage page asked a person to apply (labels/README.md).
# Nothing here describes a naming pattern: a judge told that `_L` means a part
# is the regex again, graded against the data the regex was tuned on.
PART_TRUE = ("One piece of something larger, printed on its own and glued to "
             "or placed with other pieces: a wing, a weapon, a hand, a wall "
             "section, a wheel, one slice of a modular building.")
PART_FALSE = ("A model in its own right: a figure, bust, creature, vehicle, "
              "building or scenery piece that stands alone, including the main "
              "body of a kit that the other pieces attach to.")
# The triage page's own question ("a model — label its up axis" against "a
# loose part — no upright"), which is what the labels answer. PART_* asks
# whether the file is a separate piece, and a saddle or a wall section is one
# while still standing one way up.
UPRIGHT_TRUE = ("A loose piece with no top or bottom of its own, that only "
                "makes sense attached to something: a hand, a weapon, a wing, "
                "a limb, a pipe, a pin.")
UPRIGHT_FALSE = ("Something that rests on a table one way up: a figure, a "
                 "bust, a creature, a vehicle, a building, a scenery or "
                 "terrain piece.")
KINDS = {
    "model": PART_FALSE,
    "accessory": "A loose piece that a figure wears or carries: a weapon, a "
                 "hand holding something, a shield, a backpack, a quiver, a "
                 "spell effect.",
    "component": "A loose piece of one assembly: a wall of a building, a limb "
                 "or wing of a creature, a numbered part of a machine.",
}


def api_key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    for line in (REPO / ".env").read_text().splitlines():
        name, _, value = line.partition("=")
        if name.strip().removeprefix("export ").strip() == "OPENROUTER_API_KEY":
            return value.strip().strip("\"'")
    raise SystemExit("OPENROUTER_API_KEY is not in the environment or .env")


def post(url, body, key, tries=4):
    """POST JSON, retrying rate limits and server errors with backoff."""
    req = urllib.request.Request(url, json.dumps(body).encode(), {
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
        except urllib.error.URLError:
            if attempt == tries - 1:
                raise
        time.sleep(2 ** attempt)


def state_of(path, root, siblings):
    """What a judge is shown: where the file sits and what sits beside it."""
    rel = path.relative_to(root)
    names = sorted(siblings.get(path.parent, []))
    if len(names) > MAX_SIBLINGS:      # keep the file itself in a long folder
        names = sorted(set(names[:MAX_SIBLINGS - 1]) | {path.stem})
    return {"library": "a personal collection of 3D-printable tabletop "
                       "miniatures and terrain, as bought from many sculptors",
            "folders": list(rel.parent.parts), "file": path.stem,
            "folder_files": names}


def judge_regex(row, root, siblings):
    p = row["path"]
    hit = bool(LIKELY_PART.search(p.stem)) or lettered_family(p.stem, p.parent, siblings)
    return {"p": float(hit), "kind": None, "cost": 0.0}


def judge_jev(row, root, siblings, key, model):
    body = {"model": model, "state": state_of(row["path"], root, siblings),
            "questions": {
                "part": {"type": "noul",
                         "instructions": "`file` is a loose piece of something "
                                         "larger, not a model in its own right",
                         "criteria": {"true": PART_TRUE, "false": PART_FALSE}},
                "no_upright": {"type": "noul",
                               "instructions": "`file` has no upright a person "
                                               "would commit to when printed alone",
                               "criteria": {"true": UPRIGHT_TRUE,
                                            "false": UPRIGHT_FALSE}},
                "kind": {"type": "choice",
                         "instructions": "What is `file`?", "criteria": KINDS}}}
    r = post(f"{OPENROUTER}/systemone", body, key)
    a = r["answers"]
    return {"scores": {"part": a["part"]["noul"],
                       "no_upright": a["no_upright"]["noul"]},
            "kind": a["kind"]["choice"],
            "kind_p": a["kind"]["probabilities"],
            "cost": r.get("usage", {}).get("cost", 0.0), "model": r.get("model")}


def judge_chat(row, root, siblings, key, model):
    kinds = "\n".join(f"- {k}: {v}" for k, v in KINDS.items())
    prompt = (f"Here is one file from a 3D-printing library, with the other "
              f"files in its folder:\n{json.dumps(state_of(row['path'], root, siblings))}"
              f"\n\nClassify `file` as exactly one of:\n{kinds}\n\n"
              'Answer with JSON only: {"kind": "model" | "accessory" | "component"}')
    r = post(f"{OPENROUTER}/chat/completions",
             {"model": model, "temperature": 0, "usage": {"include": True},
              "response_format": {"type": "json_object"},
              "messages": [{"role": "user", "content": prompt}]}, key)
    text = r["choices"][0]["message"].get("content") or ""
    try:
        kind = json.loads(text[text.index("{"):text.rindex("}") + 1])["kind"]
    except (ValueError, KeyError, TypeError):
        kind = None
    kind = kind if kind in KINDS else None
    return {"p": None if kind is None else float(kind != "model"), "kind": kind,
            "reply": text, "cost": r.get("usage", {}).get("cost", 0.0)}


def walk_siblings():
    """{folder: [stem, ...]} from the walk the triage draws were taken from,
    with today's vocabulary re-applied the way `pose_label_expand.render` does."""
    names = {d["walk"] for d in json.loads(TRIAGE_FILE.read_text())["draws"].values()}
    if len(names) != 1:
        raise SystemExit(f"triage draws span several walks: {sorted(names)}")
    walk = next(REPO.glob(f"embed-cache*/{names.pop()}"), None)
    if walk is None:
        raise SystemExit("the triage draws' walk file is not under embed-cache*/")
    files = [Path(p) for p in json.loads(walk.read_text())["files"]]
    return folder_index([f for f in files if not any(naming.skip(s) for s in f.parts)])


def sweep(parts, models, hit, ps):
    """For each budget of wrongly cut models, the lowest threshold that stays
    inside it and the parts it catches. The threshold is picked on the rows it
    is scored on, so these rows are an upper bound, not an operating point."""
    for budget in (0, 1, 2, 5):
        ok = [t for t in ps if sum(hit(r, t) for r in models) <= budget]
        if ok:
            t = min(ok)
            print(f"    <= {budget} models cut: threshold {t:.3f}, parts caught "
                  f"{sum(hit(r, t) for r in parts)}/{len(parts)}")


def report(rows, regex_rows=None):
    """Recall on parts and models wrongly cut, per population, at p >= 0.5, plus
    the budget sweep alone and OR'd with the regex."""
    pops = [("all draws", rows),
            ("unfiltered draws", [r for r in rows if not r["filtered"]]),
            ("label_expand3 (regex-filtered)", [r for r in rows if r["filtered"]])]
    regex = {r["path"]: r["p"] >= 0.5 for r in regex_rows or []}
    for name, pop in pops:
        have = [r for r in pop if r["p"] is not None]
        parts = [r for r in have if r["part"]]
        models = [r for r in have if not r["part"]]
        caught = sum(r["p"] >= 0.5 for r in parts)
        cut = sum(r["p"] >= 0.5 for r in models)
        print(f"\n{name}: {len(pop)} rows, {len(pop) - len(have)} unanswered")
        print(f"  @0.5  parts caught {caught}/{len(parts)}  models cut {cut}/{len(models)}")
        ps = sorted({r["p"] for r in have})
        if len(ps) <= 2:
            continue
        print("  alone:")
        sweep(parts, models, lambda r, t: r["p"] >= t, ps)
        if regex:
            print("  with regex OR:")
            sweep(parts, models, lambda r, t: regex[r["path"]] or r["p"] >= t, ps)
    classes = json.loads(PART_CLASSES_FILE.read_text())["picks"]
    kinds = [(classes[str(r["path"])[1:]], r["kind"]) for r in rows
             if r["kind"] and classes.get(str(r["path"])[1:]) in ("accessory", "component")]
    if kinds:
        print(f"\npart class (accessory/component) agreement: "
              f"{sum(a == b for a, b in kinds)}/{len(kinds)}")
    cost = sum(r.get("cost") or 0 for r in rows)
    if cost:
        print(f"cost ${cost:.4f} for {len(rows)} files")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", choices=["regex", "jev", "openrouter"], required=True)
    ap.add_argument("--model", help=f"OpenRouter model id (jev default {JEV_MODEL})")
    ap.add_argument("--score", choices=["part", "no_upright"], default="no_upright",
                    help="which jev noul to score (both are asked and saved)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="judge only the first N rows")
    ap.add_argument("--reuse", action="store_true",
                    help="rescore the saved dump instead of calling the API")
    args = ap.parse_args()
    if args.backend == "openrouter" and not args.model:
        ap.error("--backend openrouter needs --model")
    model = args.model or (JEV_MODEL if args.backend == "jev" else None)
    tag = args.backend + (f"-{model.replace('/', '_').replace('~', '')}" if model else "")
    dump = OUT / f"part_judge-{tag}.json"

    siblings = walk_siblings()
    truth = load_triage()[:args.limit]
    root = collection_root()
    regex_rows = [{**r, **judge_regex(r, root, siblings)} for r in truth]

    if args.backend == "regex":
        rows = regex_rows
    elif args.reuse:
        rows = [{**r, "path": Path(r["path"])} for r in json.loads(dump.read_text())["rows"]]
    else:
        key = api_key()
        fn = judge_jev if args.backend == "jev" else judge_chat
        t0 = time.monotonic()

        def one(row):
            try:
                return {**row, **fn(row, root, siblings, key, model)}
            except Exception as e:           # one bad call is a gap, not a lost run
                return {**row, "p": None, "kind": None, "error": repr(e)}
        with ThreadPoolExecutor(args.workers) as pool:
            rows = list(pool.map(one, truth))
        wall = time.monotonic() - t0
        dump.write_text(json.dumps({"backend": args.backend, "model": model,
                                    "wall_s": round(wall, 1),
                                    "rows": [{**r, "path": str(r["path"])} for r in rows]},
                                   indent=1))
        errors = [r for r in rows if r.get("error")]
        print(f"{len(rows)} judged in {wall:.1f}s, {len(errors)} errors -> {dump}")
        for r in errors[:3]:
            print("  ", r["path"].name, r["error"][:160])

    for r in rows:
        if "scores" in r:
            r["p"] = r["scores"][args.score]
    print(f"backend {tag}" + (f", noul {args.score}" if args.backend == "jev" else ""))
    report(rows, None if args.backend == "regex" else regex_rows)


if __name__ == "__main__":
    main()
