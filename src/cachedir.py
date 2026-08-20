"""Where things live under `--cache-dir`, and which run produced them.

Everything a run derives from the collection lives under one directory, so the
cache is one thing to pass around rather than two that have to be kept in step:
the embeddings, the pose cache, the walk lists and the debug renders are all
rebuildable, and all worthless against a different `--cache-dir`. This module
owns that layout — the subdirectory names, the key built for each entry, the
`cache_version` stamp that says the keys are readable at all, and the
`run-params.json` manifest that lets every tool agree with the classify run
that wrote the cache without retyping its flags.

It lived in `classify_stls.py` until the eval-debt cleanup, which is why the
CLI, `cluster_models.py`, `test_categories.py`, `migrate_cache_keys.py`, the
eval harnesses and the tests all used to import a *script* to find their way
around a cache. They import this instead; the CLI is now one of the consumers,
not the home.

**Import rules** (docs/actor-refactor/interfaces.md's table): stdlib,
`naming`, and `src.identity` — no torch, no open3d, no numpy, no PIL. Two
reasons, both load-bearing:

* the CLI imports this at module scope, and `mp.get_context("spawn")`
  re-imports the CLI as `__mp_main__` in the render child, so anything here
  is imported in the child too — where a torch import is exactly what the
  child-side rule forbids;
* the read-only tools (`cluster_models.py`) must be able to find their way
  around a cache without loading a model.

The rule is about this module's *callers*, not just its module scope — a
deferred import still lands in whoever calls the function. That is why
`--model`'s default is `src.identity`'s and not the Embedder's: importing it
from `src.embedder` inside `add_cache_args` kept torch out of this file and
put it into every tool that built a parser, `cluster_models.py` and
`migrate_cache_keys.py` included (836 modules and ~0.9 s, measured
2026-08-18). Nothing here imports outside the rule now, at module scope or
inside a function.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from src.naming import SKIP_TAGS, skip
from src import identity
from src.identity import (DEFAULT_ELEVATIONS, DEFAULT_MODEL,
                          cache_key_from_identity)

# Cache layout.
EMBEDS_SUBDIR = "embeds"
RENDERS_SUBDIR = "renders"


def view_config(args):
    """The token keying `front_view` entries in the pose cache.

    front_view is an index into this run's view list, and an index cached at 8
    views is meaningless at 4 — or silently wrong at the same count with
    different elevations — so the pose cache stores it per view config. Same
    elevation formatting as the embedding key, so the two never disagree about
    what one config is; `render_subdir` below names its renders directory with
    it for the same reason.

    Lives here rather than in `src/done.py`, which computes with it: it is a
    piece of cache identity, and `done` owns torch — leaving it there forced
    the CLI and `cluster_models.py` to reach a pure string function through a
    module that loads SigLIP."""
    elev = ",".join(f"{e:g}" for e in args.elevations)
    return f"{args.views}v-e{elev}"


def render_subdir(args):
    """Renders live under the camera config that produced them.

    A filename carries only stem and view index, but cache_key covers render
    size, views and elevations — so without this a rerun at a different size
    leaves the previous config's images in place and the contact sheets stop
    describing what was actually classified."""
    return f"{args.render_size}px-{view_config(args)}"


def embeds_dir(cache_dir):
    """Where the per-file .npy embeddings live, or None with caching off."""
    return Path(cache_dir) / EMBEDS_SUBDIR if cache_dir else None


def renders_dir(cache_dir, args):
    """Where --save-renders writes, under the camera config that produced them.

    Derived rather than passed: a renders directory paired with the wrong cache
    shows one run's images beside another run's embeddings, and the two have no
    way to notice."""
    return Path(cache_dir) / RENDERS_SUBDIR / render_subdir(args) if cache_dir else None


def render_index(rdir):
    """Map '<render_key>_view<i>' to the saved render, from one listing of the dir.

    Extension-agnostic on purpose: a directory may hold PNGs written before
    --render-format existed alongside new JPEGs, and switching format must
    neither re-render them nor hide them from the tools. Newest wins when a view
    exists in both. One listing rather than a glob per view — the lookup runs
    n_views times per model, and real stems contain '(' and '['."""
    if rdir is None or not Path(rdir).is_dir():
        return {}
    files = sorted((p for p in Path(rdir).iterdir() if p.is_file()),
                   key=lambda p: p.stat().st_mtime)
    return {p.stem: p for p in files}


# Writing the renders is `Renderer.save_renders` and only that (F-4): the
# pixels are already in the child's memory (data_structures Q2). The names it
# writes are the names `render_index` above parses.


def find_stls(root):
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune skipped directories before descending — big win on slow drives
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and not skip(d)]
        for name in filenames:
            if (not name.startswith(".") and name.lower().endswith(".stl")
                    and not skip(name)):
                found.append(Path(dirpath) / name)
    return sorted(found)


def load_file_list(inp, cache_dir, rescan=False):
    """Directory walk with cached file list (see --rescan)."""
    walk_cache = None
    if cache_dir:
        walk_id = hashlib.sha1(f"{inp.resolve()}|{SKIP_TAGS}|unsupported-ok".encode()).hexdigest()
        walk_cache = Path(cache_dir) / f"walk-{walk_id}.json"
    if walk_cache and walk_cache.exists() and not rescan:
        saved = json.loads(walk_cache.read_text())
        files = [Path(p) for p in saved["files"]]
        # one `exists()` per entry, not two: the second pass was only counting
        # what the first had already found (review, 2026-08-19)
        present = [f for f in files if f.exists()]
        gone = len(files) - len(present)
        files = present
        age_days = (time.time() - saved["scanned"]) / 86400
        note = f", {gone} vanished since scan" if gone else ""
        print(f"using cached file list: {len(files)} files, scanned "
              f"{age_days:.1f} days ago{note} (--rescan to refresh)")
        return files
    files = find_stls(inp)
    if walk_cache:
        walk_cache.parent.mkdir(parents=True, exist_ok=True)
        # temp + os.replace, the same treatment `Done.flush` gives the pose
        # cache. It matters more since `POST /reload {"rescan": true}` writes
        # this from a request handler while `classify_stls.py` may be writing
        # it too: a torn file is not merely a stale list, it is a
        # JSONDecodeError on every subsequent read (review, 2026-08-19).
        tmp = walk_cache.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(
            {"scanned": time.time(), "files": [str(f) for f in files]}))
        os.replace(tmp, walk_cache)
    return files


def cache_key(f, args, up_token, root):
    # The path is relative to the collection root (identity.py) so the library
    # can change drives without re-embedding everything.
    stat = f.stat()
    return cache_key_from_identity(
        f"{identity.rel_path(f, root)}|{identity.mtime_key(stat)}|{stat.st_size}",
        args, up_token)


def total_views(args):
    return args.views * len(args.elevations)


def parse_elevations(text):
    """Comma-separated camera elevations in degrees: '20' or '20,-10,55'."""
    if isinstance(text, list):  # already parsed (came from the run manifest)
        return text
    try:
        elevs = [float(v) for v in text.split(",") if v.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a list of numbers: {text!r}")
    if not elevs:
        raise argparse.ArgumentTypeError("need at least one elevation")
    if any(abs(e) > 90 for e in elevs):
        # ±90 is straight down / straight up; the renderer carries 'up' around
        # the orbit so the poles are ordinary cameras, not a degenerate look-at
        raise argparse.ArgumentTypeError("elevations must be within ±90 degrees")
    return elevs


def add_cache_args(parser, input_help):
    """Args that identify an embedding cache. Every tool reading the cache must
    agree on these, which is what the run manifest automates — declared in one
    place so a new one can't be added to the classifier and forgotten in the
    tools that read what it wrote."""
    parser.add_argument("input", nargs="?", help=input_help)
    parser.add_argument("--views", type=int, default=4,
                        help="azimuths per elevation ring (default 4)")
    parser.add_argument("--elevations", type=parse_elevations, default=DEFAULT_ELEVATIONS,
                        help="comma-separated camera elevations in degrees; each gets a "
                             "full ring of --views azimuths, so total views is the "
                             "product (default 20)")
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                        help="torch.compile the image forward: ~1.09x embed throughput for "
                             "~1e-03 embedding drift, which flips only coin-toss margins "
                             "(eval/compile_flips.py: 1 of 341, at margin 4.3e-06). "
                             "Compiled embeddings cache under their own keys, so the two "
                             "numeric regimes never mix")
    parser.add_argument("--up-axis", choices=["auto", "z", "y"], default="auto",
                        help="up axis of source meshes; auto detects the flat print base (default)")
    parser.add_argument("--cache-dir", default="embed-cache",
                        help="directory of cached per-file image embeddings; reruns with new "
                             "categories skip rendering/embedding entirely (set '' to disable)")
    # shared because every tool here walks the collection, and a stale list is
    # not merely slow — migrate_cache_keys drops entries for files it cannot see
    parser.add_argument("--rescan", action="store_true",
                        help="re-walk the input directory instead of using the cached file list")


RUN_PARAMS_FILE = "run-params.json"
# What a classify run records for the tools that read its cache. Keys are
# argparse dests; anything not declared by a given tool's parser is ignored.
# "pool" is deliberately absent: it is a scoring-time choice, not cache
# identity, and letting the classifier's afterthought default leak into
# test_categories overrode the REPL's own deliberate softmax default —
# querying happens there, so its default wins there.
RUN_PARAMS_KEYS = ("input", "views", "elevations", "render_size", "model",
                   "compile", "up_axis", "categories", "render_format",
                   "collection_root")

CACHE_META_FILE = "cache-meta.json"
# Bumped only when the *key scheme* changes incompatibly — never for
# byte-compatible additions like |compiled or |e:, which are designed to
# leave existing keys alone. That is why this is a hand-set integer and not a
# hash of the key format: an auto-derived stamp would fire on exactly the
# changes this repo makes carefully so it does not have to.
#   0 = unstamped (every cache from before the stamp existed): the up-token
#       elision, where deterministic poses keyed as the --up-axis string
#   1 = the up_str token (pose.embed_cache_token, review P2.3-B)
CACHE_VERSION = 1


def cache_version(cache_dir):
    """0 for any cache written before the stamp — i.e. every unstamped one."""
    p = Path(cache_dir) / CACHE_META_FILE
    return json.loads(p.read_text())["cache_version"] if p.exists() else 0


def stamp_cache_version(cache_dir):
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / CACHE_META_FILE).write_text(json.dumps({
        "cache_version": CACHE_VERSION,
        # informational only, never compared — see the CACHE_VERSION note
        "cache_key_format": "sha1(rel|mtime|size|views|render_size|up_token"
                            "|model|pv[|e:...][|compiled][|evN])",
    }, indent=2))


def require_cache_version(cache_dir):
    """Refuse a cache whose key scheme this code cannot read.

    A moved scheme does not error on its own — every lookup just misses, and
    the run silently re-renders and re-embeds the whole collection: hours,
    and real money once a pose entry is VLM-sourced. The stamp turns that
    into one line naming the fix. An empty cache is simply stamped current."""
    if not cache_dir:
        return
    v = cache_version(cache_dir)
    if v == CACHE_VERSION:
        return
    d = Path(cache_dir)
    # "populated" must include the pre-layout shape — root-level .npy with no
    # pose-cache.json or embeds/ (a forced --up-axis cache writes no pose
    # cache at all). Treating that as empty would stamp a genuinely
    # unmigrated cache as current, which is the exact failure this guard
    # exists to prevent (S2).
    if ((d / "pose-cache.json").exists() or (d / EMBEDS_SUBDIR).exists()
            or (d / RENDERS_SUBDIR).exists() or any(d.glob("*.npy"))):
        raise SystemExit(
            f"{cache_dir}: cache_version {v}, this code expects {CACHE_VERSION} — "
            f"every key would miss and the collection would re-embed from "
            f"scratch.\n  run: .venv/bin/python migrate_cache_keys.py "
            f"--cache-dir {cache_dir} --apply")
    stamp_cache_version(cache_dir)


def cache_root(inp, cache_dir, confirm=True, reanchor=False):
    """The root every cache key in `cache_dir` is taken relative to.

    Deliberately not just `collection_root(inp)`. The anchor belongs to the
    cache, not to the command line: running on one kit inside the library has
    to key the same way the whole-library run did, or the same file is indexed
    twice under two identities and re-rendered, re-embedded and re-arbitrated
    for the privilege.

    A mismatch stops to ask, because the two reasons for one are opposite: the
    library moved (re-key, free, everything still matches) or this cache
    belongs to a different collection (re-key, expensive, and the old entries
    are orphaned). Read-only tools pass confirm=False and only warn — they
    write nothing, and blocking a REPL on a prompt helps no one."""
    recorded = load_run_params(cache_dir).get("collection_root")
    root, note = identity.resolve_root(inp, recorded)
    if note == "subdir":
        print(f"cache keys stay anchored at {root} — this run is scoped to "
              f"{identity.collection_root(inp)}, but the cache is the library's")
    elif note in ("superdir", "mismatch"):
        if note == "superdir":
            why = (f"  every existing key is still valid under the wider root — it "
                   f"needs\n    {root and Path(recorded).relative_to(root)}/\n"
                   f"  on the front. migrate_cache_keys.py re-keys them; "
                   f"--reanchor without it orphans them.")
        else:
            gone = "" if Path(recorded).exists() else \
                " (which no longer exists, so this looks like the library moved)"
            why = f"  the recorded root{gone or ' still exists'}."
        print(f"\n  the cache in {cache_dir} was built against\n"
              f"    {recorded}\n  and you have asked for\n    {root}\n{why}")
        if reanchor:
            print("  --reanchor given; re-keying to the new root")
        elif not confirm:
            print("  read-only tool: using the root you asked for, which may miss "
                  "every cached entry")
        elif not sys.stdin.isatty():
            sys.exit("  refusing to re-key a cache without confirmation in a "
                     "non-interactive run — pass --reanchor if that is what you want")
        elif input("  re-key this cache to the new root? [y/N] ").strip().lower() \
                not in ("y", "yes"):
            sys.exit("  stopped; pass a path under the recorded root, or use a "
                     "separate --cache-dir for a different collection")
    return root


def load_run_params(cache_dir):
    if not cache_dir:
        return {}
    p = Path(cache_dir) / RUN_PARAMS_FILE
    return json.loads(p.read_text()) if p.exists() else {}


def save_run_params(args):
    """Record this run's parameters next to the cache it just wrote. Kept with
    the cache rather than in a committed config so the description cannot drift
    from what the embeddings actually are.

    Only `classify_stls.py` calls this — every other tool reads. It lives here
    anyway, with the readers: the three things a writer and a reader must agree
    on (`RUN_PARAMS_FILE`, `RUN_PARAMS_KEYS`, and the merge rule below) are all
    declared in this module, so a format change that touched only one side
    would be a bug the split made easy to write."""
    if not args.cache_dir:
        return
    params = {k: getattr(args, k, None) for k in RUN_PARAMS_KEYS}
    # a single-file run describes no collection — leave the recorded root alone
    params["input"] = str(Path(args.input).resolve()) if Path(args.input).is_dir() else None
    params = load_run_params(args.cache_dir) | {
        k: v for k, v in params.items() if v is not None}
    p = Path(args.cache_dir) / RUN_PARAMS_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(params, indent=2))


def apply_run_params(parser):
    """parse_args(), with defaults filled in from the last classify run.
    Explicit command-line values still win — set_defaults only moves the
    fallback."""
    known, _ = parser.parse_known_args()
    params = load_run_params(getattr(known, "cache_dir", None))
    dests = {a.dest for a in parser._actions}
    # RUN_PARAMS_KEYS gates the read as well as the write: a key dropped from
    # the manifest must stop flowing even from run-params.json files that
    # recorded it back when it was one
    applied = {k: v for k, v in params.items() if k in dests and k in RUN_PARAMS_KEYS}
    parser.set_defaults(**applied)
    args = parser.parse_args()
    if applied:
        print(f"defaults from {Path(known.cache_dir) / RUN_PARAMS_FILE}: "
              + ", ".join(sorted(applied)) + " (command line overrides)")
    return args
