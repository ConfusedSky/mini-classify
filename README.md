# mini-classify

Zero-shot search over a collection of printable-miniature STLs. No labels, no
training: each mesh is rendered headless, its up-axis resolved, the views
embedded with SigLIP, and then any text query — or any other model — is scored
against those embeddings.

Ask it for "wizard with a staff" and it finds one, in a library where the
filenames are things like `32_Unsupported_Renard_BodyNoMask.stl`.

## What it produces

The product is the **caches**, not the CSV. A classify run writes, per model:

* a resolved **pose** (which way is up, and which view is the front),
* per-view **embeddings** as `.npy`,
* optionally the **renders** themselves.

Everything else reads those. `results.csv` is a thin top-3 dump; the real
querying happens in the REPL or over the HTTP API, both of which are
milliseconds per query once the caches exist because no rendering happens
again.

## Requirements

* Python 3.12, a `.venv` managed with [uv](https://docs.astral.sh/uv/)
  (`uv pip install --python .venv/bin/python <pkg>`).
* A GPU for SigLIP (CUDA if present, CPU otherwise) and a working headless
  Open3D/Filament for rendering. On the development machine those are two
  different devices — rendering on an AMD iGPU, SigLIP on an RTX 4060 — which
  is a measured win, not a requirement.
* Somewhere with STL files. The library is not in the repo.

## Use

```bash
# 1. build the caches (the slow part: renders, poses, embeddings)
.venv/bin/python classify_stls.py /path/to/stls --cache-dir embed-cache

# 2. query interactively
.venv/bin/python test_categories.py --cache-dir embed-cache

# 3. or serve the same queries over HTTP
.venv/bin/python serve_api.py --cache-dir embed-cache --port 8077
```

After the first run, the cache-identity flags (`--views`, `--elevations`,
`--model`, …) default to what that run recorded, so later commands usually
need only `--cache-dir`.

### The REPL

`<enter>` reloads `categories.txt` and classifies everything; any other text is
a one-off query; `:find`, `:pool`, `:min` and `:raw` adjust the search. Results
are clickable links to the model's render.

### The API

**Start it before anything that consumes it** — nothing launches it for you,
and a consumer polling `http://127.0.0.1:8077/status` sees a connection
refusal until you do:

```bash
.venv/bin/python serve_api.py --cache-dir embed-cache --port 8077
```

It answers `/status` immediately with `ready: false` and serves queries once
SigLIP is resident (~16 s on the development machine); `/query` and `/similar`
return 503 in between, so a consumer can tell warming from not-running.

Four routes over the same code the REPL uses — `GET /status`, `POST /query`,
`POST /similar`, `POST /reload`. It is loopback-only and unauthenticated by
design: the intended caller is another local service, not a browser.

`/query` takes a `path` to search within, and reports what it could not cover
— a directory that exists but has never been classified answers `200` with
`status: "unindexed"` rather than looking like "nothing matched". Full spec in
[`docs/api/surface.md`](docs/api/surface.md); it generates its own OpenAPI
document at `/openapi.json`.

## Other tools

| | |
|---|---|
| `cluster_models.py` | k-means over the cached embeddings — discover groupings without categories |
| `migrate_cache_keys.py` | re-key a cache written under an older scheme, instead of re-rendering it |
| `unpack_models.py` | some sets ship *per-model* zips inside the download, so the walk sees no STLs and the whole set is silently absent; this extracts them |
| `eval/` | measurement harnesses; every number in `LEARNINGS.md` came from one |

## Reading the repo

The docs are load-bearing here — most of the non-obvious decisions were
measured rather than chosen, and the measurement is written down.

* [`CLAUDE.md`](CLAUDE.md) — the orientation document, and the constraints that
  cost real time to discover.
* [`LEARNINGS.md`](LEARNINGS.md) — index of dated write-ups; every measured
  number resolves through it.
* [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md) — open work, amended in place with
  answers rather than deleted.
* [`docs/actor-refactor/interfaces.md`](docs/actor-refactor/interfaces.md) —
  the live spec of the pipeline: what each module may import, and why.
* [`docs/api/`](docs/api/) — the query API's spec and its phased plan.
* [`docs/cache-rebuild.md`](docs/cache-rebuild.md) — debt that only a full
  cache rebuild can pay off. Read before rebuilding.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

The renderer is faked at the op-log level and the API is exercised against a
stub embedder, so the bulk needs no GPU and no model download. The few tests
that do need something — CUDA, a local ollama, a `7z` binary — skip when it is
absent rather than failing.
