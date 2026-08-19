# Implementation plan — the query API

Companion to [surface.md](surface.md), which is the spec: what the calls are
and what they return. This is the order to build it in, what each step has to
prove before the next one starts, and the decisions still open.

Five phases. Phases 0–1 involve no HTTP at all and are where the work is;
phase 2 is mechanical once phase 1 has the right shape. Each phase ends with
something runnable, so a phase that turns out wrong is cheap to abandon.

## What already exists

`src/query.py` (commit `0f524e7`) — pooling, robust z, ranking, and the
`rank` contract the API needs: total on an empty scope, `best=None` when
nothing was scored, deterministic ties. The formula half is done, and the
REPL is already a thin client of it.

## Phase 0 — prerequisites

**0.1 Install FastAPI.** The venv is uv-managed and has no pip:

```
uv pip install --python .venv/bin/python fastapi uvicorn
```

pydantic 2.13.4 is already there (via dash), which is most of FastAPI's
weight. Flask 3.1.3 is there too, transitively — it is not a chosen
dependency and does not reopen the framework decision (surface.md §Stack).
Record the resolved versions here once installed.

**0.2 Move `view_angles` into `src/pose.py`.** It lives in `src/renderer.py`,
which is child-side and imports open3d; the API must name a front view's
azimuth without loading a rendering library. The function is pure numpy and
`renderer` already imports `pose`, so the move itself is cycle-free and
mechanical.

**The callers are the work, and leaving them alone is the trap.** After the
move `renderer` still needs `view_angles`, so it imports the name — and every
existing `from src.renderer import view_angles` keeps working against a module
that no longer owns it. That is precisely the re-export shape the eval-debt
cleanup deleted (`OPEN_QUESTIONS.md`, amended 2026-08-18; `eval/README.md`
keeps a name→owner table so each name has one home). Repoint all of them:

| site | what to do |
|---|---|
| `src/renderer.py:324,344` | internal use — import from `pose` |
| `src/renderer.py:41` | module docstring lists `view_angles` among what this module is "the one home for" — no longer true, amend it |
| `eval/tile_count.py:93` | repoint the import; prose at `:14,:18,:87` names `src.renderer.view_angles` |
| `eval/gold_upright.py:37,50` | repoint the import |
| `eval/views_camera_rotation.py:63,162` | repoint the import (keeps `Renderer`, `orbit_camera`, `rotation_to_z_up` from `renderer`) |
| `tests/test_renderer.py:107,109` | resolves it as a `renderer_mod` **attribute**, not an import. Import from `src.pose` instead — the test's own claim is about `pose_tiles`' cameras, and `view_angles` is its oracle, so it should name the oracle's new home. The ring-subset property itself (a ring of `n` is a subset of a ring of 4) is `pose`'s to own now and belongs in `tests/test_pose.py` |
| `eval/README.md:53` | the camera-constants row needs `view_angles` split out to `src.pose` |

*Proves:* full suite green; `import src.pose` still adds ~196 modules with
open3d absent; and **no caller resolves `view_angles` through `renderer`** —
which needs two patterns, not one, because the test site reaches it by
attribute:

```
grep -rn "renderer import.*view_angles\|renderer_mod\.view_angles" eval/ tests/ src/
```

An import-only grep reports clean while `tests/test_renderer.py` still goes
through `renderer`, which is the kind of green that ends a phase early.

## Phase 1 — `src/collection.py`, the in-memory index

The bulk of the work, and none of it is HTTP. This is the object a request
handler asks questions of.

**What it owns.** The load preamble `test_categories.py` and
`cluster_models.py` both open with — `cache_root`, `load_file_list`,
`load_embedding_matrix`, `load_pose_cache`, `view_config` — plus the derived
things a hit needs. Holding it in one place is what stops the API becoming a
third copy of that preamble.

```python
class Collection:
    matrix: np.ndarray        # (n_indexed, n_views, dim), float32 unit rows
    files: list[Path]         # aligned with matrix rows
    scanned: list[Path]       # the last classify run's cached file list
    root: Path                # collection_root: the anchor and the display base
    poses: dict               # pose-cache entries by file_identity
    view_cfg: str             # keys front_view entries
```

**Four methods, and they are the whole surface:**

| method | returns | notes |
|---|---|---|
| `resolve(path)` | `Scope` | row indices + the status block |
| `pose_of(i)` | dict or `None` | the up vector and the front camera |
| `hit(i, score, z)` | dict | surface.md's `hit` shape |
| `reload(rescan)` | `Collection` | a *new* instance, see phase 2 |

**`resolve` is the one with real semantics.** It maps a path to row indices
and the `scope` block, and it is where surface.md's three rejections live:
a path containing `!/` is a zip virtual path (422), a path outside `root` is
unanswerable (400), a path that does not exist is 404. A real directory with
no indexed models under it is **not** an error — empty rows, `status:
"unindexed"`, and `rank` is already total so it flows through to a 200 with
zero hits. `n_scanned` comes from `scanned`, `n_indexed` from `files`; the two
lists exist separately for exactly this.

**No walk in a request** (surface.md §scope): `scanned` is the classify run's
*cached* list, `resolve` filters it in memory, and the only I/O the whole path
does is one `stat` for the 404.

Load time is the exception, and it is smaller than an earlier draft of this
plan said. `load_file_list` stats every cached entry to drop vanished files, so
startup and `/reload` pay that on spinning media — but the figure is the *STL
list*, not a tree walk: 133 for `embed-cache4`, 2890 for the largest cached
list here (`embed-cache2`). The "~10k" in an earlier draft was model-browser's
whole-tree entry count and measured a different thing.

**A one-line fix halves it**, and belongs in this phase: `cachedir.py:139-140`
calls `f.exists()` twice per entry — once to build `gone` for the log line,
once to filter — so the real cost is 2× the list. Keep the count, drop the
second pass. Measure the 2890 case in phase 3, not the 133 one.

**`pose_of` is where phase 0.2 pays off.** The pose cache stores an up vector
and a per-view-config `front_view` index; the viewer needs an angle. So:
`view_angles(n_views, elevations)[front]` converted to degrees, and `None`
when the pose cache holds no `front_view` for this view config — which is a
real state, not an error (surface.md §pose).

**Display name** is the REPL's rule — path relative to `root`, `/No Supports`
stripped, `.stl` dropped. Moving it here and having `test_categories.py` call
it is optional; doing it later is fine, doing it twice is not.

*Proves:* `tests/test_collection.py` against a synthetic cache built in
`tmp_path` (`tests/test_done.py`'s `make_args`/`make_rig` already build one —
reuse rather than invent). Cases that matter: every `resolve` rejection and
the unindexed-but-real directory; `pose_of` with a front view, without one,
and with a legacy integer entry; scoped row indices agreeing with a manual
filter. No torch, no GPU, no HTTP in this file or its test.

## Phase 2 — `src/api.py`, the HTTP layer

Mechanical if phase 1 is right. `create_app(state) -> FastAPI`, where `state`
carries the `Collection`, an embed callable, and a lock.

**The testability seam is the embed callable.** `state.embed(texts, raw) ->
(dim, n_texts)` numpy, unit rows. Production passes a closure over the real
`Embedder`; tests pass a stub returning deterministic vectors, so the whole
HTTP surface is testable with no GPU and no SigLIP. Building the app around
an `Embedder` directly would make every test a GPU test.

**Four routes**, exactly surface.md's tables as pydantic models. `pool` typed
`Literal["mean","max","softmax"]` so a bad value is a 422 from the framework
rather than a `ValueError` from `pool_sims` — the guard in `query.py` stays as
defence in depth, not as the primary check.

**Two rules from surface.md §Stack, both load-bearing:**

* Handlers are `def`, never `async def` — the GPU forward would stall the
  event loop; as plain `def` Starlette runs them in its threadpool.
* One uvicorn worker, no `--reload`.

**Concurrency, concretely.** The threadpool means handlers genuinely run at
once, so:

* A lock around the *text forward only*. The matmul is CPU numpy and does not
  touch the 4060; locking it too would serialise for nothing.
* `/reload` builds a whole new `Collection` and rebinds `state.collection`
  when it is finished. Readers need no lock — an in-flight query keeps using
  the instance it started with, and a half-loaded matrix is never visible.
  This is why `reload` returns a new instance rather than mutating.

  **The condition on that is bind once.** The rebind is atomic under the GIL,
  so a handler that reads `state.collection` a single time into a local is
  safe for its whole lifetime. A handler that reads it *twice* can straddle a
  reload and use row indices resolved against the old matrix to index the new
  one — silently wrong rows, not an exception. Every handler starts with
  `collection = state.collection` and never touches `state` again. Stated
  explicitly because "lock-free reads" is the framing that invites the second
  read.

*Proves:* `tests/test_api.py` with `TestClient` and the stub embedder — each
route's happy path, the four scope rejections with their status codes, a
weak query returning results with `weak: true` (the REPL suppresses, the API
must not), a scoped query whose z differs from the unscoped one, and
`/reload` picking up a file added mid-test.

## Phase 3 — `serve_api.py`, the entry point, and a live run

Root-level CLI entry, mirroring `classify_stls.py`: `add_cache_args` +
`apply_run_params` + `--host`/`--port`, builds the `Collection` and the real
`Embedder`, calls `uvicorn.run`. **Nothing imports it** — same rule as the
classifier, and for the same reason.

*Proves:* a live run against `embed-cache4` (133 models, 8 views): `/status`
reports the right `collection_root` and counts; a query returns the same
top-10 the REPL gives for the same text and pool; a scoped query narrows;
`/similar` on a known model returns its obvious neighbours. Measure query
latency and record it — the text forward is the only GPU work and everything
downstream is a matmul over ~1000 rows, so if a query is slow, that is a
finding.

**Also measure whether the server may run during a classify run**, because
that is the only way `/reload` is ever exercised — surface.md describes it as
"for after a `classify_stls.py` run adds models", which implies someone
running both. That puts two resident so400m models plus the render child on an
8188 MiB card. The in-process GPU lock says nothing about a second process,
and the repo's existing rule covers ollama, not a second SigLIP. One
measurement settles it and belongs in `CLAUDE.md`'s constraints beside the
ollama line: either they coexist, or the server has to be told to unload —
and if it must unload, `/reload` is the wrong shape and needs to drop the
model too.

## Phase 4 — docs

surface.md's status line moves from "proposal" to what shipped, amended in
place. `interfaces.md` gets two module-map lines and two import-rule rows:
`collection` (numpy, `pose`, `cachedir`, `embed_store`, `query`; **not**
torch, **not** fastapi) and `api` (fastapi, `collection`, `query`, `embedder`;
imported by nothing in `src/`). If phase 3's latency is interesting, it earns
a dated `docs/learnings/` entry and a `LEARNINGS.md` line.

## Open decisions

~~**1. `/similar`'s pooling — needs a call before phase 2.**~~ **Decided
2026-08-19: the query model's views are the text matrix.** A model is
`(n_views, dim)`, so "similar" compares two view stacks; this reduction reuses
both existing formulas and adds no third one:

```python
A = collection.matrix[i]                        # (n_views_A, dim)
sims = query.score(collection.matrix[rows], A.T, pool)   # (n_rows, n_views_A)
                                                #   pooled over each candidate's views
sims = query.pool_sims(sims, pool, axis=-1)     # (n_rows,) pooled over A's views
```

The second call is why `pool_sims` takes `axis` (already pinned by
`test_query.py::test_the_view_axis_is_selectable`) — the two poolings are the
same operation over different axes, not a new formula. Row `i` scores 1.0
against itself and is dropped before ranking.

The rejected alternative was mean-pooling A to a single vector first: cheaper,
blurrier, and it answers differently for models whose own views disagree —
exactly the models where "similar" is a hard question.

**2. Does `/similar` report `weak`?** Recommendation: no. `z` yes, computed
the same way. Weakness is a statement about a *query* being absent from the
collection; a model that is in the collection is never absent from it.

~~**3. Eager or lazy SigLIP load?**~~ **Decided: bind the port first, warm in
the background.** Neither "eager then serve" nor lazy: the model loads eagerly,
but the server answers `/status` from the moment it starts, with `ready: false`
until the load finishes, and `/query`/`/similar` return 503 meanwhile
(surface.md §`GET /status`). Loading before binding makes a warming server
indistinguishable from a dead one, and the consumer's affordance flickers off
across every restart. Loading lazily hides a bad model path until someone's
first query. This does both jobs: fails loudly at launch, and is legible while
it warms.

**4. Does the REPL adopt `Collection`?** Not in this plan. It is the obvious
follow-up — `test_categories.py` and `cluster_models.py` open with the same
preamble — but it is a refactor of working code and does not block the
server.

## Not in this plan

Auth, CORS, TLS (loopback, server-to-server), zip virtual paths (surface.md
§path space), rate limiting, and the render-bytes endpoint. The tie-break
question in `OPEN_QUESTIONS.md` bites this API harder than the REPL — a
duplicate-heavy query spends its `top`/`cap` budget on the same mesh — but
the measurement it asks for comes first, and it changes no interface here.
