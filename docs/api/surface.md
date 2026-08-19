# API surface sketch — search backend over the collection

Status: **proposal**, calls and parameters only — no server, no route, no
handler. One part has since been built: the query path this document said the
REPL and the server must share is `src/query.py` (2026-08-19), described under
**where the code goes**. Everything above that section is still a sketch.

The consumer is **`model-browser`** (`~/Documents/model-browser`) — and
specifically its Bun/Hono server on 127.0.0.1:3177, not the browser tab. This
is a loopback service call between two servers: the Hono side already walks the
library, reads zips, and renders its own three.js thumbnails, so it needs
neither listings nor images from here. It needs two things this side has and it
does not: **which models match a phrase**, and **how to orient one**.

It already has *name* search — the `file-search` capability's deep search,
rooted at a directory, returning root-relative names with a cap and a
truncation flag. Semantic search is a sibling of that, not a replacement, and
the shapes below deliberately echo it so results can render through the same
path.

Both sides address the same library: this cache is anchored at
`/run/media/masa/STLLibrary/DM Stash` (`collection_root` in `run-params.json`),
which is the tree model-browser browses. See **path space** for where the two
addressing schemes stop agreeing.

Scoring semantics are the REPL's, unchanged: templated-vs-raw text, view
pooling, robust-z, the z < 2.0 weak-query rule.

The server is one long-lived process because that is the whole point: the
embedding matrix and SigLIP load once and stay warm.

## Process parameters (startup, not per call)

The cache-identity block is `cachedir.add_cache_args` verbatim — the server
reads a cache and must agree with the classifier on what it is, same as every
other tool, and defaults come from the last run's `run-params.json`.

| param | default | note |
|---|---|---|
| `input` | last classify run | STL directory; the search scope's root |
| `--cache-dir` | `embed-cache` | |
| `--views` / `--elevations` / `--render-size` | 4 / 20 / 512 | cache identity |
| `--model` | `DEFAULT_MODEL` | cache identity |
| `--compile` / `--up-axis` | off / auto | cache identity |
| `--rescan` | off | re-walk instead of the cached file list |
| `--pool` | `softmax` | default only; every call may override |
| `--host` / `--port` | `127.0.0.1` / TBD | |

## Stack

**FastAPI + uvicorn**, decided for request validation and the OpenAPI document:
the caller is a TypeScript workspace that keeps its shared types by hand, and a
generated client is worth more than the smaller dependency Flask would have
been. Every parameter table below is a pydantic model — the tables are the
schema, not a description of one.

Two rules that follow, both load-bearing:

- **Handlers are `def`, never `async def`.** The work here is a blocking GPU
  forward plus a numpy matmul. In an `async def` handler that stalls the whole
  event loop; as a plain `def`, Starlette runs it in its threadpool, which is
  correct. This is the one foot-gun the framework choice brings and it points
  straight at this workload.
- **One worker**, no `--reload`. The matrix and SigLIP load once; a second
  worker would double the VRAM for nothing.

Neither is installed here yet (`fastapi`, `uvicorn`, and pydantic come with
it). The import-rule table in `docs/actor-refactor/interfaces.md` gets a row
for the server module: it may import torch, and nothing in `src/` may import
it back.

## Calls

### `GET /status`

No params. Cache identity (the block above), model, device, `n_models`,
`n_views`, `dim`, cache version, `missing` (walked but not in the cache),
loaded-at timestamp — and **`collection_root`**, which the Hono server needs
to decide whether a path the user is browsing is even addressable here before
it offers a semantic-search affordance over it.

**`ready` (bool) is required, and the server must answer before it is true.**
SigLIP takes real seconds to load. If the process binds the port only after
the model is resident, a probe cannot tell warming from not-running, and the
consumer's semantic affordance flickers off and on across every restart. So:
bind first, warm in the background, answer `/status` throughout with
`ready: false`, and reject `/query` and `/similar` with **503** until it flips.
A caller that sees a reply at all knows the server exists.

Bind-before-warm is load-bearing rather than polite: it is what makes warming
*observable* instead of inferred from a connection refusal, and model-browser's
`semantic-search` change depends on it — a 503 racing its probe folds back into
the warming state there rather than surfacing as a failure.

**`elapsed` (seconds since process start) rides alongside it, and is
deliberately not an ETA.** Asked whether it wanted a time-remaining estimate,
the consumer said no: an estimate would be shown to a person as a fact, and
this side cannot honestly produce one. What elapsed buys is a re-probe policy —
it separates *warming* from *wedged*, so a bounded backoff can keep checking a
load that started four seconds ago and stop re-probing (and read differently in
the UI) one that has plainly gone wrong. Neither side has to pretend to know
how long is left.

**`collection_root` comparison is the caller's, and prefix equality is a
trap.** model-browser has no configured root — `/api/dir` takes any absolute
path — so gating on this field is the only scoping that exists on that side.
The library is on removable media, and mount points move
(`/run/media/masa/STLLibrary` vs `…STLLibrary1`) for the same tree, so a string
prefix silently stops matching after a remount. Compare resolved real paths at
minimum; volume identity would be better.

### `POST /query`

| field | type | default | note |
|---|---|---|---|
| `text` | string | required | |
| `path` | string | whole collection | directory or file prefix to search within — see **scope** |
| `raw` | bool | `false` | verbatim text instead of the miniature templates (`:raw`) |
| `pool` | `mean\|max\|softmax` | server default | `:pool` |
| `top` | int | 10 | ignored when `min_score` is set |
| `min_score` | float | — | every model at or above, instead of top-N (`:min`) |
| `cap` | int | 500 | hard bound on returned hits when `min_score` is set |

Returns `{scope, weak, best_z, truncated, results: [hit]}`. `truncated` is the
`cap` biting — the same flag deep name search returns, so the UI's existing
"there are more" affordance works unchanged. `weak` is the z < 2.0 rule;
results are still returned — the REPL suppresses them, but the UI can show
them greyed and let a person judge, which is what the z number is for. z is
computed over the **scoped** subset, so a query inside one kit is judged
against that kit.

### `POST /similar`

Nearest neighbours in embedding space — "more like this one".

| field | type | default | note |
|---|---|---|---|
| `path` | string | required | the query model (absolute, or relative to the collection root) |
| `scope` | string | whole collection | restrict the neighbours, same rules as `/query`'s `path` |
| `k` | int | 10 | |
| `pool` | `mean\|max\|softmax` | server default | pooling across the two view stacks |

Returns `{scope, results: [hit]}`, the query model itself excluded. 404 when
`path` is not in the cache — a different answer from a scope with nothing
indexed, see below.

`pool` applies **twice**, because both sides are view stacks: the query
model's per-view embeddings are used as the text matrix, so `score` pools over
each candidate's views, and a second `pool_sims` over the query model's views
reduces what is left (implementation.md §Open decisions, decided). No `weak`
flag: weakness says a query is absent from the collection, and a model that is
in the collection cannot be.

### `POST /reload`

| field | type | default | note |
|---|---|---|---|
| `rescan` | bool | `false` | re-walk the input directory |

Reloads the embedding matrix and the pose cache — for after a
`classify_stls.py` run adds models. Does not reload SigLIP.

## Shared shapes

### `hit`

```jsonc
{
  "id": "Baal_Flaming_Sword_L_9f3ac1",   // identity.render_key: stem + 6 hex of the rel path
  "path": "/run/media/masa/STLLibrary/DM Stash/Kits/Baal/Baal_Flaming_Sword_L.stl",
  "rel_path": "Kits/Baal/Baal_Flaming_Sword_L.stl",   // relative to collection_root
  "name": "Kits/Baal/Baal_Flaming_Sword_L",           // display name, filler dirs stripped
  "score": 0.214,                        // pooled cosine
  "z": 3.1,                              // robust z over the scoped subset
  "pose": { ... }                        // below; null when unresolved
}
```

**`rel_path` is the join key, and a hit whose file has moved is normal.** The
consumer joins hits to its own tree snapshot by `rel_path` and stats the
misses, bounded by `top` — which is why `rel_path` matters more than `path`,
and why `hit` carries no mtime or size (model-browser keys thumbnails on
path+mtime and reads both itself). Two independent caches of one tree drift by
design: `id` is stem plus 6 hex of the *relative path*, so a moved file is
simply a different model to this index, and a deleted one lingers until the
next classify run. **Callers should drop or grey such hits quietly.** It is not
an error state and this side will not pre-validate it — that would mean a stat
per hit, on the same spinning volume the scope block refuses to walk.

No render field. model-browser makes its own thumbnails, so pointing at this
cache's images would be dead weight — and leaving it out is what keeps this API
from having to know where renders live. Cheap to add back (the render index is
one directory listing) if a caller ever wants the pixels that produced the
embedding.

### `pose` — what a viewer needs to open the model in its front pose

```jsonc
{
  "up": [0.0, 0.0, 1.0],        // ALWAYS one of six axis vectors — see below
  "azimuth_zero": [1.0, 0.0, 0.0],  // model-space direction azimuth 0 is measured from
  "source": "geometry",         // forced | geometry | siglip | vlm
  "confidence": 0.83,           // how sure we are WHICH of the six
  "front": {                    // null when no front view is cached for this view config
    "view": 2,
    "azimuth_deg": 180.0,
    "elevation_deg": 20.0
  }
}
```

**`up` is discrete, and that is a guarantee, not an accident.** Pose resolution
picks from `pose.UP_CANDIDATES` — `(0,0,±1)`, `(0,±1,0)`, `(±1,0,0)` — and
returns the winner unchanged; `FORCED_UPS` (`--up-axis z|y`) is drawn from the
same set. Verified against the live caches: `embed-cache2`'s 2945 entries and
`embed-cache4`'s 133 hold exactly six distinct values between them, all unit
axis vectors. So the six map 1:1 onto
model-browser's `OrbitAxis`, and **a consumer must not snap defensively** as
if this were a continuous rotation — there is nothing to snap, and a snap
would hide a bug rather than absorb one. If a non-axis vector ever appears
here, that is a defect on this side, not rounding.

`confidence` therefore means "how sure are we which of the six", which is the
number a UI can gate on. It is the ensemble's margin, not a claim about a
continuous orientation.

**The 1:1 mapping is for labelling and storage. It is not a substitute for the
up rotation below.** The tempting shortcut — set the viewer's spindle to the
model's up axis, pass `azimuth_deg` straight through, skip step 1 — is wrong
for half the axes. Measured against `rotation_to_z_up` and model-browser's
spindle frames over a 24×5 az/el grid; counts are from **`embed-cache2`, the
primary cache** (2945 models), with the 133-model test cache `embed-cache4`
beside it:

| `up` | cache2 | cache4 | shortcut agrees | azimuth offset if used anyway |
|---|---|---|---|---|
| `y` | 1118 | 9 | **no** | +90° |
| `z` | 1043 | 106 | yes | 0 |
| `-z` | 226 | 7 | **no** | +90° |
| `-y` | 207 | 7 | yes | 0 |
| `x` | 176 | 3 | **no** | −90° |
| `-x` | 175 | 1 | yes | 0 |

**The structural claim is the durable one: the shortcut is wrong for three of
the six axes, and `y` is one of them** — the library's single most common up.
The share of any given collection that lands on those three is whatever that
collection happens to hold: 52% of `embed-cache2` (1520/2945), 27% of
`embed-cache3`, 14% of `embed-cache4`. An earlier revision of this document led
with that last figure and called it "the failure that survives a demo", which
was a claim about a 133-model test cache rather than about the shortcut. Lead
with the axes; cite a percentage only with the cache named beside it. The offsets are exact (residual
< 1e-15), so the table also serves as a diagnostic — a model that reads 90° off
is a skipped step 1, not a bad pose.

**`azimuth_zero` exists so no consumer has to depend on that table.** It is
the model-space direction azimuth 0 is measured from — `rotation_to_z_up(up)ᵀ
· [1,0,0]` — and with it a viewer that cannot rotate meshes derives its own
offset from data rather than inheriting this side's rotation implementation.
That matters because the table is not a fact about geometry: it falls out of
one arbitrary choice inside `rotation_to_z_up`, whose antiparallel branch
resolves the `-Z` degeneracy with `Rx(π)` when infinitely many rotations would
have satisfied "take `up` to `+Z`". Rewrite that branch and the table moves,
seven models in the current cache rotate 90°, and nothing on either side
fails. **Prefer `azimuth_zero`; the table is the explanation, not the
contract.**

That choice is now pinned on this side as well
(`tests/test_pose.py::test_rotation_to_z_up_matches_open3d_bit_for_bit`, which
asserts the six against Open3D itself with `array_equal`),
because it was already load-bearing before any consumer existed: `views`
rotates the mesh by this matrix before shooting, so it decides the pixels — and
therefore the cached embeddings — for every non-`+Z` model, while the embedding
key records only the up *vector*. Changing it re-poses every non-`+Z` model
under unchanged cache keys — **1902 of `embed-cache2`'s 2945**, 65% of the
primary cache.

Two transforms, applied in this order, and the viewer needs both:

1. **Up rotation.** Rotate the mesh so `up` points at world +Z (the renderer's
   `rotation_to_z_up`: the axis-angle taking `up` to `[0,0,1]`). Without it the
   azimuth below means nothing — half the collection is modelled Y-up.
2. **Camera.** Orbit the Z-up model at `azimuth_deg` / `elevation_deg`, looking
   at the bounding-box centre. Elevation is above the horizon; azimuth is CCW
   about +Z from +X.

Both are bounds-relative and carry no distance (D4: camera state is
bounds-relative, never world coords; `stageModel` already pivots the model's
bounds to the origin). To be precise about the fit: model-browser's
`CameraState` is `{az, el, distR, target}`, all four required, so a pose
**constructs** a camera state rather than being one — it supplies two of the
four and the viewer chooses framing distance and target. This side describes
an orientation, not a shot. Note the world here is **Z-up**; three.js is Y-up
by convention, so the viewer either rotates into its own frame or treats
step 1's target as +Y.

The azimuth conventions already agree, checked rather than assumed:
model-browser's `camera.ts` under spindle `z` puts az=0 at +X increasing
toward +Y — CCW about +Z from +X, which is this document's convention. Under
that axis the mapping is degrees→radians and nothing else.

`front.view` is the index into this run's view list, and the angles are that
view's — `view_angles(views, elevations)[view]`, elevation-major. It is a
render-time metadata pick, not a stored transform, which is why it is `null`
whenever the pose cache has no `front_view` entry for the server's view config
(`views`/`elevations`): the viewer should fall back to azimuth 0 at the first
elevation, which is what view 0 always is.

**Code note:** `view_angles` currently lives in `src/renderer.py`, which is
child-side and imports open3d. It is pure numpy and needs to move to
`src/pose.py` so the server can name an angle without loading a rendering
library — same shape as the `up_axis_scores` deferral. One copy, not two.

### `scope` — path filtering, and telling the caller what isn't indexed

Every scoped response carries what the filter actually matched, because
"no results" and "you searched a directory nothing has been classified in"
are different answers and the UI must be able to say which:

```jsonc
{
  "path": "Kits/Baal",
  "status": "partial",     // indexed | partial | unindexed
  "n_indexed": 41,         // models under this path with embeddings — what was searched
  "n_scanned": 55,         // models the last classify run's walk saw under it
  "covers": ["stl"]        // the extensions this index can hold at all
}
```

- `path` accepts absolute or root-relative, directory or file.
- Not on disk at all → **404**, not a scope block.
- On disk with `n_indexed == 0` → **200**, `status: "unindexed"`, empty
  results. Not an error: the answer "that directory exists and nothing in it
  is classified yet" is a useful one, and it is the one the UI must be able to
  say instead of "nothing matched" — the same distinction deep name search
  draws between a completed search and a truncated one.
- Unscoped calls still get the block, with `path: null` and collection totals.

**No directory walk happens in a request.** `n_scanned` is a filter over the
*cached* file list the last classify run wrote, minus anything that had already
vanished when the server loaded it (`load_file_list` drops missing entries at
load). So it is "walked by the last classify run, and still present at
startup" — closer to the folder-now than a pure index claim, and it can shift
across a `/reload`. That is deliberate, and the reason is measured on
this hardware: the library lives on spinning exfat over USB, where
model-browser measured a full cold walk at ~32 s (10,614 entries at 2.4 ms
each, plus ~6.7 s of zip central-directory seeks that persist after the FS
metadata is warm). A walk in the request path would put that on every first
search of a session, and two processes walking one spinning volume contend for
the head — the pathology model-browser's `search-cancellation` change exists to
avoid, and one a Python walk could neither join nor be cancelled by.

The I/O each answer costs, stated precisely because it is the whole argument:
the **404 is one `stat`** of the scope path; `indexed`/`partial`/`unindexed`
costs **zero I/O**; only a live on-disk count would have needed the tree, and
it is the one thing dropped. `n_on_disk`/`n_unindexed` were in an earlier draft
of this document and are gone: they claimed something about the present that
this server has no cheap way to know, and "41 of 55" can be contradicted by the
grid printed beside it, where "41 models classified here" cannot.

**`covers` exists because the two sides disagree about what a model is.**
model-browser lists `.stl`, `.3mf` and `.obj`; `classify_stls.py` walks `.stl`
only. Without this field a folder of `.3mf` reports `n_scanned == n_indexed ==
0` and reads as "nothing here yet" when the truth is "nothing here is
searchable at all" — the exact ambiguity this block exists to remove. The user
sees `.3mf` tiles in the grid; the number beside them will not count them.

### Path space — real paths only

**Scope: real filesystem paths.** model-browser also addresses zip entries by
virtual path (`foo.zip!/parts/lid.stl`), and those are out of scope for now —
`classify_stls.py` walks real `.stl` files, with `unpack_models.py` flattening
archives ahead of it, so a zip-resident model has no embedding to search. A
path containing `!/` is rejected as a malformed path (**422**, pydantic's own
answer), not modelled as a state.

A path outside `collection_root` is **400**. The Hono side can pre-empt both
against `/status` — worth doing so the semantic-search affordance never appears
over a zip listing — but the server does not depend on it.

## Where the code goes

`test_categories.py` becomes a thin client of the same code the server calls —
not a client of the server (it must keep working with the server down), and
not a second copy. Today it holds scoring logic in `show_query`'s body.

~~Proposed: a new `src/query.py` owning the pure part — the scoped matrix
slice, the pooled matmul, robust z, the weak rule, top-N/`min_score` — taking
a matrix and a text-embedding callable.~~ — **landed 2026-08-19**, and
`show_query` is the printing alone now. Two departures from the paragraph
above, both deliberate:

* **A text-embedding *matrix*, not a callable.** `score(matrix, text_T, pool)`
  takes `(dim, n_texts)` of unit rows and never learns where they came from,
  which is what keeps torch out of the module (interfaces.md, row `query`:
  `import src.query` adds 135 modules, numpy and stdlib only). The templated
  and verbatim passes are the same matmul, so the choice between them stays
  the caller's — the REPL's `:raw` toggle needs it to.
* **The scoped slice is the caller's too.** A scope is a row subset of
  `matrix`: slice before calling and z narrows with it, which is the behaviour
  a scoped search wants and the reason no path filtering lives in the module.
  `rank` is total, so a scope matching nothing returns an empty ranking with
  `best is None` rather than raising — that is the `unindexed`/empty-scope
  answer above, served without a special case in the handler.

The REPL keeps its printing and its OSC-8 links, the server keeps its JSON, and
neither owns a formula. What the two do differently is the weak verdict:
`rank` returns `weak` and the caller decides, because the REPL suppresses the
listing entirely and this surface must not.

~~Open: `done.pool_sims` is pure numpy but lives in a torch-owning module.~~ —
**closed by the same change.** `pool_sims` moved into `src/query.py` and `done`
imports it from there, so the dependency runs from the pipeline's terminal
stage toward the leaf rather than the other way: nothing has to load torch, csv
and the transport to pool an array. `tests/test_query.py` pins the contract
this section describes.

## Deliberately not decided here

- **The GPU lock.** Starlette's threadpool means handlers really do run
  concurrently, so text embedding needs a lock around the 4060. (ollama and
  SigLIP cannot share the card; an HTTP surface makes that easier to violate
  than the REPL did.) Whether the lock wraps the forward or the whole handler
  is open. Less pressing than an earlier draft of this document assumed:
  model-browser's search UI separates typing — a client-side filter issuing no
  requests — from submit, so this gets one query per Enter, never one per
  keystroke.
- **Cancellation.** model-browser has a `search-cancellation` change in flight
  for name search. A semantic query is one text forward (~50 ms) plus a matmul,
  so an abandoned request costs little and latest-wins can stay client-side —
  but if the GPU lock ever queues, that stops being true. Note this reasoning
  holds *only* because no walk happens in a request (see `scope`): GPU work is
  bounded and predictable, unbounded I/O is not, and an earlier draft that
  counted a directory walk as part of a query had this argument backwards.
- **Query state in the URL.** model-browser's in-flight `search-options` change
  puts its two new search options in both localStorage and the URL, because
  they change *which results exist*. A semantic query is state of the same
  kind and will likely want the same treatment — a consumer-side decision, but
  the reason it arises is here.
- **Auth / binding.** Loopback, no auth, no CORS: the caller is the Hono
  server, not the page. If the client ever calls this directly, all three come
  back.
- **The Electron seam.** model-browser's D1 keeps the Hono app runnable on Node
  for an Electron main/sidecar. A Python service with a 4060 on it does not
  package that way; this is a dev-machine dependency, and the UI needs to
  degrade to name search when `/status` doesn't answer.
- **Whether renders are ever exposed.** Dropped from `hit` for now (above). If
  a caller wants them, the choice is a path (server-to-server, disk access) or
  a `GET /render/{id}?view=` returning bytes.
