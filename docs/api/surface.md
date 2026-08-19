# API surface sketch — search backend over the collection

Status: **proposal**, calls and parameters only. Nothing here is implemented.

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

No render field. model-browser makes its own thumbnails, so pointing at this
cache's images would be dead weight — and leaving it out is what keeps this API
from having to know where renders live. Cheap to add back (the render index is
one directory listing) if a caller ever wants the pixels that produced the
embedding.

### `pose` — what a viewer needs to open the model in its front pose

```jsonc
{
  "up": [0.0, 0.0, 1.0],        // the model's up axis in *model* space
  "source": "geometry",         // forced | geometry | siglip | vlm
  "confidence": 0.83,
  "front": {                    // null when no front view is cached for this view config
    "view": 2,
    "azimuth_deg": 180.0,
    "elevation_deg": 20.0
  }
}
```

Two transforms, applied in this order, and the viewer needs both:

1. **Up rotation.** Rotate the mesh so `up` points at world +Z (the renderer's
   `rotation_to_z_up`: the axis-angle taking `up` to `[0,0,1]`). Without it the
   azimuth below means nothing — half the collection is modelled Y-up.
2. **Camera.** Orbit the Z-up model at `azimuth_deg` / `elevation_deg`, looking
   at the bounding-box centre. Elevation is above the horizon; azimuth is CCW
   about +Z from +X.

Both are bounds-relative and carry no distance, which is what model-browser's
camera model wants (D4: camera state is bounds-relative, never world coords;
`stageModel` already pivots the model's bounds to the origin). The framing
distance stays the viewer's business — this side is describing an orientation,
not a shot. Note the world here is **Z-up**; three.js is Y-up by convention, so
the viewer either rotates into its own frame or treats step 1's target as +Y.

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
  "n_indexed": 41,         // models under this path with cached embeddings — what was searched
  "n_on_disk": 55,         // STLs the walk sees under it
  "n_unindexed": 14        // the difference: run classify_stls.py to cover them
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

Proposed: a new `src/query.py` owning the pure part — the scoped matrix slice,
the pooled matmul, robust z, the weak rule, top-N/`min_score` — taking a
matrix and a text-embedding callable. The REPL keeps its printing and its OSC-8
links, the server keeps its JSON, and neither owns a formula.

Open: `done.pool_sims` is pure numpy but lives in a torch-owning module. Both
callers here hold SigLIP already, so it costs nothing today; worth a look when
`src/query.py` lands, since a query module that needs no torch is a nicer thing
to have than one that inherits it.

## Deliberately not decided here

- **The GPU lock.** Starlette's threadpool means handlers really do run
  concurrently, so text embedding needs a lock around the 4060 — the one thing
  the framework choice makes *more* pressing, not less. (ollama and SigLIP
  cannot share the card; an HTTP surface makes that easier to violate than the
  REPL did.) Whether the lock wraps the forward or the whole handler is open.
- **Cancellation.** model-browser has a `search-cancellation` change in flight
  for name search. A semantic query is one text forward (~50 ms) plus a matmul,
  so an abandoned request costs little and latest-wins can stay client-side —
  but if the GPU lock ever queues, that stops being true.
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
