"""The HTTP surface: FastAPI over a loaded `Collection` (docs/api/surface.md).

Thin by design. Every semantic decision lives below it — scoping and the hit
shape in `src/collection.py`, pooling and robust z in `src/query.py` — so this
module is request parsing, one lock, and the mapping from this project's
errors to status codes. If a rule appears here that is not in one of those two,
it is in the wrong file.

Two rules from surface.md §Stack, both load-bearing:

* **Handlers are `def`, never `async def`.** The work is a blocking GPU forward
  plus a numpy matmul. In an `async def` handler that stalls the whole event
  loop; as a plain `def`, Starlette runs it in its threadpool. This is the one
  foot-gun the framework choice brings and it points straight at this workload.
* **One worker.** The matrix and SigLIP load once; a second worker doubles the
  VRAM for nothing.

The threadpool means handlers really do run concurrently, so:

* A lock around the *text forward only*. The matmul is CPU numpy and does not
  touch the 4060; locking it too would serialise for nothing.
* `/reload` builds a whole new `Collection` and rebinds. Readers need no lock —
  but **every handler binds `state.collection` once**, into a local, and never
  reads it again. Reading twice can straddle a reload and index a new matrix
  with rows resolved against the old one: wrong rows, no exception.

The testability seam is `ServerState.embed`, a callable rather than an
`Embedder`. Production closes over the real model; tests pass a stub, so the
whole surface is exercised with no GPU and no SigLIP. Wiring an `Embedder` in
directly would make every API test a GPU test.
"""
from __future__ import annotations

import threading
import time
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import query
from src.collection import (CacheUnusable, Collection, NoSuchPath,
                            OutsideCollection, ScopeError, VirtualPath,
                            VolumeUnavailable)

POOL = Literal["mean", "max", "softmax"]


class ServerState:
    """Everything a handler needs, and the only mutable thing in the process.

    `collection` is rebound by `/reload` and read exactly once per request.
    `ready` gates the query routes: the port binds before SigLIP is resident so
    that a probe can tell warming from not-running (surface.md §`GET /status`),
    and until it flips the two scoring routes answer 503."""

    def __init__(self, args, *, collection=None, embed=None, pool="softmax",
                 model=None, device=None):
        self.args = args
        self.collection: Collection | None = collection
        self.embed = embed                  # (texts, raw) -> (dim, n_texts)
        self.default_pool = pool
        self.model, self.device = model, device
        self.gpu = threading.Lock()
        self.started = time.monotonic()
        self.loaded_at: float | None = None
        self.load_error: Exception | None = None   # why the load did not complete
        self.ready = collection is not None and embed is not None
        if self.ready:
            self.loaded_at = time.time()

    def warm(self, load_collection, load_embed) -> None:
        """Load in the background so `/status` can answer throughout.

        Failures are recorded rather than raised: a server that cannot read its
        cache still has something true to say about why, and saying it is the
        entire reason the port binds first."""
        try:
            self.collection = load_collection()
            self.embed, self.model, self.device = load_embed()
            self.loaded_at = time.time()
            self.ready = True
            self.load_error = None
        except Exception as e:              # noqa: BLE001 - reported, not swallowed
            self.load_error = e

    # --- what `/status` renders about the load ------------------------------

    @property
    def volume(self) -> dict:
        """Storage, whether or not the collection loaded.

        The absent case must carry the *root it looked for* — reporting
        `{"present": false, "root": null}` tells a UI nothing it could show a
        person, which is what this field existed to fix (found by running the
        real server against an unmounted drive, 2026-08-19).

        `present` has three states, not two: `true` loaded, `false` checked and
        missing, `null` not checked yet. A server still warming has not looked,
        and saying `false` there would be a lie a consumer could act on."""
        if self.collection is not None:
            return self.collection.volume
        if isinstance(self.load_error, VolumeUnavailable):
            return self.load_error.as_dict()
        return {"present": None, "root": None, "missing": None}

    @property
    def failure(self) -> dict | None:
        """One shape for every reason a load did not complete, so a consumer
        branches on `ready` and reads one field rather than three."""
        e = self.load_error
        if e is None:
            return None
        return {"reason": str(e).split("\n")[0],
                "hint": getattr(e, "hint", None),
                "kind": type(e).__name__}


# --- request bodies: surface.md's tables, as schema -------------------------

class QueryRequest(BaseModel):
    text: str
    path: str | None = None
    raw: bool = False
    pool: POOL | None = None
    top: int = Field(10, ge=1, le=1000)
    min_score: float | None = None
    cap: int = Field(500, ge=1, le=10000)


class SimilarRequest(BaseModel):
    path: str
    scope: str | None = None
    k: int = Field(10, ge=1, le=1000)
    pool: POOL | None = None


class ReloadRequest(BaseModel):
    rescan: bool = False


def _http(e: ScopeError) -> HTTPException:
    """The three scope rejections are three status codes, because they are
    three different things a UI has to be able to say (surface.md §scope)."""
    code = {VirtualPath: 422, OutsideCollection: 400, NoSuchPath: 404}[type(e)]
    return HTTPException(status_code=code, detail=str(e))


def create_app(state: ServerState) -> FastAPI:
    app = FastAPI(title="mini-classify query API",
                  description="Semantic search over the cached embeddings. "
                              "Spec: docs/api/surface.md")

    def _live() -> Collection:
        """Bind the collection once, and refuse the scoring routes until the
        model is resident. 503 rather than 500: warming is a state, not a
        fault, and the consumer folds it back into its retry policy."""
        if not state.ready or state.collection is None:
            raise HTTPException(status_code=503, detail={
                "ready": False, "elapsed": round(time.monotonic() - state.started, 1),
                "failure": state.failure})
        return state.collection

    def _pool(given: str | None) -> str:
        return given or state.default_pool

    @app.get("/status")
    def status() -> dict:
        """Answers from the moment the process starts, including while warming
        and after a failed load — that is what makes warming observable rather
        than inferred from a connection refusal."""
        c = state.collection
        out = {
            "ready": state.ready,
            "elapsed": round(time.monotonic() - state.started, 1),
            "loaded_at": state.loaded_at,
            "volume": state.volume,
            "failure": state.failure,
            "model": state.model or getattr(state.args, "model", None),
            "device": state.device,
            "cache_dir": str(getattr(state.args, "cache_dir", "")),
            "views": getattr(state.args, "views", None),
            "elevations": list(getattr(state.args, "elevations", []) or []),
            "render_size": getattr(state.args, "render_size", None),
            "compile": getattr(state.args, "compile", None),
            "up_axis": getattr(state.args, "up_axis", None),
            "pool": state.default_pool,
        }
        if c is None:
            out.update(collection_root=out["volume"].get("root"),
                       n_models=None, n_views=None, dim=None, missing=None)
            return out
        out.update(collection_root=str(c.root),
                   n_models=len(c.files),
                   n_views=int(c.matrix.shape[1]),
                   dim=int(c.matrix.shape[2]),
                   missing=c.missing,
                   covers=list(c.resolve(None).covers))
        return out

    @app.post("/query")
    def post_query(req: QueryRequest) -> dict:
        c = _live()
        try:
            scope = c.resolve(req.path)
        except ScopeError as e:
            raise _http(e) from e

        pool = _pool(req.pool)
        if scope.rows.size == 0:            # a real directory with nothing in
            return {"scope": scope.as_dict(), "weak": False, "best_z": None,
                    "truncated": False, "results": []}

        with state.gpu:                     # the only GPU work in the request
            text_T = state.embed([req.text], req.raw)
        sims = query.score(c.matrix[scope.rows], text_T, pool).ravel()
        ranked = query.rank(sims, top=req.top, min_score=req.min_score)

        order = ranked.order
        truncated = bool(len(order) > req.cap)
        if truncated:
            order = order[:req.cap]
        return {
            "scope": scope.as_dict(),
            "weak": ranked.weak,
            "best_z": float(ranked.z[ranked.best]),
            "truncated": truncated,
            "results": [c.hit(int(scope.rows[j]), sims[j], ranked.z[j])
                        for j in order],
        }

    @app.post("/similar")
    def post_similar(req: SimilarRequest) -> dict:
        c = _live()
        try:
            target = c.resolve(req.path)
            scope = c.resolve(req.scope)
        except ScopeError as e:
            raise _http(e) from e
        if target.rows.size == 0:
            raise HTTPException(status_code=404,
                                detail=f"{req.path} is not in the cache")
        if target.rows.size > 1:
            raise HTTPException(
                status_code=422,
                detail=f"{req.path} names {target.rows.size} models; "
                       f"/similar takes one")

        i = int(target.rows[0])
        rows = scope.rows[scope.rows != i]          # never rank a model against
        if rows.size == 0:                          # itself: it scores 1.0 and
            return {"scope": scope.as_dict(), "results": []}   # skews the z

        pool = _pool(req.pool)
        # The query model's views *are* the text matrix, so `score` pools over
        # each candidate's views and a second pooling reduces the query's own
        # (implementation.md §Open decisions). No `weak`: measured over 200
        # models, the best neighbour's z never fell below the cutoff.
        sims = query.score(c.matrix[rows], c.matrix[i].T, pool)
        sims = query.pool_sims(sims, pool, axis=-1)
        ranked = query.rank(sims, top=req.k)
        return {"scope": scope.as_dict(),
                "results": [c.hit(int(rows[j]), sims[j], ranked.z[j])
                            for j in ranked.order]}

    @app.post("/reload")
    def post_reload(req: ReloadRequest) -> dict:
        """Rebind a freshly loaded Collection, or keep the one we have.

        A failed reload must not break a working server, so the old collection
        stays bound and the failure is reported — the same shape `/status`
        carries."""
        c = _live()
        try:
            fresh = c.reload(rescan=req.rescan)
        except (CacheUnusable, VolumeUnavailable) as e:
            # Recorded so `/status` explains the stale data, but the old
            # collection stays bound: a failed reload must not break a server
            # that is answering fine.
            state.load_error = e
            raise HTTPException(status_code=503, detail=state.failure) from e
        state.collection = fresh
        state.load_error = None
        state.loaded_at = time.time()
        return {"n_models": len(fresh.files), "missing": fresh.missing,
                "volume": fresh.volume, "loaded_at": state.loaded_at}

    return app
