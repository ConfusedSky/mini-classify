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

import logging
import threading
import time
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import query
from src.cachedir import cache_version
from src.collection import (CacheUnusable, Collection, NoSuchPath,
                            OutsideCollection, ScopeError, VirtualPath,
                            VolumeUnavailable)

POOL = Literal["mean", "max", "softmax"]

# `logging`, not the `print` the rest of this project uses: a server's output
# is uvicorn's to configure, and a print bypasses whatever level, format or
# sink the operator chose. One line per scoring request — enough to answer
# "why was that slow" and "what did it actually search" without turning the
# query text into a permanent record of what someone looked for.
log = logging.getLogger("mini_classify.api")


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
        # read once, not per request: it changes only when a migration runs,
        # and `/status` can be polled by a consumer's warmup backoff
        self.cache_version = cache_version(getattr(args, "cache_dir", ""))
        self.loaded_at: float | None = None
        self.load_error: Exception | None = None   # why the load did not complete
        # The writers' lock: `warm` runs on its own thread while `/reload`
        # handlers run in Starlette's threadpool, and the two both assign
        # collection/loaded_at/load_error. Readers stay lock-free — the
        # bind-once rule is what protects them.
        self.bind = threading.Lock()
        # At most one SigLIP load at a time: `warm` holds this across
        # `load_embed`, and `/reload`'s retry takes it non-blocking — two
        # concurrent loads would double the VRAM for nothing.
        self.embed_loading = threading.Lock()
        self._load_embed = None      # stashed by warm, so /reload can retry it
        self._generation = 0         # bumped per state-writing /reload,
                                     # failure included — see warm
        if self.ready:
            self.loaded_at = time.time()

    def is_ready(self, c) -> bool:
        """Readiness of an *already-bound* collection.

        Takes the collection rather than reading it, so a handler that has
        bound it once does not read it again through a property — the
        bind-once rule applies to the accessors too, and `/status` was reading
        three times before this: directly, through `ready`, and through
        `volume` (review, 2026-08-19)."""
        return c is not None and self.embed is not None

    @property
    def ready(self) -> bool:
        """Derived, never assigned. A flag would drift from the two things it
        summarises — and did: `/reload` could not run while it was false, so a
        server whose first load failed had no way back except a restart, even
        once the drive was mounted again (review, 2026-08-19)."""
        return self.is_ready(self.collection)

    def warm(self, load_collection, load_embed) -> None:
        """Load in the background so `/status` can answer throughout.

        Failures are recorded rather than raised: a server that cannot read its
        cache still has something true to say about why, and saying it is the
        entire reason the port binds first.

        `/reload` is deliberately reachable while this runs, so a reload can
        finish *inside* our `load_collection()` — a real window, ~2801 np.load
        calls on embed-cache2 — and an unconditional assignment here then
        silently reverted the fresher post-rescan collection the 200 had just
        promised (review, 2026-08-20). The generation check is the fix: any
        reload that wrote state bumps it (`supersede_warm`, failure included),
        and a warm finding from before the bump is stale by definition. Each
        write here is guarded against its own erasure:

        * the tail's `load_error = None`, against erasing a *failed* mid-warm
          reload's record — with a `VolumeUnavailable` erased, `/status` went
          back to `present: true, failure: null` off warm's collection while
          the drive was gone, the exact 2026-08-19 invariant `volume_of`
          exists to hold (reproduced, review follow-up 2026-08-21);
        * the except arm, the same erasure mirrored — a warm failure landing
          after a successful reload must not mark a healthy server broken;
        * the collection bind, against reverting a successful reload's — and
          only that, which is why it also binds when *nothing* is bound,
          whatever the generation says. A failed reload binds no collection,
          so an empty slot means this read is the only collection in
          existence and publishing it reverts nothing. Discarding it cost the
          server every query until the next reload, to protect a finding
          `load_error` carries anyway: a *serving* process keeps its old
          collection bound through exactly this failure (`post_reload`'s
          409), and `volume_of` reports the drive gone over a bound
          collection rather than by unbinding one (review follow-up,
          2026-08-21).

        `load_embed` is stashed first, so `/reload` can retry a failed SigLIP
        load — this thread is one-shot and was the only thing that ever ran
        it. The embed bind itself is *not* generation-guarded: the model is
        process state, not a cache finding, and a resident model is right
        under any generation."""
        self._load_embed = load_embed
        gen = self._generation
        bound = False
        try:
            fresh = load_collection()
            with self.bind:
                if self._generation == gen or self.collection is None:
                    self.collection = fresh
                    bound = True
            with self.embed_loading:
                # a mid-warm /reload may have run the retry path already;
                # a second SigLIP load would only double the wait
                if self.embed is None:
                    embed, model, device = load_embed()
                    with self.bind:
                        self.embed, self.model, self.device = embed, model, device
            with self.bind:
                # `loaded_at` follows the bind, not the generation: it
                # describes the collection that is actually bound, and a
                # successful mid-warm reload stamped its own
                if bound:
                    self.loaded_at = time.time()
                if self._generation == gen:
                    self.load_error = None
        except Exception as e:              # noqa: BLE001 - reported, not swallowed
            with self.bind:
                if self._generation == gen:
                    self.load_error = e

    def supersede_warm(self) -> None:
        """Called (under `bind`) by every `/reload` that writes state, the
        failure branch included: what the reload just learned is newer than
        anything `warm` still has in flight, so warm's tail must not publish
        over it."""
        self._generation += 1

    def retry_embed(self) -> Exception | None:
        """`/reload`'s half of the warmup: load SigLIP iff it never loaded.

        Returns the failure to report, or None (which also means "nothing to
        do"). Non-blocking on `embed_loading`: if warm or another reload is
        loading right now, the model is already on its way and a second load
        would double the VRAM."""
        if self.embed is not None or self._load_embed is None:
            return None
        if not self.embed_loading.acquire(blocking=False):
            return None
        try:
            if self.embed is None:
                try:
                    embed, model, device = self._load_embed()
                except Exception as e:      # noqa: BLE001 - reported by /reload
                    return e
                with self.bind:
                    self.embed = embed
                    self.model, self.device = model, device
        finally:
            self.embed_loading.release()
        return None

    # --- what `/status` renders about the load ------------------------------

    def volume_of(self, c) -> dict:
        """Storage, whether or not the collection loaded. Takes the bound
        collection rather than reading it — see `is_ready`. (The property
        form of this died unused once `/status` bound its collection first;
        review, 2026-08-20.)

        The absent case must carry the *root it looked for* — reporting
        `{"present": false, "root": null}` tells a UI nothing it could show a
        person, which is what this field existed to fix (found by running the
        real server against an unmounted drive, 2026-08-19).

        `present` has three states, not two: `true` loaded, `false` checked and
        missing, `null` not checked yet. A server still warming has not looked,
        and saying `false` there would be a lie a consumer could act on.

        The most recent *finding* wins over the bound collection, which is the
        same lie inverted: a reload that discovered the drive gone left this
        reporting `present: true` off the last successful load, and that is the
        one a consumer acts on when deciding whether to offer the affordance
        (review, 2026-08-19). Serving 200s from the intact local matrix stays
        right; claiming the volume is there does not."""
        if isinstance(self.load_error, VolumeUnavailable):
            return self.load_error.as_dict()
        if c is not None:
            return c.volume
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

    def _unready(c) -> dict:
        """One 503 body for every reason a request cannot be served, so a
        consumer parses one shape. Two routes used to return different
        schemas under the same status code (review, 2026-08-19). Takes the
        bound collection so it adds no second read."""
        return {"ready": state.is_ready(c),
                "elapsed": round(time.monotonic() - state.started, 1),
                "failure": state.failure}

    def _live() -> Collection:
        """Bind the collection **once** and refuse the scoring routes until the
        model is resident. 503 rather than 500: warming is a state, not a
        fault, and the consumer folds it back into its retry policy.

        The single read is the point. Reading `state.collection` twice — even
        as a `is None` check and then a return — can straddle a `/reload` and
        hand back a different instance than the one that was checked; a handler
        that then mixed indices across both would use rows resolved against one
        matrix to index another. `tests/test_api.py` counts the reads rather
        than trying to construct the interleaving, which no test could observe
        reliably."""
        c = state.collection
        if c is None or state.embed is None:
            raise HTTPException(status_code=503, detail=_unready(c))
        return c

    def _pool(given: str | None) -> str:
        return given or state.default_pool

    @app.get("/status")
    def status() -> dict:
        """Answers from the moment the process starts, including while warming
        and after a failed load — that is what makes warming observable rather
        than inferred from a connection refusal."""
        c = state.collection                # bound once; everything below
        out = {                             # is derived from this local
            "ready": state.is_ready(c),
            "elapsed": round(time.monotonic() - state.started, 1),
            "loaded_at": state.loaded_at,
            "volume": state.volume_of(c),
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
            # Promised by surface.md §`GET /status` and missing until
            # 2026-08-19. Largely redundant — a stale key scheme refuses to
            # load at all (`CacheUnusable`), so a ready server's cache is
            # current by construction — but a consumer diagnosing a server
            # that will not start should not have to infer it.
            "cache_version": state.cache_version,
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
        t0 = time.monotonic()
        c = _live()
        try:
            scope = c.resolve(req.path)
        except ScopeError as e:
            raise _http(e) from e

        pool = _pool(req.pool)
        if scope.rows.size == 0:            # a real directory with nothing in
            log.info("query %r scope=%s %s — nothing indexed here",
                     req.text[:60], scope.path or "-", scope.status)
            return {"scope": scope.as_dict(), "weak": False, "best_z": None,
                    "truncated": False, "results": []}

        with state.gpu:                     # the only GPU work in the request
            text_T = state.embed([req.text], req.raw)
        # An unscoped query must not copy the matrix: `rows` is then the full
        # arange, and fancy-indexing with it materialises a fresh ~206 MB
        # array per request on embed-cache2 — measured at 3-4x the scoring
        # matmul itself (review, 2026-08-20). Size equality is identity: rows
        # are unique indices into files.
        scoped = c.matrix if scope.rows.size == len(c.files) \
            else c.matrix[scope.rows]
        sims = query.score(scoped, text_T, pool).ravel()
        ranked = query.rank(sims, top=req.top, min_score=req.min_score)

        order = ranked.order
        truncated = bool(len(order) > req.cap)
        if truncated:
            order = order[:req.cap]
        log.info("query %r scope=%s %s %d/%d hits%s in %.0f ms",
                 req.text[:60], scope.path or "-", scope.status,
                 len(order), scope.n_indexed, " weak" if ranked.weak else "",
                 (time.monotonic() - t0) * 1000)
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
        t0 = time.monotonic()
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
        log.info("similar %s scope=%s %d neighbours in %.0f ms",
                 req.path, scope.path or "-", len(ranked.order),
                 (time.monotonic() - t0) * 1000)
        return {"scope": scope.as_dict(),
                "results": [c.hit(int(rows[j]), sims[j], ranked.z[j])
                            for j in ranked.order]}

    @app.post("/reload")
    def post_reload(req: ReloadRequest) -> dict:
        """Rebind a freshly loaded Collection, or keep the one we have.

        A failed reload must not break a working server, so the old collection
        stays bound and the failure is reported. On a server that is *serving*,
        that failure is a **409** naming the reload, not a 503: `_unready` off
        the still-healthy collection produced `503 {"ready": true}` — a
        self-contradiction surface.md's consumer folded into its warming
        state, flickering the search affordance off while queries returned
        200s (review, 2026-08-20). A server that is not ready keeps the 503
        envelope: there the reload's failure and the server's state agree.

        **This route does not require `ready`**, and that is the point: it is
        the retry a failed startup tells the operator to perform. A server that
        printed "mount the volume and retry" and then refused every attempt
        until it was restarted had the last step of that story missing
        (review, 2026-08-19). It loads from `state.args` rather than from a
        bound collection, because when the first load failed there is none.
        The same story is why this also retries a failed SigLIP load: `warm`
        is one-shot, so without the retry a server whose model never landed
        had `/reload` return 200 while every query kept 503ing — worse, the
        route used to clear `load_error` it had not repaired, leaving
        `/status` at `ready: false, failure: null` forever."""
        c = state.collection                # bound once, like every handler
        try:
            fresh = Collection.load_with(state.args, rescan=req.rescan)
        except (CacheUnusable, VolumeUnavailable) as e:
            # Recorded so `/status` explains the stale data, but the old
            # collection stays bound: a failed reload must not break a server
            # that is answering fine. The supersede is the failure branch's
            # too — without it a warm finishing after this erased the
            # recorded failure (reproduced, review follow-up 2026-08-21).
            with state.bind:
                state.load_error = e
                state.supersede_warm()
            if state.is_ready(c):
                raise HTTPException(status_code=409, detail={
                    "reloaded": False, "ready": True,
                    "failure": state.failure}) from e
            raise HTTPException(status_code=503, detail=_unready(c)) from e

        embed_error = state.retry_embed()   # a no-op once the model is resident

        with state.bind:
            state.collection = fresh
            state.supersede_warm()          # a warm finding older than this
            state.loaded_at = time.time()   # bind must not overwrite it
            if state.is_ready(fresh):
                # clear only what is repaired: with the embed still missing,
                # erasing the recorded failure would leave `/status` at
                # `ready: false, failure: null`
                state.load_error = None
            elif embed_error is not None:
                state.load_error = embed_error
        state.cache_version = cache_version(getattr(state.args, "cache_dir", ""))
        log.info("reload rescan=%s -> %d models, %d missing",
                 req.rescan, len(fresh.files), fresh.missing)
        return {"n_models": len(fresh.files), "missing": fresh.missing,
                "volume": fresh.volume, "loaded_at": state.loaded_at,
                "ready": state.is_ready(fresh)}

    return app
