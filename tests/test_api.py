"""src/api.py: the four routes, their status codes, and the warmup gate.

No GPU and no SigLIP anywhere here — `ServerState.embed` is a callable, so the
whole HTTP surface runs against a stub that returns deterministic vectors. That
seam is the point of the design: wiring an `Embedder` in directly would make
every one of these a GPU test.

What is pinned is the *contract model-browser codes against*: which status code
each rejection produces, that a weak query still returns its hits, that a
scoped query is judged against its scope, and that `/status` answers before the
server is ready. The scoring itself belongs to tests/test_query.py and the
scoping to tests/test_collection.py; duplicating them here would pin the same
formula twice.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import ServerState, create_app
from src.collection import Collection
from test_collection import build          # the real-cache-on-disk fixture

DIM = 8


def stub_embed(vectors=None):
    """A text pass with no model: each text maps to a fixed unit vector.

    `vectors` maps text -> a DIM-length list, so a test can aim a query at a
    particular model by giving it that model's embedding. Anything unnamed gets
    a vector orthogonal to the fixture's, which is how a "matches nothing"
    query is expressed."""
    vectors = vectors or {}
    other = np.zeros(DIM, dtype=np.float32)
    other[-1] = 1.0

    def embed(texts, raw=False):
        cols = [np.asarray(vectors.get(t, other), dtype=np.float32) for t in texts]
        m = np.stack(cols, axis=1)                       # (dim, n_texts)
        return m / np.linalg.norm(m, axis=0, keepdims=True)
    return embed


def serve(tmp_path, layout=("a/one.stl", "a/two.stl", "b/three.stl"), *,
          vectors=None, ready=True, **over):
    args, root, files = build(tmp_path, list(layout), **over)
    c = Collection.load(args)
    state = ServerState(args, collection=c if ready else None,
                        embed=stub_embed(vectors) if ready else None,
                        model="stub-model", device="cpu")
    return TestClient(create_app(state), raise_server_exceptions=False), state, c


# --- /status ----------------------------------------------------------------

def test_status_reports_the_cache_and_the_collection_root(tmp_path):
    client, _, c = serve(tmp_path)
    s = client.get("/status").json()
    assert s["ready"] is True
    assert s["collection_root"] == str(c.root)
    assert (s["n_models"], s["n_views"], s["dim"]) == (3, 2, DIM)
    assert s["volume"] == {"present": True, "root": str(c.root), "missing": None}
    assert s["covers"] == ["stl"]


def test_status_answers_while_warming_and_queries_do_not(tmp_path):
    """Bind before warm: a probe that only checks for a response must be able
    to tell warming from not-running, or the consumer's affordance flickers off
    across every restart (surface.md §`GET /status`)."""
    client, state, _ = serve(tmp_path, ready=False)
    s = client.get("/status").json()
    assert s["ready"] is False
    assert s["elapsed"] >= 0 and s["n_models"] is None
    # three states, not two: null is "not checked yet", distinct from the
    # false a failed load reports. Saying false while warming would be a lie
    assert s["volume"]["present"] is None and s["failure"] is None

    for route, body in (("/query", {"text": "x"}), ("/similar", {"path": "a"})):
        r = client.post(route, json=body)
        assert r.status_code == 503, route
        assert r.json()["detail"]["ready"] is False


def test_status_reports_why_a_load_failed(tmp_path):
    """A server that cannot read its cache still has something true to say."""
    from src.collection import CacheUnusable

    def boom():
        raise CacheUnusable("cache_version 0, this code expects 1",
                            "run: migrate_cache_keys.py --apply")

    client, state, _ = serve(tmp_path, ready=False)
    state.warm(boom, lambda: (None, None, None))
    s = client.get("/status").json()
    assert s["ready"] is False
    assert "cache_version" in s["failure"]["reason"]
    assert "migrate_cache_keys" in s["failure"]["hint"]
    assert s["failure"]["kind"] == "CacheUnusable"


def test_status_names_the_volume_it_looked_for_when_it_is_absent(tmp_path):
    """`{"present": false, "root": null}` tells a UI nothing it could show a
    person. Found by running the real server against an unmounted drive: the
    root was in `failure` and `volume` was empty (2026-08-19)."""
    import shutil
    from src.collection import Collection as C
    args, root, _ = build(tmp_path, ["a/one.stl"])
    client, state, _ = serve(tmp_path, ready=False)
    shutil.rmtree(root)
    state.warm(lambda: C.load(args), lambda: (None, None, None))

    s = client.get("/status").json()
    assert s["volume"] == {"present": False, "root": str(root),
                           "missing": str(root)}
    assert s["collection_root"] == str(root)     # usable even unloaded
    assert s["failure"]["kind"] == "VolumeUnavailable"


# --- /query -----------------------------------------------------------------

def test_query_returns_hits_in_the_documented_shape(tmp_path):
    client, _, _ = serve(tmp_path)
    body = client.post("/query", json={"text": "anything"}).json()
    assert set(body) == {"scope", "weak", "best_z", "truncated", "results"}
    assert set(body["results"][0]) == {"id", "path", "rel_path", "name",
                                       "score", "z", "pose"}
    assert body["scope"]["status"] == "indexed"


def test_query_scopes_to_a_path(tmp_path):
    client, _, _ = serve(tmp_path)
    body = client.post("/query", json={"text": "x", "path": "a"}).json()
    assert {h["rel_path"] for h in body["results"]} == {"a/one.stl", "a/two.stl"}
    assert body["scope"]["path"] == "a" and body["scope"]["n_indexed"] == 2


def test_query_still_returns_hits_when_the_query_is_weak(tmp_path):
    """The REPL suppresses a weak query's output; the API must not. The UI can
    grey them and let a person judge, which is what the z is for
    (surface.md §`POST /query`)."""
    client, _, _ = serve(tmp_path)
    body = client.post("/query", json={"text": "nothing like this"}).json()
    assert body["weak"] is True
    assert body["results"], "a weak query still reports what it found"
    assert body["best_z"] is not None


def test_query_on_an_unindexed_directory_is_a_200_not_an_error(tmp_path):
    """'nothing matched' and 'nothing here is classified' are different
    answers and the UI has to be able to say which."""
    client, _, _ = serve(tmp_path, layout=["a/one.stl", "b/two.stl"],
                         embed=["a/one.stl"])
    r = client.post("/query", json={"text": "x", "path": "b"})
    assert r.status_code == 200
    body = r.json()
    assert body["results"] == [] and body["weak"] is False
    assert body["scope"]["status"] == "unindexed"


def test_query_min_score_truncates_at_the_cap(tmp_path):
    client, _, _ = serve(tmp_path)
    body = client.post("/query", json={"text": "x", "min_score": -1.0,
                                       "cap": 2}).json()
    assert body["truncated"] is True and len(body["results"]) == 2


def test_query_top_bounds_the_result_count(tmp_path):
    client, _, _ = serve(tmp_path)
    body = client.post("/query", json={"text": "x", "top": 1}).json()
    assert len(body["results"]) == 1 and body["truncated"] is False


def test_the_cap_is_a_ceiling_on_top_too(tmp_path):
    """It bounds what the server will serialise, not just a `min_score` floor
    — so a caller asking for more than the cap is told, rather than quietly
    given a smaller response (review, 2026-08-19: the spec said `min_score`
    only while the code capped both)."""
    client, _, _ = serve(tmp_path)
    body = client.post("/query", json={"text": "x", "top": 3, "cap": 2}).json()
    assert len(body["results"]) == 2 and body["truncated"] is True


@pytest.mark.parametrize("path,code", [
    ("a/pack.zip!/inner.stl", 422),         # zip virtual path: unaddressable
    ("/etc", 400),                          # outside the collection
    ("a/nowhere", 404),                     # not on disk
])
def test_query_maps_each_scope_rejection_to_its_own_code(tmp_path, path, code):
    client, _, _ = serve(tmp_path)
    assert client.post("/query", json={"text": "x", "path": path}).status_code == code


def test_an_unknown_pool_is_rejected_by_the_schema(tmp_path):
    """`pool` is a per-request field, so a typo must be a 422 rather than
    softmax results a caller cannot distinguish."""
    client, _, _ = serve(tmp_path)
    assert client.post("/query", json={"text": "x", "pool": "meen"}).status_code == 422


def test_an_unscoped_query_scores_the_matrix_itself_not_a_copy(tmp_path, monkeypatch):
    """The full-collection scope must hand `score` the matrix, not a
    fancy-indexed copy — ~206 MB per request on embed-cache2, 3-4x the
    scoring matmul (review, 2026-08-20). Pinned by identity, which is the
    argument the shortcut rests on: rows == arange means the slice IS the
    matrix. A scoped query still slices."""
    from src import query
    client, _, c = serve(tmp_path)
    seen, real_score = [], query.score

    def watching(matrix, text_T, pool):
        seen.append(matrix)
        return real_score(matrix, text_T, pool)

    monkeypatch.setattr(query, "score", watching)
    client.post("/query", json={"text": "x"})
    assert seen[0] is c.matrix                       # identity, not equality
    client.post("/query", json={"text": "x", "path": "a"})
    assert seen[1] is not c.matrix and seen[1].shape[0] == 2


def test_the_text_forward_is_serialised(tmp_path):
    """The threadpool means handlers really do run at once, and ollama and
    SigLIP cannot share the card. The lock covers the forward and not the
    matmul, which is CPU numpy."""
    client, state, _ = serve(tmp_path)
    inner = state.embed
    held = []

    def watching(texts, raw=False):
        held.append(state.gpu.locked())
        return inner(texts, raw)

    state.embed = watching
    client.post("/query", json={"text": "x"})
    assert held == [True]


# --- /similar ---------------------------------------------------------------

def test_similar_ranks_neighbours_and_excludes_the_model_itself(tmp_path):
    client, _, c = serve(tmp_path)
    body = client.post("/similar", json={"path": "a/one.stl"}).json()
    rels = [h["rel_path"] for h in body["results"]]
    assert "a/one.stl" not in rels                 # never itself
    assert len(rels) == 2


def test_similar_reports_z_but_never_weak(tmp_path):
    """Measured over 200 real models, the best neighbour's z never fell below
    the cutoff, so a `weak` flag here would be permanently false — worse than
    absent (implementation.md §Open decisions)."""
    client, _, _ = serve(tmp_path)
    body = client.post("/similar", json={"path": "a/one.stl"}).json()
    assert set(body) == {"scope", "results"}
    assert "weak" not in body
    assert all("z" in h for h in body["results"])


def test_similar_honours_a_scope(tmp_path):
    client, _, _ = serve(tmp_path)
    body = client.post("/similar", json={"path": "a/one.stl", "scope": "b"}).json()
    assert [h["rel_path"] for h in body["results"]] == ["b/three.stl"]


def test_similar_scopes_the_neighbours_not_the_query_model(tmp_path):
    """`scope` restricts what may be *returned*; the query model does not have
    to be inside it (surface.md §`POST /similar`). Both readings are
    defensible, so the one chosen is pinned."""
    client, _, _ = serve(tmp_path)
    body = client.post("/similar",
                       json={"path": "a/one.stl", "scope": "b"}).json()
    assert body["scope"]["path"] == "b"
    assert [h["rel_path"] for h in body["results"]] == ["b/three.stl"]
    # and the target is excluded only where it would otherwise appear
    inside = client.post("/similar",
                         json={"path": "a/one.stl", "scope": "a"}).json()
    assert [h["rel_path"] for h in inside["results"]] == ["a/two.stl"]


def test_similar_on_a_directory_is_unprocessable(tmp_path):
    """`path` names the query model; a directory names many."""
    client, _, _ = serve(tmp_path)
    assert client.post("/similar", json={"path": "a"}).status_code == 422


def test_similar_on_an_unindexed_file_is_a_404(tmp_path):
    client, _, _ = serve(tmp_path, layout=["a/one.stl", "b/two.stl"],
                         embed=["a/one.stl"])
    assert client.post("/similar", json={"path": "b/two.stl"}).status_code == 404


def test_similar_with_no_other_candidate_returns_empty(tmp_path):
    client, _, _ = serve(tmp_path, layout=["a/one.stl"])
    body = client.post("/similar", json={"path": "a/one.stl"}).json()
    assert body["results"] == []


# --- /reload ----------------------------------------------------------------

def test_reload_picks_up_a_new_model(tmp_path):
    client, state, first = serve(tmp_path, layout=["a/one.stl"])
    build(tmp_path, ["a/one.stl", "a/two.stl"])       # a classify run happened
    body = client.post("/reload", json={"rescan": True}).json()
    assert body["n_models"] == 2
    assert state.collection is not first              # rebound, not mutated
    assert len(first.files) == 1                      # the old one still valid


def test_reload_is_the_way_back_from_a_failed_startup(tmp_path):
    """The retry the startup message tells the operator to perform.

    A server that printed "mount the volume and retry" and then refused every
    attempt until it was restarted had the last step of its own story missing:
    `/reload` required `ready`, and `ready` was only ever set by the one-shot
    warmup thread (review, 2026-08-19)."""
    import shutil
    args, root, _ = build(tmp_path, ["a/one.stl"])
    saved = tmp_path / "saved"
    shutil.copytree(root, saved)
    shutil.rmtree(root)                                  # drive unplugged

    state = ServerState(args, embed=stub_embed(), model="stub", device="cpu")
    client = TestClient(create_app(state), raise_server_exceptions=False)
    state.warm(lambda: Collection.load(args), lambda: (state.embed, "stub", "cpu"))
    assert client.get("/status").json()["ready"] is False
    assert client.post("/query", json={"text": "x"}).status_code == 503

    shutil.copytree(saved, root)                         # drive plugged back in
    assert client.post("/reload", json={}).status_code == 200
    assert client.get("/status").json()["ready"] is True
    assert client.post("/query", json={"text": "x"}).status_code == 200


def test_status_stops_claiming_the_volume_after_a_reload_finds_it_gone(tmp_path):
    """Serving 200s off the intact local matrix is right — that is the whole
    VolumeUnavailable argument. Claiming the volume is still there is not: the
    field was derived from the bound collection, so it reported the last
    *successful* load rather than what the server most recently learned, and
    that is the one a consumer acts on (review, 2026-08-19)."""
    import shutil
    client, state, c = serve(tmp_path, layout=["a/one.stl"])
    shutil.rmtree(c.root)

    assert client.post("/reload", json={}).status_code == 409
    s = client.get("/status").json()
    assert s["volume"]["present"] is False               # not the stale true
    assert s["failure"]["kind"] == "VolumeUnavailable"
    assert client.post("/query", json={"text": "x"}).status_code == 200  # still serves


def test_a_corrupt_cache_reload_is_enveloped_never_a_bare_500(tmp_path):
    """Every way a reload can fail must be enveloped. A torn walk cache raised
    JSONDecodeError past `post_reload`'s catch and became a bare 500 with a
    plain-text body (review, 2026-08-19). On a *serving* process the envelope
    is the 409 reload-failure shape — 503 with `ready: true` inside was a
    self-contradiction the consumer folded into its warming state (review,
    2026-08-20)."""
    from pathlib import Path as P
    client, state, first = serve(tmp_path)
    next(P(state.args.cache_dir).glob("walk-*.json")).write_text("")

    r = client.post("/reload", json={})
    assert r.status_code == 409
    assert set(r.json()["detail"]) == {"reloaded", "ready", "failure"}
    assert r.json()["detail"]["reloaded"] is False
    assert r.json()["detail"]["ready"] is True        # and it says so honestly
    assert state.collection is first                  # still serving the old one
    assert client.post("/query", json={"text": "x"}).status_code == 200


def test_a_corrupt_run_params_or_pose_cache_is_enveloped_too(tmp_path):
    """The two escapes the 2026-08-20 review found: `cache_root`'s json.loads
    of run-params.json ran before the CacheUnusable boundary, and a
    list-shaped pose-cache.json raised AttributeError from `raw.items()` —
    both reached the client as bare 500s with `/status` reporting the stale
    success."""
    from pathlib import Path as P

    client, state, first = serve(tmp_path)
    params = P(state.args.cache_dir) / "run-params.json"
    good = params.read_text()
    params.write_text("{ torn")
    assert client.post("/reload", json={}).status_code == 409
    assert client.get("/status").json()["failure"]["kind"] == "CacheUnusable"

    params.write_text(good)
    (P(state.args.cache_dir) / "pose-cache.json").write_text("[]")
    assert client.post("/reload", json={}).status_code == 409
    assert client.post("/query", json={"text": "x"}).status_code == 200


def test_a_failed_reload_keeps_the_server_working(tmp_path):
    """A reload that cannot complete must not break a serving process: the old
    collection stays bound and the failure is reported as a 409, not as the
    503 the consumer reads as warming."""
    import shutil
    from src.cachedir import embeds_dir
    client, state, first = serve(tmp_path)
    shutil.rmtree(embeds_dir(state.args.cache_dir))

    r = client.post("/reload", json={})
    assert r.status_code == 409
    assert state.collection is first                  # unchanged
    assert client.post("/query", json={"text": "x"}).status_code == 200


def test_a_failed_reload_on_an_unready_server_keeps_the_503_envelope(tmp_path):
    """Not ready and cannot reload agree with each other: the 503 there is
    true, and it is the shape the consumer's warming policy already parses."""
    import shutil
    args, root, _ = build(tmp_path, ["a/one.stl"])
    shutil.rmtree(root)                                  # never loadable
    state = ServerState(args, embed=stub_embed(), model="stub", device="cpu")
    client = TestClient(create_app(state), raise_server_exceptions=False)

    r = client.post("/reload", json={})
    assert r.status_code == 503
    assert set(r.json()["detail"]) == {"ready", "elapsed", "failure"}
    assert r.json()["detail"]["ready"] is False


def test_reload_does_not_erase_a_siglip_failure_it_cannot_repair(tmp_path):
    """`post_reload` cleared `load_error` unconditionally while reloading only
    the Collection — a failed SigLIP load was erased, nothing ever re-ran it,
    and the one route built to explain a broken server answered
    `ready: false, failure: null` forever (review, 2026-08-20). The reload
    now retries the embed load; when that fails again, the failure stays
    reported."""
    args, root, _ = build(tmp_path, ["a/one.stl"])
    state = ServerState(args)

    def no_snapshot():
        raise RuntimeError("no HF snapshot for stub-model")

    client = TestClient(create_app(state), raise_server_exceptions=False)
    state.warm(lambda: Collection.load(args), no_snapshot)
    assert client.get("/status").json()["failure"] is not None

    r = client.post("/reload", json={})
    assert r.status_code == 200                       # the collection half worked
    assert r.json()["ready"] is False                 # and the body says so
    s = client.get("/status").json()
    assert s["ready"] is False
    assert s["failure"] is not None                   # not erased
    assert "snapshot" in s["failure"]["reason"]


def test_reload_is_the_retry_for_a_failed_siglip_load_too(tmp_path):
    """`warm` is one-shot, so `/reload` is the only way the model ever gets a
    second chance without a restart — the same story as the volume retry."""
    args, root, _ = build(tmp_path, ["a/one.stl"])
    state = ServerState(args)
    attempts = []

    def flaky_embed():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("CUDA out of memory")
        return stub_embed(), "stub-model", "cpu"

    client = TestClient(create_app(state), raise_server_exceptions=False)
    state.warm(lambda: Collection.load(args), flaky_embed)
    assert client.get("/status").json()["ready"] is False
    assert client.post("/query", json={"text": "x"}).status_code == 503

    r = client.post("/reload", json={})
    assert r.status_code == 200 and r.json()["ready"] is True
    assert client.get("/status").json()["failure"] is None
    assert client.post("/query", json={"text": "x"}).status_code == 200


def test_a_finished_warmup_does_not_revert_a_reload_that_landed_inside_it(tmp_path):
    """`/reload` is reachable while warming — deliberately — and handlers run
    concurrently with the warm thread, so a reload can finish inside `warm`'s
    `load_collection()`. The unconditional assignment then reverted the
    fresher post-rescan collection the 200 had just promised (review,
    2026-08-20). Driven synchronously: the reload fires from inside warm's
    loader, exactly the interleaving the generation counter exists for."""
    args, root, _ = build(tmp_path, ["a/one.stl"])
    stale = Collection.load(args)
    state = ServerState(args)
    client = TestClient(create_app(state), raise_server_exceptions=False)

    def load_collection_with_reload_inside():
        assert client.post("/reload", json={}).status_code == 200
        return stale                       # warm's own, older result

    state.warm(load_collection_with_reload_inside,
               lambda: (stub_embed(), "stub-model", "cpu"))
    assert state.collection is not stale   # the reload's newer instance stands
    assert state.ready


def test_a_finished_warmup_does_not_erase_a_failed_reloads_record(tmp_path):
    """The residual W4-class hole (reproduced, review follow-up 2026-08-21):
    warm's tail cleared `load_error` unconditionally, and a *failed* reload
    did not bump the generation — so a reload failing during the warmup
    window had its recorded failure erased when warm finished. With a
    VolumeUnavailable erased, `/status` went back to `present: true,
    failure: null` off warm's collection while the drive was gone — the
    exact invariant `volume_of` exists to hold."""
    from pathlib import Path as P
    args, root, _ = build(tmp_path, ["a/one.stl"])
    loadable = Collection.load(args)       # read while the cache is intact
    state = ServerState(args)
    client = TestClient(create_app(state), raise_server_exceptions=False)

    def load_collection_with_failing_reload_inside():
        (P(args.cache_dir) / "run-params.json").write_text("{ torn")
        assert client.post("/reload", json={}).status_code == 503  # not ready yet
        assert state.load_error is not None
        return loadable

    state.warm(load_collection_with_failing_reload_inside,
               lambda: (stub_embed(), "stub-model", "cpu"))
    assert state.load_error is not None    # warm did not erase the finding
    assert client.get("/status").json()["failure"]["kind"] == "CacheUnusable"
    # ...and keeping the finding does not cost the collection: the failed
    # reload bound nothing, so warm's read is the only one there is and
    # publishing it reverts nothing. Discarding it left the server answering
    # 503 to every query until someone reloaded again, where a *serving*
    # process survives the identical failure with its old collection bound
    # (the 409 arm). The failure is still reported over it.
    assert state.collection is loadable
    assert state.ready
    assert client.post("/query", json={"text": "x"}).status_code == 200
    s = client.get("/status").json()
    assert s["failure"]["kind"] == "CacheUnusable"
    assert s["loaded_at"] is not None      # ready with no load time is no state


def test_a_late_warm_failure_does_not_clobber_a_successful_reload(tmp_path):
    """The same erasure mirrored: warm's except arm wrote `load_error`
    unconditionally, so a warm thread failing *after* a successful mid-warm
    reload marked a healthy serving process broken."""
    args, root, _ = build(tmp_path, ["a/one.stl"])
    state = ServerState(args)
    client = TestClient(create_app(state), raise_server_exceptions=False)

    def load_collection_then_die():
        assert client.post("/reload", json={}).status_code == 200
        raise RuntimeError("walk cache torn under warm")

    state.warm(load_collection_then_die,
               lambda: (stub_embed(), "stub-model", "cpu"))
    assert state.load_error is None        # the reload's clean state stands
    assert client.get("/status").json()["failure"] is None


# --- bind once --------------------------------------------------------------

class CountingState(ServerState):
    """Records every read of `.collection`, so a second one is visible."""

    def __init__(self, *a, **k):
        # before super().__init__, which assigns and then reads through `ready`
        self.reads, self._collection = [], None
        super().__init__(*a, **k)

    @property
    def collection(self):
        self.reads.append(1)
        return self._collection

    @collection.setter
    def collection(self, value):
        self._collection = value


@pytest.mark.parametrize("call", [
    pytest.param(lambda cl: cl.get("/status"), id="status"),
    pytest.param(lambda cl: cl.post("/query", json={"text": "x"}), id="query"),
    pytest.param(lambda cl: cl.post("/similar", json={"path": "a/one.stl"}),
                 id="similar"),
    pytest.param(lambda cl: cl.post("/reload", json={}), id="reload"),
])
def test_every_handler_binds_the_collection_at_most_once(tmp_path, call):
    """A second read can straddle a `/reload` and index a new matrix with rows
    resolved against the old one: wrong rows, no exception, and no interleaving
    a test could reliably construct. So the rule is enforced by *counting*
    reads rather than by hoping the race never lands — an interleaving test
    would pass on any implementation and prove nothing (review, 2026-08-19).

    Before this, `/status` read three times and every other route twice, split
    between `_live()`'s check-then-return and the `volume` property."""
    args, root, files = build(tmp_path, ["a/one.stl", "a/two.stl"])
    c = Collection.load(args)
    state = CountingState(args, embed=stub_embed(), model="stub", device="cpu")
    state.reads = []
    state.collection = c
    client = TestClient(create_app(state), raise_server_exceptions=False)

    state.reads = []
    assert call(client).status_code == 200
    assert len(state.reads) <= 1, f"read {len(state.reads)} times"


# --- the contract with the consumer ----------------------------------------

def test_every_response_is_json_serialisable_without_numpy_types(tmp_path):
    """model-browser reads these over the wire; a numpy float is not JSON and
    a silently stringified one is worse."""
    import json
    client, _, _ = serve(tmp_path)
    for route, body in (("/query", {"text": "x"}),
                        ("/similar", {"path": "a/one.stl"})):
        payload = client.post(route, json=body).json()
        json.dumps(payload)                            # round-trips
        for hit in payload["results"]:
            assert type(hit["score"]) is float and type(hit["z"]) is float
            assert type(hit["id"]) is str
