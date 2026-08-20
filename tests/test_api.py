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


def test_a_failed_reload_keeps_the_server_working(tmp_path):
    """A reload that cannot complete must not break a serving process: the old
    collection stays bound and the failure is reported."""
    import shutil
    from src.cachedir import embeds_dir
    client, state, first = serve(tmp_path)
    shutil.rmtree(embeds_dir(state.args.cache_dir))

    r = client.post("/reload", json={})
    assert r.status_code == 503
    assert state.collection is first                  # unchanged
    assert client.post("/query", json={"text": "x"}).status_code == 200


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
