"""src/api.py: the six routes, their status codes, and the warmup gate.

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
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import RESPONSE_CAP, ServerState, create_app
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


def client_of(tmp_path, **kw):
    """`serve` when the test wants only the client."""
    return serve(tmp_path, **kw)[0]


# --- /status ----------------------------------------------------------------

def test_status_reports_the_cache_and_the_collection_root(tmp_path):
    client, _, c = serve(tmp_path)
    s = client.get("/status").json()
    assert s["ready"] is True
    assert s["collection_root"] == str(c.root)
    assert (s["n_models"], s["n_views"], s["dim"]) == (3, 2, DIM)
    assert s["volume"] == {"present": True, "root": str(c.root),
                           "missing": None, "required": True}
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

    for route, body in (("/query", {"text": "x"}), ("/similar", {"path": "a"}),
                        ("/poses", {"paths": ["a/one.stl"]}),
                        ("/under", {"path": "a", "limit": 10})):
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
                           "missing": str(root), "required": True}
    assert s["collection_root"] == str(root)     # usable even unloaded
    assert s["failure"]["kind"] == "VolumeUnavailable"


# --- /query -----------------------------------------------------------------

def test_query_returns_hits_in_the_documented_shape(tmp_path):
    client, _, _ = serve(tmp_path)
    body = client.post("/query", json={"text": "anything"}).json()
    assert set(body) == {"scope", "weak", "best_z", "truncated", "matched",
                         "results"}
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


# a dozen: enough that a ten-row default would be visible as one
MANY = [f"a/m{i:02d}.stl" for i in range(12)]


def test_a_floor_with_no_count_is_not_cut_to_a_default(tmp_path):
    """The regression the composing branch would otherwise have shipped.

    model-browser sends a floor and omits `top` — that is its default view —
    so while `top` defaulted to 10 and the bounds composed, every floor-only
    request came back as ten rows of a set that could be hundreds. Absent means
    not in force; this is the test that says so over HTTP."""
    client, _, _ = serve(tmp_path, layout=MANY)
    body = client.post("/query", json={"text": "x", "min_score": -1.0}).json()
    assert len(body["results"]) == 12
    assert body["matched"] == 12 and body["truncated"] is False


def test_a_bare_query_is_bounded_by_the_cap_alone(tmp_path):
    """`top` has no default at all now, which is a contract change: a request
    naming no bound used to be cut to ten and is now cut by `cap` (surface.md
    §`POST /query`). Pinned because "default 10" is what every reader of that
    table believed until 2026-08-27."""
    client, _, _ = serve(tmp_path, layout=MANY)
    body = client.post("/query", json={"text": "x"}).json()
    assert len(body["results"]) == 12


def test_the_bounds_compose_over_http_and_matched_says_what_was_cut(tmp_path):
    client, _, _ = serve(tmp_path, layout=MANY)
    body = client.post("/query", json={"text": "x", "min_score": -1.0,
                                       "top": 4}).json()
    assert len(body["results"]) == 4
    # the count's cut is reported by `matched`, never by `truncated` — that bit
    # is the cap's alone, and a count at or under the cap can never fire it
    assert body["matched"] == 12 and body["truncated"] is False


def test_matched_counts_the_floor_set_not_the_collection(tmp_path):
    client, _, _ = serve(tmp_path, layout=MANY)
    body = client.post("/query", json={"text": "x", "min_score": 2.0}).json()
    assert body["results"] == [] and body["matched"] == 0


def test_an_unindexed_scope_still_answers_with_a_matched_count(tmp_path):
    """Zero, not absent: a client reading the field must not have to special-case
    the one response shape that skips the ranking entirely."""
    client, _, _ = serve(tmp_path, layout=["a/one.stl", "b/two.stl"],
                         embed=["a/one.stl"])
    body = client.post("/query", json={"text": "x", "path": "b"}).json()
    assert body["matched"] == 0


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
    client, _, _ = serve(tmp_path)
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


# --- /poses -----------------------------------------------------------------

def test_poses_returns_exactly_the_block_a_hit_carries(tmp_path):
    """The whole point of the call: a listing gets the pose a search would
    have carried for the same model, byte for byte. Compared against a real
    hit rather than against a hand-written block, because "the shared shape"
    is a claim about these two agreeing and a literal would only agree with
    itself."""
    client, _, c = serve(tmp_path, ups={"a/one.stl": [0.0, 1.0, 0.0]},
                         front={"a/one.stl": 1})
    hit = next(h for h in client.post("/query", json={"text": "x"}).json()["results"]
               if h["rel_path"] == "a/one.stl")
    body = client.post("/poses", json={"paths": [hit["path"]]}).json()
    assert body["poses"][hit["path"]] == hit["pose"]
    assert hit["pose"]["up"] == [0.0, 1.0, 0.0]      # not the default: a real pose


def test_poses_takes_absolute_and_root_relative_alike(tmp_path):
    """`/similar`'s rule for its one path, applied per member of the batch."""
    client, _, c = serve(tmp_path)
    abs_path = next(str(f) for f in c.files if f.name == "one.stl")
    body = client.post("/poses", json={"paths": [abs_path, "a/one.stl"]}).json()
    assert body["poses"][abs_path] == body["poses"]["a/one.stl"] is not None


@pytest.mark.parametrize("path", [
    "a/nowhere.stl",                        # never walked
    "a",                                    # a directory, not a model
    "/etc/passwd",                          # outside the collection
    "a/pack.zip!/inner.stl",                # a zip virtual path
    "",                                     # nothing at all
])
def test_an_unaddressable_path_is_null_not_an_error(tmp_path, path):
    """The difference from `/query`'s `path`, where these are 404/400/422:
    here each is one member of a batch, and the caller's library legitimately
    holds files this index has never seen. One of them must not cost the
    others their answer."""
    r = client_of(tmp_path).post("/poses", json={"paths": [path]})
    assert r.status_code == 200
    assert r.json()["poses"] == {path: None}


def test_a_mixed_batch_keys_every_path_it_was_given(tmp_path):
    """Every requested path present as a key, echoed as it was sent — the
    caller joins on the string it already holds, not on one this side
    normalised."""
    asked = ["a/one.stl", "a/nowhere.stl", "b/three.stl", "/etc"]
    body = client_of(tmp_path).post("/poses", json={"paths": asked}).json()
    assert list(body["poses"]) == asked
    assert [body["poses"][p] is None for p in asked] == [False, True, False, True]


def test_a_repeated_path_collapses_to_one_key(tmp_path):
    """The limit of "every requested path is a key": a JSON object cannot
    carry the same key twice, so a batch that repeats a path answers it once.
    Surface.md says *distinct* for this reason — a caller counting keys
    against `len(paths)` to detect a dropped model would misread its own
    duplicate as a loss."""
    asked = ["a/one.stl", "a/nowhere.stl", "a/one.stl"]
    body = client_of(tmp_path).post("/poses", json={"paths": asked}).json()
    assert list(body["poses"]) == ["a/one.stl", "a/nowhere.stl"]
    assert body["poses"]["a/one.stl"] is not None
    assert body["poses"]["a/nowhere.stl"] is None


def test_the_batch_is_bounded_and_the_refusal_is_the_schemas_own(tmp_path):
    """1024, stated in surface.md. The bound is a pydantic constraint so the
    refusal is a 422 in the same shape a malformed path already gets — this
    route invents no error of its own — and so the response dict is bounded
    by the request rather than by the collection."""
    from src.api import POSES_MAX
    assert POSES_MAX == 1024, "surface.md states the number; change both"
    client = client_of(tmp_path)
    at = client.post("/poses", json={"paths": ["a/one.stl"] * POSES_MAX})
    over = client.post("/poses", json={"paths": ["a/one.stl"] * (POSES_MAX + 1)})
    assert (at.status_code, over.status_code) == (200, 422)


def test_poses_never_touches_the_gpu_lock(tmp_path):
    """A store lookup: no text forward, so nothing to serialise. If this ever
    took the lock a listing's batch could queue behind a query, which is the
    cost the whole no-GPU framing in surface.md promises it cannot pay."""
    client, state, _ = serve(tmp_path)
    state.embed = lambda *a, **k: pytest.fail("/poses embedded something")
    held = []

    class Watching:
        def __enter__(self): held.append(1)
        def __exit__(self, *a): pass

    state.gpu = Watching()
    assert client.post("/poses", json={"paths": ["a/one.stl"]}).status_code == 200
    assert held == []


# --- /under -----------------------------------------------------------------
#
# The folder contact sheet. model-browser's own walk cannot build it: a peek's
# 64-entry budget is spent inside a first-sorted "(Presupported)" subtree while
# the posed kits sit one subdirectory later, so a fully indexed folder comes
# back empty there. This index already holds that walk, and the poses with it.

UNDER = ["Kits/A-B/x.stl", "Kits/A/y.stl", "other/z.stl"]


def serve_as_recorded(tmp_path, layout=UNDER, **kw):
    """`serve` over a cache whose recorded walk order is not the rel order.

    Row order is not something this API gets to inherit. `load_file_list`
    replays the walk cache verbatim, so it is whatever that file holds; and
    `find_stls`'s own order is `sorted()` over `Path`s, which compared whole
    strings before Python 3.12 and components since — the two disagree on
    exactly the `Kits/A-B` vs `Kits/A` pair above, where `-` sorts under `/`.
    So the fixture records a reversed list through the production writer and
    loads it back: a layout whose stored order already happens to be its rel
    order would pin nothing."""
    import json
    from pathlib import Path as P
    args, *_ = build(tmp_path, layout, **kw)
    Collection.load(args)                          # writes the walk cache
    walk = next(P(args.cache_dir).glob("walk-*.json"))
    saved = json.loads(walk.read_text())
    saved["files"].reverse()
    walk.write_text(json.dumps(saved))
    return serve(tmp_path, layout=layout, rescan=False, **kw)


def test_under_lists_a_directory_in_rel_path_order(tmp_path):
    """Sorted by the root-relative path tuple, which is not the order the rows
    are stored in — and with mixed poses, since an indexed model whose pose is
    unresolved is a null the caller has to be able to receive."""
    client, _, c = serve_as_recorded(tmp_path)
    assert [f.name for f in c.files] == ["z.stl", "x.stl", "y.stl"]  # the premise
    key = next(k for k in c.poses if k.startswith("Kits/A-B/x.stl|"))
    c.poses[key] = None                            # indexed, pose unresolved

    body = client.post("/under", json={"path": "Kits", "limit": 10}).json()
    assert set(body) == {"status", "models", "matched", "truncated",
                        "n_scanned", "covers"}
    assert body["status"] == "ok"
    assert [m["path"] for m in body["models"]] == [
        str(c.root / "Kits" / "A" / "y.stl"),
        str(c.root / "Kits" / "A-B" / "x.stl")]    # y first: 'A' < 'A-B'
    assert body["models"][0]["pose"]["up"] == [0.0, 0.0, 1.0]
    assert body["models"][1]["pose"] is None
    assert (body["matched"], body["truncated"]) == (2, False)
    assert "z.stl" not in str(body)                # the scope is the directory


def test_under_cuts_the_sorted_listing_at_the_limit(tmp_path):
    """`matched` is the scope before the cut, so a client showing 64 of 210 can
    say 210; `truncated` is this bound's own act. The cut takes a *prefix* of
    the sorted listing, which is why the order is contract rather than a
    courtesy: drop either the sort or the cut and a different model comes
    back."""
    client, _, c = serve_as_recorded(tmp_path)
    body = client.post("/under", json={"path": "Kits", "limit": 1}).json()
    assert body["matched"] == 2 and body["truncated"] is True
    assert [m["path"] for m in body["models"]] == \
        [str(c.root / "Kits" / "A" / "y.stl")]


def test_under_on_an_unindexed_directory_is_a_200_that_says_so(tmp_path):
    """The tri-state, through the same `resolve` every scoped route uses: a
    real directory nothing has been classified in is an answer, not a 404 —
    and never an empty `ok`, which is the one shape a UI cannot tell from
    "this folder is empty" (surface.md §scope)."""
    client, _, _ = serve(tmp_path, layout=["a/one.stl", "b/two.stl"],
                         embed=["a/one.stl"])
    r = client.post("/under", json={"path": "b", "limit": 10})
    assert r.status_code == 200
    assert r.json() == {"status": "unindexed", "models": [], "matched": 0,
                        "truncated": False, "n_scanned": 1, "covers": ["stl"]}
    # the other half of the distinction: `ok` never arrives empty, so the two
    # answers differ in the status rather than in a list a caller must read
    ok = client.post("/under", json={"path": "a", "limit": 10}).json()
    assert ok["status"] == "ok" and len(ok["models"]) == 1


def test_under_tells_a_folder_of_3mf_from_one_that_is_merely_unclassified(tmp_path):
    """The pair §scope's fields exist for, and the reason `status` alone
    cannot carry this route: both folders answer `unindexed` with `matched:
    0`, and the whole difference is `n_scanned`.

    Zero where the classify run's walk had nothing it could even look at —
    model-browser lists `.stl`, `.3mf` and `.obj`, `classify_stls.py` walks
    `.stl` — against one where it walked an STL that is simply not embedded
    yet. "Nothing here is searchable at all" and "nothing here *yet*" are
    different things to show a person, and a two-valued status re-created
    exactly the ambiguity the scope block was invented to remove (review,
    2026-09-01).

    `covers` is the constant that explains why zero is *possible*, not the
    field that discriminates — it is byte-identical in both answers. Pinned
    that way round on purpose: two readers in a row have taken `covers` for
    the discriminator."""
    client, _, _ = serve(tmp_path,
                         layout=["a/one.stl", "b/two.stl", "tiles/m.3mf"],
                         embed=["a/one.stl"])
    unsearchable = client.post("/under", json={"path": "tiles",
                                               "limit": 10}).json()
    not_yet = client.post("/under", json={"path": "b", "limit": 10}).json()

    assert unsearchable["status"] == not_yet["status"] == "unindexed"
    assert unsearchable["matched"] == not_yet["matched"] == 0
    assert unsearchable["n_scanned"] == 0      # the walk saw nothing to see
    assert not_yet["n_scanned"] == 1           # walked, not embedded yet
    assert unsearchable["covers"] == not_yet["covers"] == ["stl"]
    # and it is the *only* difference: everything else about the two answers,
    # `covers` included, is the same bytes
    assert unsearchable == {**not_yet, "n_scanned": 0}


def test_a_symlink_spelled_directory_reaches_the_same_rows(tmp_path):
    """Scoping through `resolve` is what buys this: it compares realpaths, and
    the library is on removable media that remounts under another name for the
    same tree. A listing route matching strings would answer `unindexed` for a
    folder full of indexed models after a remount."""
    client, _, c = serve(tmp_path, layout=UNDER)
    link = tmp_path / "by-another-name"
    link.symlink_to(c.root / "Kits")
    canonical = client.post("/under", json={"path": "Kits", "limit": 10}).json()
    aliased = client.post("/under", json={"path": str(link), "limit": 10}).json()
    assert aliased == canonical and canonical["matched"] == 2


def test_under_names_a_model_exactly_as_a_hit_does(tmp_path):
    """The consistency the contract rests on: a caller lists a folder and then
    hands one of those strings back to `/poses` or `/similar`. One spelling for
    one model, and the pose block byte-identical to the one a search carried."""
    client, _, _ = serve(tmp_path, layout=["Kits/A/y.stl"],
                         ups={"Kits/A/y.stl": [0.0, 1.0, 0.0]},
                         front={"Kits/A/y.stl": 1})
    hit = client.post("/query", json={"text": "x"}).json()["results"][0]
    m = client.post("/under", json={"path": "Kits", "limit": 10}).json()["models"][0]
    assert set(m) == {"path", "pose"}
    assert m["path"] == hit["path"]
    assert m["pose"] == hit["pose"] and m["pose"]["up"] == [0.0, 1.0, 0.0]
    # ...and the string round-trips through the batch route unnormalised
    poses = client.post("/poses", json={"paths": [m["path"]]}).json()["poses"]
    assert poses[m["path"]] == m["pose"]


@pytest.mark.parametrize("body", [
    {"path": "Kits"},                       # no count: an unbounded listing
    {"path": "Kits", "limit": 0},
    {"path": "Kits", "limit": -1},
    {"path": "Kits", "limit": "many"},
    {"limit": 10},                          # no directory
])
def test_under_refuses_a_request_without_a_usable_limit(tmp_path, body):
    """`limit` is required — the one bound on this surface that is. Absent
    would mean a response bounded by the collection rather than by the
    request, and the refusal is pydantic's own 422, the same answer every
    other malformed field already gets."""
    client = client_of(tmp_path, layout=UNDER)
    assert client.post("/under", json=body).status_code == 422


def test_the_two_response_bounds_stop_at_one_shared_ceiling(tmp_path):
    """`/query`'s `cap` and `/under`'s `limit` are bounded by the same number,
    because it is one statement about what this server will serialise and not
    two opinions about two routes. It lived as two unshared literals until
    2026-09-01 (review), where either could have been raised without the
    other and surface.md would still have described one ceiling. Asserted at
    the edge on both routes, and against the number the document states."""
    assert RESPONSE_CAP == 10000                 # what surface.md publishes
    client = client_of(tmp_path, layout=UNDER)
    at = [client.post("/query", json={"text": "x", "cap": RESPONSE_CAP}),
          client.post("/under", json={"path": "Kits", "limit": RESPONSE_CAP})]
    over = [client.post("/query", json={"text": "x", "cap": RESPONSE_CAP + 1}),
            client.post("/under", json={"path": "Kits",
                                        "limit": RESPONSE_CAP + 1})]
    assert [r.status_code for r in at] == [200, 200]
    assert [r.status_code for r in over] == [422, 422]


@pytest.mark.parametrize("path,code", [
    ("a/pack.zip!/inner.stl", 422),         # zip virtual path: unaddressable
    ("/etc", 400),                          # outside the collection
    ("a/nowhere", 404),                     # not on disk
])
def test_under_maps_each_scope_rejection_the_way_query_does(tmp_path, path, code):
    """Same scoper, same three codes, asserted against `/query` in the same
    breath: a consumer branching on them must not need a per-route table."""
    client, _, _ = serve(tmp_path)
    assert client.post("/under", json={"path": path,
                                       "limit": 10}).status_code == code
    assert client.post("/query", json={"text": "x",
                                       "path": path}).status_code == code


def test_under_never_touches_the_gpu_lock(tmp_path):
    """A store scan: `resolve`, tuple arithmetic, dict gets. If this ever took
    the lock a folder's listing could queue behind a query, which is the cost
    the no-GPU framing in surface.md promises it cannot pay."""
    client, state, _ = serve(tmp_path)
    state.embed = lambda *a, **k: pytest.fail("/under embedded something")
    held = []

    class Watching:
        def __enter__(self): held.append(1)
        def __exit__(self, *a): pass

    state.gpu = Watching()
    r = client.post("/under", json={"path": "a", "limit": 10})
    assert r.status_code == 200 and held == []



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


def test_status_says_the_volume_was_never_required_under_no_volume(tmp_path):
    """`present: false` is a claim — looked, and the drive is gone — that a
    consumer acts on by telling someone to plug it in. A server that was never
    going to look must not make it, and `null` alone does not say which: it is
    also what a warming server reports. `required` is the field that separates
    them (surface.md §Process parameters)."""
    import shutil
    client, _, c = serve(tmp_path, layout=["a/one.stl"], no_volume=True)
    shutil.rmtree(c.root)

    s = client.get("/status").json()
    assert s["volume"] == {"present": None, "root": str(c.root),
                           "missing": None, "required": False}
    assert s["failure"] is None and s["ready"] is True
    assert client.post("/query", json={"text": "x"}).status_code == 200


@pytest.mark.parametrize("no_volume,required", [(False, True), (True, False)])
def test_a_warming_server_already_says_whether_it_needs_a_volume(tmp_path,
                                                                 no_volume,
                                                                 required):
    """`required` is read off the process's flags, not off a load, so it is
    answerable before one — and it has to be answered there. `present: null`
    is both "still warming" and "never going to look", and this key is the
    only thing that separates them; leaving it out of the warming block
    dropped it from precisely the state it was invented for."""
    client, _, _ = serve(tmp_path / str(no_volume), layout=["a/one.stl"],
                         ready=False, no_volume=no_volume)
    v = client.get("/status").json()["volume"]
    assert v["present"] is None                  # nothing loaded yet, either way
    assert v["required"] is required


def test_a_no_volume_server_refuses_a_rescan_and_reloads_without_one(tmp_path):
    """A rescan re-walks the input directory, which `--no-volume` promised the
    process would never touch — refused rather than quietly downgraded, on the
    CLI's --repose precedent. The plain reload is the one that matters anyway:
    re-reading pose-cache.json and the `.npy` files is how a fresh classify run
    reaches a manifest-mode server, and it works with the volume gone."""
    import shutil
    client, state, c = serve(tmp_path, layout=["a/one.stl"], no_volume=True)
    build(tmp_path, ["a/one.stl", "a/two.stl"])          # a classify run happened
    shutil.rmtree(c.root)                                # and the drive left

    r = client.post("/reload", json={"rescan": True})
    assert r.status_code == 400
    assert "--no-volume" in r.json()["detail"]
    assert state.collection is c                         # nothing was rebound

    body = client.post("/reload", json={}).json()
    assert body["n_models"] == 2                         # the new model, no walk
    assert body["volume"]["required"] is False
    assert state.collection is not c


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


# --- what the terminal says -------------------------------------------------
#
# Every state `/status` reports gets exactly one line at the moment it
# changes. The operator report these exist for (2026-08-31): a serve_api
# started against a cache this code cannot read sat at `ready: false` for
# 300+ s with a silent terminal — the failure was visible only to a client
# polling `/status`, which is not who was watching. The fragments asserted
# here are the load-bearing ones (the kind, the reason, the counts); the
# wording around them is free to change.

def api_lines(caplog, level=None):
    """The `mini_classify.api` records, rendered — nothing else's."""
    return [r.getMessage() for r in caplog.records
            if r.name == "mini_classify.api"
            and (level is None or r.levelno == level)]


def test_a_failed_warmup_says_why_in_the_log(caplog, tmp_path):
    """The reported case. `/status` had the whole story and said it to
    whoever asked; the terminal said nothing at all."""
    from src.collection import CacheUnusable

    def boom():
        raise CacheUnusable("cache_version 1, this code expects 2",
                            "run: migrate_cache_keys.py --apply")

    _, state, _ = serve(tmp_path, ready=False)
    with caplog.at_level(logging.INFO, logger="mini_classify.api"):
        state.warm(boom, lambda: (None, None, None))

    failed = api_lines(caplog, logging.ERROR)
    assert len(failed) == 1                       # one line, not a banner
    assert "CacheUnusable" in failed[0]           # the kind /status reports
    assert "cache_version" in failed[0]           # ...and its reason
    assert "migrate_cache_keys" in failed[0]      # ...and the actionable half


def test_a_failed_reload_says_why_in_the_log(caplog, tmp_path):
    """The same for the retry path: a reload that cannot complete answers the
    caller with a 409 and the terminal with a reason."""
    import shutil
    client, state, c = serve(tmp_path, layout=["a/one.stl"])
    shutil.rmtree(c.root)

    with caplog.at_level(logging.INFO, logger="mini_classify.api"):
        assert client.post("/reload", json={}).status_code == 409

    failed = api_lines(caplog, logging.ERROR)
    assert len(failed) == 1
    assert "VolumeUnavailable" in failed[0]
    assert str(c.root) in failed[0]               # *which* volume is gone
    # and a reload that never returned would still be visible as one that ran
    assert any("reload" in ln for ln in api_lines(caplog, logging.INFO))


def test_a_finished_warmup_logs_one_line_at_info(caplog, tmp_path):
    """The healthy transition: what loaded, where, how long."""
    args, root, _ = build(tmp_path, ["a/one.stl", "a/two.stl"])
    state = ServerState(args)

    with caplog.at_level(logging.INFO, logger="mini_classify.api"):
        state.warm(lambda: Collection.load(args),
                   lambda: (stub_embed(), "stub-model", "cpu"))

    assert state.ready
    info = api_lines(caplog, logging.INFO)
    assert len(info) == 2                         # started, then finished
    assert "2 models" in info[-1] and "cpu" in info[-1]
    assert not [r for r in caplog.records         # nothing louder on a good day
                if r.name == "mini_classify.api" and r.levelno > logging.INFO]


def test_a_superseded_warmup_says_what_it_discarded(caplog, tmp_path):
    """The generation counter's whole job, and until now invisible: warm
    finished, wrote nothing over the reload that beat it, and left no trace
    of having run at all (the interleaving of
    `test_a_finished_warmup_does_not_revert_a_reload_that_landed_inside_it`)."""
    args, root, _ = build(tmp_path, ["a/one.stl"])
    stale = Collection.load(args)
    state = ServerState(args)
    client = TestClient(create_app(state), raise_server_exceptions=False)

    def load_collection_with_reload_inside():
        assert client.post("/reload", json={}).status_code == 200
        return stale

    with caplog.at_level(logging.INFO, logger="mini_classify.api"):
        state.warm(load_collection_with_reload_inside,
                   lambda: (stub_embed(), "stub-model", "cpu"))

    warned = api_lines(caplog, logging.WARNING)
    assert len(warned) == 1
    assert "superseded" in warned[0]
    assert "collection" in warned[0]              # what was thrown away
    assert not api_lines(caplog, logging.ERROR)   # a supersede is not a failure


def test_a_blank_first_line_reads_the_same_in_the_log_and_in_status(caplog,
                                                                    tmp_path):
    """The envelope and the terminal name one reason, not two.

    `/status` used to take the literal first line of the exception, so the
    multi-line case `_why` was written for — a missing-backend `ImportError`,
    whose message opens with a blank line — answered `reason: ""` to the
    client polling while the log carried the real text (adversarial review,
    2026-08-31)."""
    def boom():
        raise ImportError("\nSiglipModel requires the PyTorch library but it "
                          "was not found in your environment.")

    client, state, _ = serve(tmp_path, ready=False)
    with caplog.at_level(logging.INFO, logger="mini_classify.api"):
        state.warm(boom, lambda: (None, None, None))

    reason = client.get("/status").json()["failure"]["reason"]
    assert reason.startswith("SiglipModel requires")   # not "", not "ImportError"
    assert reason in api_lines(caplog, logging.ERROR)[0]


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
    pytest.param(lambda cl: cl.post("/poses", json={"paths": ["a/one.stl"]}),
                 id="poses"),
    pytest.param(lambda cl: cl.post("/under", json={"path": "a", "limit": 10}),
                 id="under"),
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
    # the same pose block through a different envelope: `up` and
    # `azimuth_zero` come off numpy and a stringified float would be worse
    # than a missing one
    payload = client.post("/poses", json={"paths": ["a/one.stl"]}).json()
    json.dumps(payload)
    block = payload["poses"]["a/one.stl"]
    assert type(block["confidence"]) is float
    for field in ("up", "azimuth_zero"):
        assert all(type(x) is float for x in block[field]), field
    # and once more through the listing envelope, which reaches that same
    # block by a third route
    payload = client.post("/under", json={"path": "a", "limit": 10}).json()
    json.dumps(payload)
    assert type(payload["matched"]) is int
    assert all(type(m["path"]) is str for m in payload["models"])
