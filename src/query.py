"""Querying the cached embeddings: view pooling, robust z, ranking.

The decisions behind `test_categories.py`'s query loop, with the printing and
the SigLIP model left to the caller. Two consumers, which is why it exists: the
REPL, and the query API sketched in `docs/api/surface.md`. Neither may own a
formula the other also owns — that is the mistake the eval-debt cleanup spent
2026-08-18 undoing, where one tool imported scoring out of another.

The split is: **this module decides, the caller renders.** `rank` returns the
order, the z scores and the weak verdict; whether a weak query prints a warning
(the REPL) or ships the hits with a flag (the API) is a presentation call, and
the two answer it differently.

Deliberately numpy-only. Text embedding is the caller's — it holds the model,
and hands in a `(dim, n_texts)` matrix of unit rows. `pool_sims` moved here
from `src/done.py` for that reason: it is pure numpy, and every caller that
wanted it out of `done` was importing the pipeline's terminal stage (torch,
csv, the transport) to pool an array. `done` imports it from here now, which
is the direction the dependency belonged in — scoring is a leaf, and the stage
that scores is the consumer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Below this, a query is noise: nothing in the collection stands out from the
# collection's own spread. Measured (docs/learnings/queries-and-filters.md):
# correct matches ran z 2.4+, and no cutoff separates a modest correct match
# from a semantic near-miss (a "witch on a broomstick" query hit a mounted
# rider at 3.7), so 2.0 is set to catch only unambiguous noise. Raising it
# suppresses good results without stopping near-misses.
WEAK_Z = 2.0


def pool_sims(view_sims, mode, axis=-2):
    """Pool per-view similarity scores (..., n_views, n_categories) over views.

    mean: robust whole-object consensus (a feature seen in 1 of 4 views keeps
    ~25% weight). max: "clearly visible from some angle" — lets single-view
    features decide. softmax: in between (sharpness set by BETA).

    The one copy: `done` scores with it, the REPL and the evals query with it.
    It lived in `done` until `src/query.py` existed to hold it (2026-08-19).

    Unknown modes raise. While every caller was argparse-constrained this
    function could treat softmax as the fallback and never be wrong; the API's
    per-request `pool` field (docs/api/surface.md `POST /query`) arrives
    untrusted, and a typo silently answering with softmax is a 200 that should
    have been a 400. The check belongs here rather than in each caller because
    this is the function that knows the set."""
    if mode == "mean":
        return view_sims.mean(axis)
    if mode == "max":
        return view_sims.max(axis)
    if mode != "softmax":
        raise ValueError(f"unknown pool mode {mode!r} (mean, max or softmax)")
    BETA = 50.0
    w = np.exp(BETA * (view_sims - view_sims.max(axis, keepdims=True)))
    return (w * view_sims).sum(axis) / w.sum(axis)


def score(matrix, text_T, pool):
    """(n_files, n_texts) pooled cosine similarity.

    `matrix` is the cached embeddings, (n_files, n_views, dim) with unit rows
    (`embed_store.load_embedding_matrix`); `text_T` is (dim, n_texts), also
    unit rows, from whichever text pass the caller wanted — templated
    (`embedder.embed_texts`) or verbatim (`embedder.embed_raw`). The two are
    the same matmul either way, so the choice stays entirely the caller's.

    A scoped search is this function over a row subset of `matrix`: slice
    before calling and everything downstream — z included — narrows with it.
    That is why no path filtering lives here."""
    return pool_sims(matrix @ text_T, pool)


def robust_z(sims):
    """How far each model stands out from the collection, for one query.

    Cosine scores are comparable only *within* a query — some phrasings run hot,
    some cold — so a raw threshold cannot tell "present" from "absent"; something
    always ranks first. z is comparable across queries.

    Median/MAD rather than mean/std: a well-represented category (eight robots)
    drags the mean up and makes its own query look weak. The measurement is in
    docs/learnings/queries-and-filters.md."""
    med = np.median(sims)
    mad = np.median(np.abs(sims - med)) * 1.4826 + 1e-9
    return (sims - med) / mad


@dataclass(frozen=True, eq=False)
class Ranked:
    """One query's verdict. `z` and `scores` are per-model over everything
    scored; `order` indexes into them, best first, already cut to `top` or
    filtered by `min_score`. `weak` judges the *query*, not the cut, so it is
    still set when `order` is non-empty — a caller may show weak hits.

    `best` is the top-scoring model before any cut, and is what the weak
    verdict was read off. It is a separate field because `order` is not: a
    `min_score` floor can empty `order` entirely, and "no model cleared the
    floor" must still be able to report the query's best z. It is `None` only
    when nothing was scored at all — `None` rather than `-1` because `-1` is a
    perfectly good numpy index, so the sentinel a caller forgets to check
    would silently read the last model instead of failing.

    `eq=False`: the generated `__eq__` compares numpy fields and raises
    ValueError on the ambiguous truth value (for *distinct* instances — tuple
    comparison shortcuts on identity, so `r == r` misleadingly works), and the
    generated `__hash__` raises TypeError on an unhashable array. Nothing
    compares or hashes a result; identity semantics are the honest ones."""
    order: np.ndarray
    z: np.ndarray
    scores: np.ndarray
    weak: bool
    best: int | None


def rank(sims, top=10, min_score=None):
    """Rank one query's scores. `sims` is 1-D, one pooled score per model.

    `min_score` replaces the top-N cut with a floor: every model at or above
    it, which is what an exhaustive listing wants. It is opt-in because the
    top-N cut already hides most of what it would catch (`--min-score`/`:min`),
    and it can legitimately return nothing.

    Scoring nothing is not an error: a path-scoped query whose directory holds
    no cached models slices `matrix` to zero rows, and that is a request the
    API answers with zero hits, not an exception (`docs/api/surface.md`, the
    `unindexed` scope status). The early return is before `robust_z` and not
    merely before the ranking: `np.median` of an empty array warns twice and
    returns nan, so guarding the index alone would leave the noise behind."""
    if len(sims) == 0:
        return Ranked(order=np.empty(0, dtype=np.intp), z=sims, scores=sims,
                      weak=False, best=None)
    z = robust_z(sims)
    # stable, so exact ties break by ascending index rather than by whatever
    # quicksort did with them: two byte-identical renders across duplicated
    # kits tie honestly, and `best` is a field two consumers read. Note this
    # decides membership, not just order — when a tie straddles the top-N cut
    # the tie-break picks *which* model is listed at all (measured: 2124 of
    # 16000 fuzzed cut-biting cases substitute a model, always at an equal
    # score). Deterministic either way; this way the two consumers agree.
    order = np.argsort(-sims, kind="stable")
    best = int(order[0])
    weak = bool(z[best] < WEAK_Z)
    if min_score is not None:
        order = order[sims[order] >= min_score]
    else:
        order = order[:top]
    return Ranked(order=order, z=z, scores=sims, weak=weak, best=best)
