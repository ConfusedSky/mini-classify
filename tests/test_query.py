"""Query-path tests (interfaces.md §the import-rule table, row `query`).

`src/query.py` exists because two consumers need the same formula — the REPL
and the API sketched in docs/api/surface.md — so what is pinned here is the
*contract between them*, not the arithmetic: which cut `rank` applies, that
`weak` judges the query rather than the cut, that `z` and `scores` stay
full-length when `order` is cut, and that an empty scope is answerable rather
than fatal.

Two oracles guard the extraction itself: `repl_show_query` replays the
pre-extraction `test_categories.show_query` body line for line (the same shape
of pin `tests/test_done.py` uses for the score block), and the module is
imported bare to hold the torch-free rule.

numpy only; no torch, no GPU, no cache.
"""
import warnings

import numpy as np
import pytest

from src import query
from src.query import WEAK_Z, Ranked, pool_sims, rank, robust_z


# --- fixtures ----------------------------------------------------------------

def spread(peak):
    """Nine scores whose median is 0 and whose MAD is exactly 1, plus `peak`.

    Hand-checkable so the weak boundary can be asserted against a number
    instead of against the function under test: sorted, the sample is
    [-2,-1,-1,0,0,1,1,2,peak] (median 0) and its absolute deviations are
    [0,0,1,1,1,1,2,2,peak] (median 1), for any peak > 2. So
    z(peak) == peak / 1.4826, and WEAK_Z is crossed at peak 2.9652."""
    return np.array([-2, -1, -1, 0, 0, 1, 1, 2, peak], dtype=np.float32)


def views(seed=5, n_files=6, n_views=4, n_cats=3):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_files, n_views, n_cats)).astype(np.float32)


def repl_show_query(sims_1d, top=10, min_score=None):
    """`test_categories.show_query`'s body as it stood before the extraction
    — the parity oracle for the ranking decisions, minus the printing.

    Returns what the REPL would have *shown*: the weak verdict, or the indices
    it would have listed."""
    med = np.median(sims_1d)
    mad = np.median(np.abs(sims_1d - med)) * 1.4826 + 1e-9
    z = (sims_1d - med) / mad
    order = np.argsort(-sims_1d)
    if z[order[0]] < 2.0:
        return ("weak", z[order[0]])
    if min_score is not None:
        order = order[sims_1d[order] >= min_score]
    else:
        order = order[:top]
    return ("list", [(int(i), z[i]) for i in order])


def as_repl(r: Ranked):
    """The same tuple, read off a `Ranked`, the way the REPL reads it now."""
    if r.weak:
        return ("weak", r.z[r.best])
    return ("list", [(int(i), r.z[i]) for i in r.order])


def assert_same_listing(sims, want, got):
    """The two listings agree, allowing for the one deliberate divergence.

    The extraction changed exactly one thing about the output: `argsort` is
    stable now, so exact ties break by ascending index instead of by whatever
    quicksort did. What that may move is *which* models are listed, not only
    their order — a tie straddling the top-N cut hands the last slot to a
    different model of equal score (measured: 105 of 16000 fuzzed cases, and
    `test_a_tie_at_the_top_n_boundary_goes_to_the_lower_index` pins which one
    wins). So membership is deliberately not asserted here; asserting it would
    contradict that test and would fire spuriously the moment this oracle's
    fixtures got tie-dense.

    What does survive any tie-break, and is therefore what a real regression
    would have to break: the sequence of scores listed, and the sequence of z
    values listed. Unequal lengths fail these too."""
    want_i, got_i = [i for i, _ in want], [i for i, _ in got]
    assert np.array_equal(sims[want_i], sims[got_i])                # scores, in order
    assert np.allclose([z for _, z in want], [z for _, z in got])   # z, in order


# --- pool_sims: the one copy (it moved here from `done`, 2026-08-19) ---------

def test_mean_and_max_pool_over_the_view_axis():
    v = np.array([[[1.0, 0.0], [3.0, 0.0], [2.0, 6.0]]], dtype=np.float32)
    assert pool_sims(v, "mean").tolist() == [[2.0, 2.0]]
    assert pool_sims(v, "max").tolist() == [[3.0, 6.0]]


def test_pooling_collapses_views_and_keeps_files_and_categories():
    v = views()
    for mode in ("mean", "max", "softmax"):
        assert pool_sims(v, mode).shape == (6, 3)


def test_softmax_sits_between_mean_and_max():
    # the documented "in between" claim: a feature visible in one view of four
    # counts for more than 25% but does not decide the score outright
    v = views()
    lo, mid, hi = (pool_sims(v, m) for m in ("mean", "softmax", "max"))
    assert np.all(lo <= mid + 1e-6) and np.all(mid <= hi + 1e-6)
    assert np.any(mid > lo + 1e-3)   # not silently equal to the mean


def test_softmax_is_finite_when_the_scores_are_large():
    # BETA=50 against un-shifted scores would overflow; the max subtraction in
    # the body is what keeps it finite, so pin it rather than trust it
    assert np.all(np.isfinite(pool_sims(views() * 100, "softmax")))


def test_the_view_axis_is_selectable():
    v = views()
    assert np.allclose(pool_sims(v, "mean", axis=0), v.mean(0))


@pytest.mark.parametrize("mode", ["meen", "", None, "MEAN", "softmax "])
def test_an_unknown_pool_mode_is_rejected_rather_than_softmaxed(mode):
    # the softmax branch is the fall-through, so an unrecognised mode used to
    # return softmax scores and a caller could not tell. That was survivable
    # while every caller came through argparse `choices`; `pool` is a
    # per-request field in docs/api/surface.md, so the check belongs here now
    with pytest.raises(ValueError, match="unknown pool mode"):
        pool_sims(views(), mode)


# --- robust_z: comparable across queries, unskewed by a full category -------

def test_z_is_the_median_mad_distance():
    z = robust_z(spread(4.0))
    assert z[8] == pytest.approx(4.0 / 1.4826, rel=1e-6)
    assert z[3] == pytest.approx(0.0, abs=1e-6)


def test_a_well_represented_category_does_not_look_weak():
    # the reason median/MAD replaced mean/std: eight genuine matches drag the
    # mean up to sit among them, and the query flags itself as noise
    sims = np.array([0.05] * 20 + [0.30] * 8, dtype=np.float32)
    assert robust_z(sims)[-1] > WEAK_Z
    mean_std_z = (sims[-1] - sims.mean()) / sims.std()
    assert mean_std_z < WEAK_Z          # what the rejected formula would say


def test_a_flat_collection_is_all_zeros_rather_than_nan():
    # MAD is 0 here; the epsilon exists so this is 0/eps, not 0/0
    z = robust_z(np.full(10, 0.3, dtype=np.float32))
    assert np.all(z == 0.0) and np.all(np.isfinite(z))


# --- rank: the weak verdict, and what it is read off ------------------------

def test_the_weak_boundary_is_WEAK_Z():
    assert WEAK_Z == 2.0
    assert rank(spread(2.9)).weak is True         # z 1.96
    assert rank(spread(3.1)).weak is False        # z 2.09


def test_weak_is_judged_on_the_best_score_before_any_cut():
    r = rank(spread(3.1), top=1)
    assert r.best == 8 and r.weak is False
    assert r.z[r.best] > WEAK_Z


def test_weak_judges_the_query_not_the_cut():
    # the docstring's promise, and the REPL/API split depends on it: a caller
    # that shows weak hits still needs `order` populated
    r = rank(spread(2.9))
    assert r.weak is True and len(r.order) > 0


def test_a_weak_query_still_reports_its_best():
    # `best` is a field precisely so "nothing cleared the floor" can still name
    # the query's best z — a floor that empties `order` must not hide it
    r = rank(spread(2.9), min_score=99.0)
    assert len(r.order) == 0
    assert r.best == 8 and np.isfinite(r.z[r.best])


# --- rank: the two cuts -----------------------------------------------------

def test_the_default_cut_is_top_n_best_first():
    sims = np.array([0.1, 0.9, 0.3, 0.7, 0.5], dtype=np.float32)
    r = rank(sims, top=3)
    assert r.order.tolist() == [1, 3, 4]
    assert np.all(np.diff(sims[r.order]) <= 0)


def test_min_score_replaces_the_top_n_cut_with_a_floor():
    # not "the floor applied to the top 10": an exhaustive listing is the point
    sims = np.linspace(0.0, 1.0, 40, dtype=np.float32)
    r = rank(sims, top=10, min_score=0.5)
    assert len(r.order) == 20
    assert np.all(sims[r.order] >= 0.5)


def test_a_floor_that_nothing_clears_returns_an_empty_order():
    r = rank(spread(3.1), min_score=99.0)
    assert r.order.tolist() == []


def test_z_and_scores_stay_full_length_when_order_is_cut():
    # they are per-model over everything scored; `order` indexes into them
    sims = spread(3.1)
    r = rank(sims, top=2)
    assert len(r.order) == 2
    assert len(r.z) == len(sims) and len(r.scores) == len(sims)
    assert r.scores is sims or np.array_equal(r.scores, sims)


def test_ties_break_by_collection_order():
    # duplicated kits render byte-identically, so exact ties are real here;
    # `kind="stable"` is what makes two consumers agree on the same listing.
    # Three tie groups (1&2, 3&4, 5&6), each of which must come out ascending
    # — the unstable sort this replaced ordered 5 and 6 the other way round,
    # which is what makes this fixture discriminating rather than lucky
    assert rank(spread(3.1), top=9).order.tolist() == [8, 7, 5, 6, 3, 4, 1, 2, 0]


def test_a_tie_at_the_top_n_boundary_goes_to_the_lower_index():
    # not just the order of what is listed: when a tie straddles the cut, the
    # tie-break decides *which* model is listed at all. Three models tie for
    # the last slot and the first of them takes it
    sims = np.array([5.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert rank(sims, top=2).order.tolist() == [0, 1]


# --- rank: an empty scope is answerable, not fatal --------------------------

def test_an_empty_scope_returns_an_empty_ranking():
    # docs/api/surface.md's scoped search slices the matrix before scoring, so
    # a scope matching no files reaches `rank` with nothing in it
    r = rank(np.array([], dtype=np.float32))
    assert r.best is None            # not -1: a valid index would read the tail
    assert r.order.tolist() == [] and len(r.z) == 0 and len(r.scores) == 0
    assert r.weak is False           # a statement about the scope, not the query


def test_an_empty_scope_is_warning_free():
    # the guard has to return before `robust_z`, not merely before `order[0]`:
    # np.median of an empty array is nan and emits two RuntimeWarnings on the
    # way. Silent nan is exactly what a later refactor reintroduces unnoticed
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rank(np.array([], dtype=np.float32))


def test_an_empty_scope_survives_a_floor_too():
    r = rank(np.array([], dtype=np.float32), min_score=0.5)
    assert r.best is None and r.order.tolist() == []


# --- Ranked: comparing or hashing one must never raise ----------------------

def test_two_rankings_compare_without_raising():
    # frozen=True with numpy fields generates an __eq__ that raises ValueError
    # ("truth value of an array is ambiguous") on any two distinct instances —
    # identity shortcuts hide it, so these must be two separate calls
    a, b = rank(spread(3.1)), rank(spread(3.1))
    assert a is not b
    assert (a == b) is False        # eq=False: identity, not value equality
    assert (a == a) is True
    # value equality is deliberately not offered; compare the fields directly


def test_a_ranking_is_hashable():
    a, b = rank(spread(3.1)), rank(spread(3.1))
    assert len({a, b, a}) == 2


# --- score: the caller owns the text pass, and the scope is a row slice -----

def test_score_is_the_pooled_matmul():
    rng = np.random.default_rng(11)
    matrix = rng.standard_normal((6, 4, 8)).astype(np.float32)
    text_T = rng.standard_normal((8, 3)).astype(np.float32)
    for mode in ("mean", "max", "softmax"):
        assert np.allclose(query.score(matrix, text_T, mode),
                           pool_sims(matrix @ text_T, mode))


def test_a_scoped_search_is_the_same_function_over_a_row_subset():
    # the docstring's claim, and why no path filtering lives in the module:
    # slicing before the call must not change any surviving row's score
    rng = np.random.default_rng(12)
    matrix = rng.standard_normal((10, 4, 8)).astype(np.float32)
    text_T = rng.standard_normal((8, 2)).astype(np.float32)
    keep = [1, 4, 7]
    assert np.allclose(query.score(matrix[keep], text_T, "softmax"),
                       query.score(matrix, text_T, "softmax")[keep])


def test_a_scope_matching_nothing_flows_through_to_rank():
    # the end-to-end shape of the API's empty scope, not just rank's guard
    rng = np.random.default_rng(13)
    matrix = rng.standard_normal((10, 4, 8)).astype(np.float32)
    text_T = rng.standard_normal((8, 1)).astype(np.float32)
    sims = query.score(matrix[[]], text_T, "softmax").ravel()
    assert sims.shape == (0,)
    assert rank(sims).best is None


# --- parity with the REPL the module was extracted from ---------------------

@pytest.mark.parametrize("peak", [2.5, 2.9, 3.1, 8.0])
@pytest.mark.parametrize("min_score", [None, -5.0, 1.0, 99.0])
def test_ranking_matches_the_pre_extraction_repl(peak, min_score):
    sims = spread(peak)
    want = repl_show_query(sims, min_score=min_score)
    got = as_repl(rank(sims, min_score=min_score))
    assert want[0] == got[0]
    if want[0] == "weak":
        assert want[1] == pytest.approx(got[1])
    else:
        assert_same_listing(sims, want[1], got[1])


def test_parity_holds_over_random_collections():
    # the REPL's real input shape: one pooled score per model, no structure
    rng = np.random.default_rng(17)
    for trial in range(200):
        sims = rng.normal(size=int(rng.integers(2, 130))).astype(np.float32)
        if trial % 2 == 0:
            sims = np.round(sims, 1)                # exact ties, deliberately:
            # continuous float32 normals essentially never tie, so a fuzz over
            # them cannot see the tie-break at all — which is how the change's
            # effect on top-N membership went unmeasured in the first place
        if trial % 3 == 0:
            sims[rng.integers(len(sims))] += 8      # force the non-weak arm
        want, got = repl_show_query(sims), as_repl(rank(sims))
        assert want[0] == got[0]
        if want[0] == "weak":
            assert want[1] == pytest.approx(got[1])
        else:
            assert_same_listing(sims, want[1], got[1])


# --- the import rule --------------------------------------------------------

def test_the_query_path_costs_no_torch():
    # interfaces.md row `query`: numpy and stdlib only. The module exists so a
    # caller can pool an array without loading the pipeline's terminal stage
    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; import src.query; "
         "print(any(m in sys.modules for m in ('torch', 'open3d')))"],
        capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
