"""src/poser.py: the Poser lifecycle against fake futures and a recording
record_pose stub (interfaces.md §Poser). The ensemble math itself is
src/pose's and has its own tests — here the inputs are crafted so its
answers are known, and what is pinned is the calling convention: stash /
embed-request, record-before-park (I15's floor), the single `Resolved` exit
the driver re-routes, park/resume through poll, `drop`, the fold_done/settle
abort split, settle's timeout abandonment and its CancelledError skip, and
J3's fold-error → Failure. No VLM call is ever real: the arbiter is a fake
returning manually driven Futures.

Note what is NOT here any more: nothing asserts on a task the Poser built,
because it builds none. Every answer is `Resolved(file, index)` and the
driver re-routes it through `route` (the second-call rule) — a test here
that expected an `EmbedRenderTask` would be pinning the re-embed regression
back into place."""
import threading
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest
import torch

from src import pose
from src.messages import Failure, PoseTiles, Resolved, TileEmbeds
from src.poser import Poser, VlmConfig

# One text-probe dimension: upright_scores == 2 * embed value, so the tile
# embeddings below hand the SigLIP vote to exactly one candidate.
UP_T = np.array([[1.0]])
DOWN_T = np.array([[-1.0]])

F = Path("models/thing.stl")

# strong flat-base evidence for candidate 0: geometry votes at full weight
GEO_CONFIDENT = np.array([0.5, 0.01, 0.0, 0.0, 0.0, 0.0])
# best score far under ABS_SCORE_FLOOR: geometry's vote is nearly silenced
GEO_BASELESS = np.array([0.001, 0.0009, 0.0, 0.0, 0.0, 0.0])


class FakeArbiter:
    """Captures submitted calls; hands back real Futures the test drives."""

    def __init__(self):
        self.calls, self.futures = [], []

    def submit(self, call):
        f = Future()
        self.calls.append(call)
        self.futures.append(f)
        return f


class RecordingDone:
    def __init__(self):
        self.poses = []                     # (file, index, Pose)

    def record_pose(self, file, index, p):
        self.poses.append((file, index, p))


def tile(v):
    return np.full((4, 4, 3), v, dtype=np.uint8)


def grid_tiles(n_az=2):
    # 6 candidates x n_az azimuths; pixel value encodes (candidate, azimuth)
    return [[tile(c * 10 + a) for a in range(n_az)] for c in range(6)]


def sig_embeds(win, n_az=2):
    e = np.zeros((6 * n_az, 1), dtype=np.float32)
    e[win * n_az:(win + 1) * n_az] = 1.0
    return torch.from_numpy(e)


class FakeClock:
    """The breaker's injected clock (C5): time only moves when a test says so."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def make_poser(done=None, arb=None, clock=None, **cfg):
    done = done or RecordingDone()
    arb = arb or FakeArbiter()
    poser = Poser(UP_T, DOWN_T, arb, done.record_pose, VlmConfig(**cfg),
                  clock=clock or FakeClock())
    return poser, done, arb


def feed(poser, win=0, geo=GEO_CONFIDENT, index=7, n_az=2, file=F):
    """One file through on_tiles → on_tile_embeds, SigLIP favouring `win`."""
    poser.on_tiles(PoseTiles(file=file, index=index, geo_scores=geo,
                             tiles=grid_tiles(n_az)))
    return poser.on_tile_embeds(
        TileEmbeds(file=file, index=index, embeds=sig_embeds(win, n_az)))


ESCALATE = dict(backend="gemini", margin_threshold=5.0,   # everything parks;
                ask=lambda tiles: None)                   # never really called


# --- on_tiles ---------------------------------------------------------------

def test_on_tiles_stashes_and_returns_stacked_request():
    poser, _, _ = make_poser()
    req = poser.on_tiles(PoseTiles(file=F, index=3, geo_scores=GEO_CONFIDENT,
                                   tiles=grid_tiles()))
    assert (req.file, req.index) == (F, 3)
    assert req.tiles.shape == (12, 4, 4, 3)
    # candidate-major, azimuth-minor — the grid's own order, preserved
    assert req.tiles[0][0, 0, 0] == 0      # c0 az0
    assert req.tiles[1][0, 0, 0] == 1      # c0 az1
    assert req.tiles[2][0, 0, 0] == 10     # c1 az0
    assert 3 in poser._stash


# --- VlmConfig ---------------------------------------------------------------

def test_vlm_config_rejects_ollama_at_construction():
    """C-R1-4: the pool has no inline arm, and a pooled ollama call would
    overlap SigLIP on the 4060 (10.1 s reload vs 0.49 s inference — CLAUDE.md).
    The flag is retired, so the shape that carries it must say so."""
    with pytest.raises(ValueError, match="ollama"):
        VlmConfig(backend="ollama")


def test_vlm_config_accepts_the_two_remote_backends_and_none():
    for backend in (None, "gemini", "claude"):
        assert VlmConfig(backend=backend).backend == backend


# --- on_tile_embeds: resolve, or park ----------------------------------------

def test_confident_geometry_records_and_resolves():
    poser, done, arb = make_poser()
    out = feed(poser, win=0)
    # the driver re-routes this; geometry did not move the pose, so the saved
    # renders still show it and no redraw is forced (classify_stls.py:1146)
    assert out == Resolved(F, 7, pose_changed=False)
    (_, _, p) = done.poses[0]
    assert p.source == "geometry" and p.up == (0.0, 0.0, 1.0)
    assert p.confidence == 0.02            # runner/best = 0.01/0.5
    assert p.margin == 1.98                # combined [2.0, 0.02, ...]
    assert p.v == pose.POSE_CACHE_VERSION
    assert done.poses == [(F, 7, p)]
    assert not poser.parked and not arb.calls and not poser._stash


def test_siglip_moves_a_baseless_geometry_pick():
    poser, done, _ = make_poser()
    # siglip MOVED the answer, so pose_changed rides out true: any saved
    # render predates it and route must force the redraw
    assert feed(poser, win=2, geo=GEO_BASELESS) == \
        Resolved(F, 7, pose_changed=True)
    p = done.poses[-1][2]
    assert p.source == "siglip"
    assert p.up == tuple(pose.UP_CANDIDATES[2])
    assert p.confidence == 0.9             # geometry's ratio, kept as today


def test_run_mode_never_changes_the_exit():
    """There is one exit and the driver owns what follows it: no skip_embed /
    save_renders here to make the Poser branch on run mode."""
    assert not hasattr(VlmConfig(), "skip_embed")
    assert not hasattr(VlmConfig(), "save_renders")
    poser, _, _ = make_poser()
    assert feed(poser) == Resolved(F, 7, pose_changed=False)


def test_no_backend_never_parks_however_low_the_margin():
    poser, done, arb = make_poser(backend=None, margin_threshold=5.0)
    assert feed(poser) == Resolved(F, 7, pose_changed=False)
    assert not arb.calls and not poser.parked
    # ...and the gate firing with no call to make is recorded, not silent (C3):
    # an arbiterless run used to leave the key absent for everything it
    # resolved, foreclosing arbitration for the whole run's output
    assert done.poses[-1][2].arbitrated is False
    assert poser.gate_fired_no_call == 1


def test_an_ungated_resolution_makes_no_claim():
    """`None` -> the key stays absent: nothing asked for an escalation here, so
    there is nothing to say. Only the gate firing writes `false`."""
    poser, done, _ = make_poser(backend="gemini", margin_threshold=0.45)
    feed(poser)                                  # margin 1.98, well clear
    assert done.poses[-1][2].arbitrated is None
    assert poser.gate_fired_no_call == 0


def test_can_arbitrate_follows_the_backend():
    assert not make_poser(backend=None)[0].can_arbitrate()
    assert make_poser(backend="gemini")[0].can_arbitrate()


# --- park and resume ---------------------------------------------------------

def test_low_margin_parks_after_recording_the_ensemble_pose():
    poser, done, arb = make_poser(**ESCALATE)
    assert feed(poser) is None
    assert set(poser.parked) == {7}
    pf = poser.parked[7]
    assert pf.file == F and pf.future is arb.futures[0]
    assert pf.resolved == ((0.0, 0.0, 1.0), 0.02, "geometry", 1.98)
    # I15's floor: the ensemble pose is recorded BEFORE the park — and as
    # `false` since C3, so every completion path *overwrites* a record that is
    # already correct, rather than each having to remember to write one
    assert len(done.poses) == 1 and done.poses[0][2].source == "geometry"
    assert done.poses[0][2].arbitrated is False


def test_vlm_call_sees_the_grids_first_column_as_pil():
    seen = []
    poser, _, arb = make_poser(backend="gemini", margin_threshold=5.0,
                               ask=lambda tiles: seen.append(tiles) or 4)
    feed(poser)
    assert arb.calls[0]() == 4             # the stub stands in for ask_vlm_up
    (tiles,) = seen
    assert len(tiles) == 6                 # six candidates, azimuth 0 only
    assert [np.asarray(im)[0, 0, 0] for im in tiles] == [0, 10, 20, 30, 40, 50]


def test_poll_leaves_unresolved_futures_parked():
    poser, _, _ = make_poser(**ESCALATE)
    feed(poser)
    assert poser.poll() == []
    assert 7 in poser.parked


def test_poll_folds_a_vlm_override():
    poser, done, arb = make_poser(**ESCALATE)
    feed(poser)
    arb.futures[0].set_result(3)           # differs from the ensemble's up
    # the arbiter moved it: pose_changed true, though the ensemble that
    # parked this file had settled on geometry
    assert poser.poll() == [Resolved(F, 7, pose_changed=True)]
    assert done.poses[-1][2].source == "vlm"
    assert done.poses[-1][2].up == tuple(pose.UP_CANDIDATES[3])
    assert not poser.parked
    assert [p.source for _, _, p in done.poses] == ["geometry", "vlm"]


def test_poll_confirmation_keeps_the_ensemble_label():
    """A paid call that agrees moves nothing: `source` records which tier
    MOVED the answer (P2.3-A), so a confirmation stays 'geometry'."""
    poser, done, arb = make_poser(**ESCALATE)
    feed(poser)
    arb.futures[0].set_result(0)           # same up the ensemble picked
    # nothing moved, so nothing is stale: pose_changed follows the recorded
    # source, not the fact that an arbiter was paid
    assert poser.poll() == [Resolved(F, 7, pose_changed=False)]
    assert done.poses[-1][2].source == "geometry"


def test_arbitrated_separates_a_confirmation_from_a_refusal():
    """`source` records which tier MOVED the answer, so a call that ran and
    agreed is written exactly like one that never happened — the two were
    indistinguishable on disk, and 1243 entries in embed-cache2 are that
    ambiguity (2026-08-19). `arbitrated` is the fact that the call *ran*.

    Four populations, and the pair (source, arbitrated) names each:
    moved = ('vlm', True), confirmed = (ensemble, True), not-yet-asked =
    (ensemble, False), judged = (ensemble, 'rejected')."""
    def fold(result=None, exc=None):
        poser, done, arb = make_poser(**ESCALATE)
        feed(poser)
        if exc is not None:
            arb.futures[0].set_exception(exc)
        else:
            arb.futures[0].set_result(result)
        poser.poll()
        p = done.poses[-1][2]
        return p.source, p.arbitrated

    assert fold(result=3) == ("vlm", True)          # moved
    assert fold(result=0) == ("geometry", True)     # confirmed — the new fact
    assert fold(exc=pose.RateLimited("HTTP 429")) == ("geometry", False)  # refused
    # any transient failure is the same fact as a rate limit here: the API
    # never judged the request, so the answer is still worth asking for. The
    # split used to be RateLimited-vs-everything, which recorded a mid-run
    # network blip as permanent (review, 2026-08-20)
    assert fold(exc=pose.VLMUnavailable("network failure: timed out")) == \
        ("geometry", False)
    # a request the API rejects on its merits cannot succeed on a retry, so it
    # is settled permanently — and only its own type buys that now
    assert fold(exc=pose.VLMRejected("HTTP 400")) == ("geometry", "rejected")
    # the flipped default (C1, docs/tri-state-pass-2.md, 2026-08-21): an
    # unknown failure type retries loudly instead of pinning silently. Three
    # passes each found a transient failure sitting on the permanent side,
    # because permanent was the open-ended fallthrough
    assert fold(exc=RuntimeError("HTTP 400")) == ("geometry", False)
    assert fold(exc=TypeError("NoneType is not subscriptable")) == \
        ("geometry", False)

    # and a model that never escalated leaves it absent, which is what keeps
    # "not asked yet" separable from "no claim" and from every legacy entry
    poser, done, _ = make_poser()               # no escalation configured
    feed(poser)
    assert done.poses[-1][2].arbitrated is None


def test_the_real_call_path_maps_each_failure_to_its_own_record(monkeypatch):
    """End to end through `pose.ask_vlm_up`, not by setting an exception on the
    Future by hand.

    That distinction is the finding: `ask_vlm_up` swallowed every non-rate-limit
    exception and returned None, so `_fold` took its *success* branch and wrote
    `arbitrated=False` for a request the API rejects on its merits — which then
    re-escalated forever, the exact thing the change was meant to stop. A test
    that drove the Future directly reached the right branch and showed the fix
    working while production never could (review, 2026-08-19)."""
    from PIL import Image as PILImage

    def outcome(exc=None, ret=None):
        def fake(*a, **k):
            if exc is not None:
                raise exc
            return ret
        monkeypatch.setattr(pose, "_ask_gemini", fake)
        monkeypatch.setattr(pose, "make_contact_sheet",
                            lambda tiles: PILImage.new("RGB", (8, 8)))
        # cfg.ask=None, so `_vlm_call` builds the real ask_vlm_up closure
        poser, done, arb = make_poser(backend="gemini", margin_threshold=5.0)
        feed(poser)
        # run the call the way a ThreadPoolExecutor does: result or exception
        try:
            arb.futures[0].set_result(arb.calls[0]())
        except Exception as e:                      # noqa: BLE001
            arb.futures[0].set_exception(e)
        poser.poll()
        return done.poses[-1][2].arbitrated

    assert outcome(ret=2) is True                          # answered
    assert outcome(exc=pose.RateLimited("HTTP 429")) is False   # retry later
    assert outcome(exc=pose.VLMUnavailable("HTTP 502")) is False  # retry later
    assert outcome(exc=pose.VLMRejected("HTTP 400")) == "rejected"  # judged
    # an unknown type is retryable now, not permanent (C1)
    assert outcome(exc=RuntimeError("HTTP 418")) is False
    # an unparseable answer stays retryable — a judgement, not a leftover
    assert outcome(ret=None) is False


def test_arbitrated_round_trips_as_four_states_on_disk(tmp_path):
    """true / "rejected" / false / absent, asserted **on disk** and through
    the production writers (B1, docs/tri-state-pass-2.md, 2026-08-21).

    The Pose-level assertion alone passes with the bug this pins: `to_cache`
    coerced `bool(self.arbitrated)`, and `bool("rejected") is True` — the
    schema collapsed to three states in the file while every in-memory test
    stayed green."""
    def mk(**kw):
        return pose.Pose(up=(0.0, 0.0, 1.0), confidence=0.5, source="geometry",
                         v=pose.POSE_CACHE_VERSION, margin=0.2, **kw)

    assert mk(arbitrated=True).to_cache()["arbitrated"] is True
    assert mk(arbitrated=False).to_cache()["arbitrated"] is False
    assert mk(arbitrated="rejected").to_cache()["arbitrated"] == "rejected"
    assert "arbitrated" not in mk().to_cache()          # no claim

    cache = {f"model-{i}|0|0": mk(**kw).to_cache() for i, kw in enumerate(
        [{"arbitrated": True}, {"arbitrated": False},
         {"arbitrated": "rejected"}, {}])}
    pose.save_pose_cache(tmp_path, cache)
    loaded = pose.load_pose_cache(tmp_path)
    assert [e.get("arbitrated") for e in loaded.values()] == \
        [True, False, "rejected", None]
    assert [pose.Pose.from_cache(e).arbitrated for e in loaded.values()] == \
        [True, False, "rejected", None]

    # a legacy entry has no such key. It reads as `false` now — absence is no
    # claim, and the gate check is what keeps that affordable — but it must
    # still round-trip as absent rather than being stamped with a verdict
    legacy = {"up": [0, 0, 1], "confidence": 0.5, "source": "siglip",
              "margin": 0.2, "v": pose.POSE_CACHE_VERSION}
    assert pose.Pose.from_cache(legacy).arbitrated is None


def test_poll_failed_call_keeps_the_ensemble_pose(capsys):
    poser, done, arb = make_poser(**ESCALATE)
    feed(poser)
    arb.futures[0].set_exception(RuntimeError("HTTP 500"))
    # resumed, not failed; the park-time pose stands and its source rules
    assert poser.poll() == [Resolved(F, 7, pose_changed=False)]
    assert done.poses[-1][2].source == "geometry"
    assert "arbiter failed" in capsys.readouterr().out
    assert not poser.parked


def test_poll_resumes_a_cancelled_future_on_the_park_time_pose():
    """arbiter.shutdown cancels queued calls; if the driver polls after that,
    the file still resolves on the pose recorded at park time — nothing was
    billed and the answer never arrived.

    It *is* re-recorded, with `arbitrated=False`: a cancellation is "asked and
    not answered yet", so a later run must ask again. Leaving the park-time row
    untouched made it a permanent hit indistinguishable from "never asked", and
    every Ctrl-C cancels a whole queue of them (review, 2026-08-19)."""
    poser, done, arb = make_poser(**ESCALATE)
    feed(poser)
    assert arb.futures[0].cancel()
    assert poser.poll() == [Resolved(F, 7, pose_changed=False)]
    assert len(done.poses) == 2                     # re-recorded, not skipped
    assert done.poses[-1][2].source == "geometry"   # the park-time answer stands
    assert done.poses[-1][2].arbitrated is False    # ...and will be asked again
    # under ESCALATE's threshold, by a run that can ask: both conditions C4
    # added, on the fixture's own margin (1.98)
    assert not pose.pose_is_sufficient(done.poses[-1][2].to_cache(),
                                       True, ESCALATE["margin_threshold"])
    assert not poser.parked


def test_a_cancelled_fold_reads_pose_changed_off_the_park_time_source():
    """The branch a hardcoded False would silently break: the ensemble parked
    this file on a SigLIP pick, so even with the arbiter answer thrown away
    the recorded source is 'siglip' and the stale renders still need redrawing.
    pose_changed follows the pose that was recorded, never the exit taken."""
    poser, done, arb = make_poser(**ESCALATE)
    assert feed(poser, win=2, geo=GEO_BASELESS) is None    # parked on siglip
    assert done.poses[0][2].source == "siglip"
    assert arb.futures[0].cancel()
    assert poser.poll() == [Resolved(F, 7, pose_changed=True)]
    # re-recorded as un-arbitrated (2026-08-19), and the park-time source is
    # what rides out — the cancelled arm must not hardcode either of them
    assert len(done.poses) == 2
    assert done.poses[-1][2].source == "siglip"
    assert done.poses[-1][2].arbitrated is False


def test_poll_fold_error_becomes_failure_for_that_file():
    """J3: poll is its own error boundary — a raise inside the fold cannot
    be attributed by the driver, so it comes back as a Failure message."""
    done = RecordingDone()

    def record_pose(file, index, p):
        if done.poses:
            raise RuntimeError("disk gone")
        done.poses.append((file, index, p))

    arb = FakeArbiter()
    poser = Poser(UP_T, DOWN_T, arb, record_pose, VlmConfig(**ESCALATE))
    feed(poser)                            # park-time record succeeds
    arb.futures[0].set_result(3)
    (out,) = poser.poll()
    assert out == Failure(F, 7, "disk gone")
    assert not poser.parked                # retired via Failure, not stuck


# --- drop: the driver's Failure arm (C-R1-5) ---------------------------------

def test_drop_forgets_a_parked_file_and_cancels_its_call():
    poser, done, arb = make_poser(**ESCALATE)
    feed(poser)
    poser.drop(7)
    assert not poser.parked and not poser._stash
    assert arb.futures[0].cancelled()       # queued work, unbought
    assert poser.poll() == []               # nothing left to resume


def test_drop_forgets_a_file_stashed_but_not_yet_embedded():
    """The window the driver's Failure arm actually covers: on_tiles ran, the
    child died before TileEmbeds came back, and the stash would otherwise be
    held for a file nobody will finish."""
    poser, _, _ = make_poser()
    poser.on_tiles(PoseTiles(file=F, index=7, geo_scores=GEO_CONFIDENT,
                             tiles=grid_tiles()))
    assert 7 in poser._stash
    poser.drop(7)
    assert not poser._stash


def test_drop_is_a_no_op_on_an_unknown_or_dropped_index():
    """Called unconditionally from the driver, like Release child-side (K1)."""
    poser, _, arb = make_poser(**ESCALATE)
    feed(poser)
    poser.drop(999)                         # never seen
    assert set(poser.parked) == {7}
    poser.drop(7)
    poser.drop(7)                           # double drop: still no exception
    assert not poser.parked


# --- the abort pair: fold_done / settle --------------------------------------

def test_fold_done_folds_only_already_resolved_futures():
    poser, done, arb = make_poser(**ESCALATE)
    feed(poser, index=7)
    feed(poser, index=8, file=Path("models/other.stl"))
    arb.futures[0].set_result(3)           # 7 answered; 8 still in flight
    assert poser.fold_done() == 1          # no wait, no exposure
    assert set(poser.parked) == {8}        # the wait's size, left for settle
    assert [i for _, i, _ in done.poses] == [7, 8, 7]   # two parks + one fold


def test_settle_timeout_abandons_to_the_ensemble_pose():
    """Abandonment keeps the park-time pose — and re-records it
    `arbitrated=False`, because "asked, answer never arrived" is exactly the
    tri-state's retry case. Without the record the abandoned call was on disk
    indistinguishable from "never asked", so a Ctrl-C landing on in-flight
    calls silently pinned each one to its ensemble answer forever (review,
    2026-08-20)."""
    poser, done, arb = make_poser(**ESCALATE)
    feed(poser)                            # future never answers
    assert poser.settle(0.05) == 1         # the abort's closing count
    assert not poser.parked
    assert len(done.poses) == 2            # park-time floor + the retry record
    assert done.poses[-1][2].source == "geometry"   # the park-time answer stands
    assert done.poses[-1][2].arbitrated is False    # ...and will be asked again
    assert not pose.pose_is_sufficient(done.poses[-1][2].to_cache(),
                                       True, ESCALATE["margin_threshold"])


def test_settle_waits_out_and_folds_an_in_flight_answer():
    poser, done, arb = make_poser(**ESCALATE)
    feed(poser)
    threading.Timer(0.05, arb.futures[0].set_result, args=(3,)).start()
    assert poser.settle(5.0) == 0
    assert not poser.parked
    assert done.poses[-1][2].source == "vlm"    # the paid answer landed


def test_settle_skips_cancelled_futures():
    """shutdown's cancel_futures hits the queued-never-started calls; those
    were never billed, so settle neither folds nor counts them."""
    poser, done, arb = make_poser(**ESCALATE)
    feed(poser, index=7)                   # will be cancelled
    feed(poser, index=8, file=Path("models/other.stl"))   # never answers
    arb.futures[0].cancel()
    assert poser.settle(0.05) == 1         # only the unanswered one abandons
    assert not poser.parked
    # both are re-recorded as un-arbitrated — 7 by the cancelled fold (which
    # records and still returns None, what settle counts on), 8 by the
    # abandonment record (review, 2026-08-20); only 8 is *counted*
    assert [i for _, i, _ in done.poses] == [7, 8, 7, 8]
    assert all(p.arbitrated is False for _, i, p in done.poses[2:])


def test_fold_done_drops_cancelled_futures_uncounted():
    poser, done, arb = make_poser(**ESCALATE)
    feed(poser)
    arb.futures[0].cancel()
    assert poser.fold_done() == 0          # uncounted: the fold returns None
    assert not poser.parked                # dropped, keeps the park-time pose
    assert len(done.poses) == 2            # but re-recorded as un-arbitrated
    assert done.poses[-1][2].arbitrated is False


def test_settle_folds_a_call_that_failed_before_the_wait(capsys):
    """The future is already done, so settle never waits: it folds straight
    through, the failed call keeps the ensemble's pose, and nothing is
    abandoned — abandonment counts unanswered calls, not unhappy ones."""
    poser, done, arb = make_poser(**ESCALATE)
    feed(poser)
    arb.futures[0].set_exception(RuntimeError("HTTP 500"))
    assert poser.settle(5.0) == 0
    assert not poser.parked
    assert [p.source for _, _, p in done.poses] == ["geometry", "geometry"]
    assert "arbiter failed" in capsys.readouterr().out


def test_settle_distinguishes_a_calls_own_timeout_from_the_waits(capsys):
    """The branch that needs the `future.done()` re-check: since 3.11
    concurrent.futures.TimeoutError IS builtins.TimeoutError, so a VLM call
    that times out over HTTP raises exactly what an expired `result(timeout=)`
    raises. The call answered — badly — so it folds to the ensemble pose and
    must NOT be counted as abandoned."""
    poser, done, arb = make_poser(**ESCALATE)
    feed(poser)
    # in flight when settle starts, so the wait is real; the exception then
    # comes out of `result(timeout=...)` looking exactly like an expiry
    threading.Timer(0.05, arb.futures[0].set_exception,
                    args=(TimeoutError("read timed out"),)).start()
    assert poser.settle(5.0) == 0          # not abandonment: the call spoke
    assert not poser.parked
    assert [p.source for _, _, p in done.poses] == ["geometry", "geometry"]
    assert "arbiter failed" in capsys.readouterr().out


# --- the run-level circuit breaker (C5) --------------------------------------

def fold_outcome(poser, arb, outcome, index):
    """One file through park → fold. `outcome` is an exception, "cancel", or
    the arbiter's answer."""
    feed(poser, index=index, file=Path(f"models/{index}.stl"))
    future = arb.futures[-1]
    if isinstance(outcome, BaseException):
        future.set_exception(outcome)
    elif outcome == "cancel":
        future.cancel()
    else:
        future.set_result(outcome)
    poser.poll()


def unavailable(n, poser, arb, clock, gap, first=0):
    """`n` consecutive transient failures, `gap` seconds apart."""
    for i in range(n):
        if i:
            clock.advance(gap)
        fold_outcome(poser, arb, pose.VLMUnavailable(f"HTTP 502 #{i}"),
                     first + i)


def test_the_breaker_trips_on_five_consecutive_failures_spanning_a_minute(capsys):
    """N = 5 **and** ≥ 60 s, both (review 2, confirmed by Masa 2026-08-21).
    Vertex quota is a rate refilled per minute, effective concurrency is 3 and
    VLM_BACKOFF makes five consecutive rate-limited folds reachable in well
    under a minute — a sub-minute storm must not zero out an overnight run's
    arbitration."""
    clock = FakeClock()
    poser, _, arb = make_poser(clock=clock, **ESCALATE)
    unavailable(4, poser, arb, clock, gap=20.0)
    assert poser.can_arbitrate()           # four is not five
    clock.advance(20.0)                    # 80 s spanned by the fifth
    fold_outcome(poser, arb, pose.RateLimited("HTTP 429"), 4)
    assert not poser.can_arbitrate()
    out = capsys.readouterr()
    assert "arbiter disabled" in out.err    # one loud line, on stderr


def test_a_sub_minute_storm_does_not_trip_the_breaker():
    clock = FakeClock()
    poser, _, arb = make_poser(clock=clock, **ESCALATE)
    unavailable(8, poser, arb, clock, gap=5.0)     # 35 s, eight failures
    assert poser.can_arbitrate()
    clock.advance(30.0)                            # the ninth spans 65 s
    fold_outcome(poser, arb, pose.VLMUnavailable("HTTP 502"), 8)
    assert not poser.can_arbitrate()


def test_every_fold_the_api_answered_resets_the_counter():
    """The API spoke, so the environment is healthy — whatever it said. Four
    failures, one answer, four more: consecutive is what counts."""
    for spoke in (3, None, pose.VLMRejected("HTTP 400")):
        clock = FakeClock()
        poser, _, arb = make_poser(clock=clock, **ESCALATE)
        unavailable(4, poser, arb, clock, gap=30.0)
        clock.advance(30.0)
        fold_outcome(poser, arb, spoke, 4)         # answer / unparseable / judged
        clock.advance(30.0)
        unavailable(4, poser, arb, clock, gap=30.0, first=5)
        assert poser.can_arbitrate(), spoke


def test_cancellation_neither_counts_nor_resets():
    """The abort path folds and settles a whole queue of cancellations: it
    must not trip the breaker on the way out, nor clear a streak that was
    about to."""
    clock = FakeClock()
    poser, _, arb = make_poser(clock=clock, **ESCALATE)
    for i in range(6):                             # six cancels, spread wide
        clock.advance(30.0)
        fold_outcome(poser, arb, "cancel", i)
    assert poser.can_arbitrate()                   # counted nothing

    unavailable(4, poser, arb, clock, gap=30.0, first=10)
    clock.advance(30.0)
    fold_outcome(poser, arb, "cancel", 20)         # cleared nothing
    clock.advance(30.0)
    fold_outcome(poser, arb, pose.VLMUnavailable("HTTP 502"), 21)
    assert not poser.can_arbitrate()               # the fifth, not a first


def test_a_tripped_breaker_marks_instead_of_parking():
    """After the trip: nothing new parks, the gate's firing is still recorded
    `false`, and the count the run reports rises. `can_arbitrate()` is what
    the driver hands `route`, which is what stops the marked backlog
    re-rendering for the rest of the run."""
    clock = FakeClock()
    poser, done, arb = make_poser(clock=clock, **ESCALATE)
    unavailable(5, poser, arb, clock, gap=20.0)
    assert not poser.can_arbitrate()
    assert poser.gate_fired_no_call == 0           # every one of those asked

    parked_before = len(arb.calls)
    assert feed(poser, index=99) == Resolved(F, 99, pose_changed=False)
    assert len(arb.calls) == parked_before and not poser.parked
    assert done.poses[-1][2].arbitrated is False
    assert poser.gate_fired_no_call == 1


def test_an_already_parked_call_still_folds_after_the_trip():
    """The trip stops *submissions*, not folds: the ~N calls already in flight
    are paid for and their answers must land. A healthy one among them does
    not re-enable arbitration — the breaker never untrips within a run."""
    clock = FakeClock()
    poser, done, arb = make_poser(clock=clock, **ESCALATE)
    feed(poser, index=90, file=Path("models/90.stl"))      # in flight throughout
    unavailable(5, poser, arb, clock, gap=20.0)
    assert not poser.can_arbitrate() and set(poser.parked) == {90}

    arb.futures[0].set_result(3)
    assert poser.poll() == [Resolved(Path("models/90.stl"), 90,
                                     pose_changed=True)]
    assert done.poses[-1][2].arbitrated is True           # the paid answer
    assert not poser.can_arbitrate()
