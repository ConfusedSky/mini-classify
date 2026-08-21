import subprocess
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
from PIL import Image

from src import pose

REPO = Path(__file__).resolve().parent.parent


def prepared(mesh):
    mesh.compute_vertex_normals()
    return mesh


def test_cone_up_is_decisive():
    # a cone has exactly one flat face (its base) -> unambiguous print base
    cone = prepared(o3d.geometry.TriangleMesh.create_cone(radius=0.5, height=2.0))
    up, ratio, best = pose.detect_up_axis(cone)
    assert np.allclose(up, [0, 0, 1])
    assert best > pose.ABS_SCORE_FLOOR
    assert ratio < 0.6


def test_rotated_cone_finds_new_up():
    cone = prepared(o3d.geometry.TriangleMesh.create_cone(radius=0.5, height=2.0))
    # Rx(-90deg) maps the model's up (+Z) onto +Y
    cone.rotate(o3d.geometry.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0)),
                center=(0, 0, 0))
    up, ratio, best = pose.detect_up_axis(cone)
    assert np.allclose(up, [0, 1, 0])


def test_cylinder_is_ambiguous():
    # flat cap on both ends: +Z and -Z score the same -> ratio ~ 1
    cyl = prepared(o3d.geometry.TriangleMesh.create_cylinder(radius=0.5, height=2.0))
    up, ratio, best = pose.detect_up_axis(cyl)
    assert abs(float(up @ np.array([0.0, 0.0, 1.0]))) > 0.99  # either cap wins
    assert ratio > 0.6


def test_rotation_to_z_up_matches_open3d_bit_for_bit():
    """The table is byte-identical to the Open3D construction it replaced.

    `Renderer.views` rotates by this before shooting, so it decides the pixels
    — and therefore the cached embeddings — of every non-`+Z` model (1902 of
    embed-cache2's 2945), while the embedding key records only the up *vector*.
    A value that differed in the last bit would re-pose those models under
    unchanged keys with nothing failing.

    So this asserts against Open3D itself rather than against transcribed
    constants: the point is the *equality*, and a test that only restated the
    table would drift with it. Note `np.array_equal` is the right comparison
    and `allclose` is not — the whole risk here lives in the last bits."""
    z = np.array([0.0, 0.0, 1.0])
    for up in pose.UP_CANDIDATES:
        if np.allclose(up, z):
            want = np.eye(3)
        elif np.allclose(up, -z):
            want = o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        else:
            axis = np.cross(up, z)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.clip(up @ z, -1, 1))
            want = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        got = pose.rotation_to_z_up(up)
        # `.tobytes()`, not `array_equal` — the latter reports -0.0 == 0.0, and
        # four of the six matrices carry signed zeros the table transcribes
        # deliberately. array_equal let a sign-flipped table pass (found in
        # review, 2026-08-19)
        assert got.tobytes() == np.ascontiguousarray(want, np.float64).tobytes(), tuple(up)
        # and the properties that make it a legitimate rotation, so a rewrite
        # that is merely *different* reads differently from one that is wrong
        assert np.allclose(got @ np.asarray(up, float), z, atol=1e-12), tuple(up)
        assert np.allclose(got @ got.T, np.eye(3), atol=1e-12), tuple(up)
        assert abs(np.linalg.det(got) - 1.0) < 1e-12, tuple(up)


def test_rotation_to_z_up_falls_back_for_a_non_axis_vector():
    """Total, not a lookup that raises. Nothing in the pipeline reaches this —
    poses are always one of the six — so agreement with Open3D at ~1e-15 is
    the bar, not bit-identity."""
    up = np.array([0.3, -0.5, 0.81])
    up = up / np.linalg.norm(up)
    axis = np.cross(up, [0.0, 0.0, 1.0])
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(up @ np.array([0.0, 0.0, 1.0]), -1, 1))
    want = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
    got = pose.rotation_to_z_up(up)
    assert np.allclose(got, want, atol=1e-14)
    assert np.allclose(got @ up, [0.0, 0.0, 1.0], atol=1e-14)


def test_rotation_to_z_up_handles_collinear_and_non_unit_input():
    """A vector along Z but not unit must not reach Rodrigues.

    Its cross product with Z is zero, so the axis is 0/0 and the matrix comes
    out all-NaN. Open3D's version hid exactly this: handed a nan axis-angle it
    returned the *identity*, so `[0,0,-2]` rendered upside down and said
    nothing. Normalising first sends these to the table, where `-2*z` is
    `Rx(pi)` and not the identity — which is the answer the old code got
    wrong, not merely differently."""
    z_up = pose.rotation_to_z_up(np.array([0.0, 0.0, 1.0]))
    z_down = pose.rotation_to_z_up(np.array([0.0, 0.0, -1.0]))
    for scale in (2.0, 0.5, 1.0000001, 0.99999):
        assert np.array_equal(pose.rotation_to_z_up(np.array([0.0, 0.0, scale])), z_up), scale
        assert np.array_equal(pose.rotation_to_z_up(np.array([0.0, 0.0, -scale])), z_down), scale
    # and an off-axis vector still normalises to a proper rotation
    R = pose.rotation_to_z_up(np.array([0.0, 3.0, 3.0]))
    assert np.isfinite(R).all()
    assert np.allclose(R @ (np.array([0.0, 3.0, 3.0]) / np.linalg.norm([0.0, 3.0, 3.0])),
                       [0.0, 0.0, 1.0], atol=1e-12)


@pytest.mark.parametrize("bad", [[0.0, 0.0, 0.0], [np.nan, 0.0, 1.0], [np.inf, 0.0, 0.0]])
def test_rotation_to_z_up_rejects_a_degenerate_vector(bad):
    """No rotation takes nothing to +Z. Raising beats returning NaN, which
    `azimuth_zero` would otherwise publish to a consumer as a pose."""
    with pytest.raises(ValueError, match="finite non-zero"):
        pose.rotation_to_z_up(np.array(bad))


def test_rotation_to_z_up_does_not_hand_out_its_table():
    """A caller that mutates the result must not corrupt every later call."""
    first = pose.rotation_to_z_up(np.array([0.0, 1.0, 0.0]))
    first[0, 0] = 99.0
    assert pose.rotation_to_z_up(np.array([0.0, 1.0, 0.0]))[0, 0] == 1.0


def test_a_rate_limited_vlm_waits_before_retrying(monkeypatch):
    """429/503 mean "later", not "this request is wrong", and they return in
    milliseconds — so an immediate retry is a second refusal and the freed
    worker starts a third. A collection-scale run hit Vertex quota this way
    (2026-08-19)."""
    calls, slept = [], []

    def refuse(*a, **k):
        calls.append(1)
        raise pose.RateLimited("HTTP 429: Resource exhausted")

    monkeypatch.setattr(pose, "_ask_gemini", refuse)
    out = pose.ask_vlm_up([Image.new("RGB", (8, 8))] * 6, "gemini", "/tmp",
                          vlm_model="m", sleep=slept.append)
    assert out is None                      # the geometry answer stands
    assert len(calls) == 2                  # tried twice, as before
    assert slept == [pose.VLM_BACKOFF[0]]   # and waited between them


def test_only_the_last_attempts_failure_decides_the_record(monkeypatch):
    """The rule is the *last* attempt, and `last_error` used to survive one
    that did not raise: an attempt returning an unparseable answer left the
    previous exception standing, so "400 then unparseable" raised the stale
    400 and cached the model once — where the last attempt raised nothing and
    an unparseable answer is retryable (review, 2026-08-19)."""
    def sequence(*outcomes):
        it = iter(outcomes)

        def fake(*a, **k):
            v = next(it)
            if isinstance(v, Exception):
                raise v
            return v

        monkeypatch.setattr(pose, "_ask_gemini", fake)
        return pose.ask_vlm_up([Image.new("RGB", (8, 8))] * 6, "gemini", "/tmp",
                               vlm_model="m", sleep=lambda s: None,
                               raise_failures=True)

    # last attempt returned (unparseably) — no exception may escape
    assert sequence(RuntimeError("HTTP 400"), None) is None
    assert sequence(pose.RateLimited("HTTP 429"), None) is None
    # last attempt raised — that one decides, not the first
    with pytest.raises(RuntimeError):
        sequence(None, RuntimeError("HTTP 400"))
    with pytest.raises(pose.RateLimited):
        sequence(RuntimeError("HTTP 400"), pose.RateLimited("HTTP 429"))
    # and an answer on the retry beats anything before it
    assert sequence(RuntimeError("HTTP 400"), 2) == 2


def test_an_ordinary_vlm_error_still_retries_at_once(monkeypatch):
    """Only a rate limit waits. A malformed request should fail fast and let
    the run carry on with the ensemble's answer."""
    slept = []

    def broken(*a, **k):
        raise RuntimeError("HTTP 400: bad request")

    monkeypatch.setattr(pose, "_ask_gemini", broken)
    assert pose.ask_vlm_up([Image.new("RGB", (8, 8))] * 6, "gemini", "/tmp",
                           vlm_model="m", sleep=slept.append) is None
    assert slept == []


def test_a_transient_failure_retries_at_once_and_raises_its_own_type(monkeypatch):
    """`VLMUnavailable` sits between the two arms above: no backoff (it is not
    a quota signal), but the type must survive to the caller so `_fold`
    records it retryable rather than permanent (review, 2026-08-20)."""
    calls, slept = [], []

    def unreachable(*a, **k):
        calls.append(1)
        raise pose.VLMUnavailable("network failure: connection reset")

    monkeypatch.setattr(pose, "_ask_gemini", unreachable)
    with pytest.raises(pose.VLMUnavailable):
        pose.ask_vlm_up([Image.new("RGB", (8, 8))] * 6, "gemini", "/tmp",
                        vlm_model="m", sleep=slept.append, raise_failures=True)
    assert len(calls) == 2 and slept == []


def test_gemini_maps_each_transport_failure_to_the_retry_split(monkeypatch):
    """The record `_fold` writes hangs off the exception type alone, so the
    mapping IS the retry policy: 429/503 back off, any other 5xx and every
    network-layer failure are transient, and only a request judged on its
    merits (4xx) is permanent. Before this, a socket timeout or a 502 landed
    on the permanent side and the model was never re-asked (review,
    2026-08-20)."""
    import io as io_mod
    import socket
    import urllib.error
    import urllib.request

    monkeypatch.setattr(pose, "gcloud_token", lambda: "tok")

    def with_urlopen(exc):
        def fake_urlopen(req, timeout=None):
            raise exc
        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        try:
            pose._ask_gemini(b"png", 6, "m", project="p")
        except Exception as e:              # noqa: BLE001 - the type is the assertion
            return type(e)
        return None

    def http_error(code):
        return urllib.error.HTTPError("http://x", code, "boom", None,
                                      io_mod.BytesIO(b"detail"))

    assert with_urlopen(http_error(429)) is pose.RateLimited
    assert with_urlopen(http_error(503)) is pose.RateLimited
    assert with_urlopen(http_error(500)) is pose.VLMUnavailable
    assert with_urlopen(http_error(502)) is pose.VLMUnavailable
    assert with_urlopen(http_error(400)) is RuntimeError      # judged: permanent
    assert with_urlopen(urllib.error.URLError("dns down")) is pose.VLMUnavailable
    assert with_urlopen(socket.timeout("timed out")) is pose.VLMUnavailable


def test_claude_cli_failures_are_transient(monkeypatch):
    """The claude backend could never say `RateLimited` at all: a timed-out
    CLI raised TimeoutExpired into the permanent branch, and a non-zero exit
    flattened to None — indistinguishable from an unparseable answer
    (review, 2026-08-20)."""
    import subprocess as sp

    def run_fails(*a, **k):
        raise sp.TimeoutExpired(cmd="claude", timeout=180)

    monkeypatch.setattr(pose.subprocess, "run", run_fails)
    with pytest.raises(pose.VLMUnavailable):
        pose._ask_claude("/nowhere/sheet.png", 6)

    class Exited:
        returncode, stdout, stderr = 1, "", "quota exceeded"

    monkeypatch.setattr(pose.subprocess, "run", lambda *a, **k: Exited())
    with pytest.raises(pose.VLMUnavailable, match="quota exceeded"):
        pose._ask_claude("/nowhere/sheet.png", 6)


def test_claude_sheet_files_are_unique_per_call_and_cleaned_up(monkeypatch, tmp_path):
    """All arbiter workers share one scratch_dir; a fixed `pose-sheet.png` let
    worker B's save land between A's save and A's subprocess read, so A's
    model was judged on B's renders and stamped `source: vlm` (review,
    2026-08-20). Unique per call, existing while the CLI reads it, gone
    after."""
    seen = []

    def fake_ask(sheet_path, n_tiles):
        assert Path(sheet_path).exists()     # the CLI reads a real file
        seen.append(Path(sheet_path))
        return 1

    monkeypatch.setattr(pose, "_ask_claude", fake_ask)
    tiles = [Image.new("RGB", (8, 8))] * 6
    assert pose.ask_vlm_up(tiles, "claude", tmp_path) == 1
    assert pose.ask_vlm_up(tiles, "claude", tmp_path) == 1
    assert len(seen) == 2 and seen[0] != seen[1]
    assert all(p.parent == tmp_path for p in seen)
    assert not any(p.exists() for p in seen)             # unlinked either way


def test_view_angles_rings_are_nested_subsets():
    """A ring of n azimuths is a subset of a ring of m when n divides m.

    The property `eval/tile_count.py` relies on to slice a cached 24-tile grid
    instead of re-rendering per n_az, and `tests/test_renderer.py` uses as its
    oracle for `pose_tiles`' cameras. It belongs to the function, which lives
    here since 2026-08-19."""
    four = pose.view_angles(4, [20.0])
    for n in (2, 1):
        assert all(a in four for a in pose.view_angles(n, [20.0])), n


def test_view_angles_is_elevation_major():
    """views 0..n-1 are the first elevation's ring — `front_view` indices and
    every saved `view<i>.png` name are positions in this list."""
    angles = pose.view_angles(4, [20.0, -20.0])
    assert len(angles) == 8
    assert [round(np.rad2deg(e), 6) for _, e in angles] == [20.0] * 4 + [-20.0] * 4
    assert [round(np.rad2deg(a), 6) for a, _ in angles[:4]] == [0.0, 90.0, 180.0, 270.0]


def test_pose_cache_roundtrip(tmp_path):
    cache = {"some|identity": {"up": [0.0, 0.0, 1.0], "front_view": 2,
                               "confidence": 0.15, "source": "geometry",
                               "v": pose.POSE_CACHE_VERSION}}
    pose.save_pose_cache(tmp_path, cache)
    assert pose.load_pose_cache(tmp_path) == cache
    assert pose.load_pose_cache(tmp_path / "missing") == {}
    assert pose.load_pose_cache(None) == {}  # cache disabled


def test_pose_cache_drops_stale_versions(tmp_path):
    # a pose decided under an older ensemble or gate is not this version's pose
    cache = {"old": {"up": [0.0, 0.0, 1.0], "source": "vlm"},                    # no v
             "older": {"up": [0.0, 1.0, 0.0], "source": "vlm", "v": 1},
             "current": {"up": [0.0, 0.0, 1.0], "source": "vlm",
                         "v": pose.POSE_CACHE_VERSION}}
    pose.save_pose_cache(tmp_path, cache)
    assert set(pose.load_pose_cache(tmp_path)) == {"current"}


def test_combine_up_reports_the_winning_margin():
    geo = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    agree = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    idx, margin = pose.combine_up(geo, agree)
    assert idx == 0 and margin == pytest.approx(2.0)      # both vote, unanimously

    split = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])      # SigLIP says the opposite
    idx, margin = pose.combine_up(geo, split)
    assert margin == pytest.approx(0.0)                   # tied: maximally unsure
    assert pose.needs_arbiter_margin(margin)

    assert pose.combine_up_scores(geo, agree) == 0        # legacy wrapper unchanged


def test_contact_sheet_numerals_scale_with_the_tile():
    # the trap: a 512px sheet with PIL's ~11px bitmap face measures worse than
    # 256, because the model cannot read the numbers it is asked to answer with
    assert pose.sheet_font(512).size == 44
    assert pose.sheet_font(256).size == 22
    assert pose.sheet_font(64).size == 11        # never smaller than the bitmap


def test_contact_sheet_defaults_to_512():
    tiles = [Image.new("RGB", (400, 400), "white") for _ in range(6)]
    assert pose.make_contact_sheet(tiles).size == (3 * 512, 2 * 512)
    assert pose.make_contact_sheet(tiles, thumb=256).size == (3 * 256, 2 * 256)


def test_gemini_backend_is_dispatched_and_degrades(monkeypatch, tmp_path):
    tiles = [Image.new("RGB", (64, 64), "white") for _ in range(6)]
    seen = {}

    def fake(png, n_tiles, model, project=None):
        seen.update(model=model, project=project, n_tiles=n_tiles)
        return 3

    monkeypatch.setattr(pose, "_ask_gemini", fake)
    assert pose.ask_vlm_up(tiles, "gemini", tmp_path, "gemini-3.5-flash",
                           project="proj-x") == 3
    assert seen == {"model": "gemini-3.5-flash", "project": "proj-x", "n_tiles": 6}

    # an arbiter that fails must never fail the run — the geometry guess stands
    def boom(*a, **k):
        raise RuntimeError("HTTP 403")

    monkeypatch.setattr(pose, "_ask_gemini", boom)
    assert pose.ask_vlm_up(tiles, "gemini", tmp_path, "gemini-3.5-flash") is None


def test_geometry_vote_is_scaled_by_its_base_evidence():
    real_base = np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0])      # well over the floor
    no_base = np.array([0.0075, 0.0032, 0.0, 0.0, 0.0, 0.0])  # under it, but unequal
    assert pose.geo_weight(real_base) == pytest.approx(1.0)
    assert pose.geo_weight(no_base) < 0.2

    # the failure this fixes: geometry with no base outvoting SigLIP on ratio alone
    sig = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert pose.combine_up(no_base, sig)[0] == 2          # SigLIP decides
    assert pose.combine_up(real_base, sig)[0] in (0, 2)   # a real base still argues


def test_margin_gate_escalates_only_the_unsure():
    assert pose.needs_arbiter_margin(0.1)
    assert not pose.needs_arbiter_margin(1.3)
    # the point of the change: geometry with no base at all can still be
    # confident enough that the ensemble does not need arbitrating
    assert not pose.needs_arbiter_margin(0.9, threshold=0.45)
    assert pose.needs_arbiter_margin(0.9, threshold=1.0)


def test_geometry_only_pose_is_a_miss():
    # margin is None exactly when the ensemble did not run: one old
    # geometry-only pass must not pin the pose, so this run re-resolves the
    # entry. The `ensemble_available` parameter is retired with
    # --no-up-ensemble (2026-08-17) — the ensemble always runs, so the arm
    # that took any cached answer had no reachable caller left.
    geo = {"up": [0, 0, 1], "confidence": 0.4, "source": "geometry", "margin": None}
    assert not pose.pose_is_sufficient(geo)


def test_full_run_poses_stay_cached():
    # geometry-with-margin means the ensemble ran and agreed with geometry
    assert pose.pose_is_sufficient({"source": "geometry", "margin": 0.62})
    assert pose.pose_is_sufficient({"source": "siglip", "margin": 0.51})
    # a VLM answer outranks the ensemble whichever gate escalated it, so a
    # geometry-gated arbiter call from an old --no-up-ensemble run is not
    # re-bought
    assert pose.pose_is_sufficient({"source": "vlm", "margin": None})


def test_a_refused_arbiter_call_is_a_miss():
    """`arbitrated: false` is the ensemble's answer standing in for one that
    was asked for and never arrived — a 429, an error, a cancellation. Without
    this it was a permanent cache hit that nothing would ever re-ask
    (2026-08-19)."""
    refused = {"source": "siglip", "margin": 0.2, "arbitrated": False}
    assert not pose.pose_is_sufficient(refused)


def test_only_an_explicit_false_re_escalates():
    """The distinction that made this affordable. An *absent* key is every
    entry written before the flag and every model that never escalated —
    treating those as misses would re-escalate ~1243 models of embed-cache2 on
    the first run, which is why this waited for the flag to be tri-state."""
    legacy = {"source": "siglip", "margin": 0.2}            # no key at all
    answered = {"source": "geometry", "margin": 0.2, "arbitrated": True}
    assert pose.pose_is_sufficient(legacy)
    assert pose.pose_is_sufficient(answered)
    # and a vlm answer is sufficient however the flag reads — it moved the pose
    assert pose.pose_is_sufficient({"source": "vlm", "margin": 0.2,
                                    "arbitrated": False})


def test_no_entry_is_never_sufficient():
    assert not pose.pose_is_sufficient(None)


def test_file_identity_changes_with_mtime_and_size(tmp_path):
    f = tmp_path / "a.stl"
    f.write_text("x")
    first = pose.file_identity(f, tmp_path)
    assert "a.stl" in first
    f.write_text("xy")
    assert pose.file_identity(f, tmp_path) != first


def test_embed_cache_token_is_the_up_vector():
    # review P2.3-B: only `up` changes the pixels, so `up` IS the token —
    # source is not render identity
    assert pose.embed_cache_token(
        {"up": [0.0, 0.0, 1.0], "source": "geometry"}, "auto") == "0,0,1"
    assert pose.embed_cache_token(
        {"up": [0.0, 1.0, 0.0], "source": "vlm"}, "auto") == "0,1,0"
    # a forced axis and a geometry answer that agree render identical pixels;
    # the old elision filed them under two keys, now they share one
    assert pose.embed_cache_token(None, "z") == \
        pose.embed_cache_token({"up": [0.0, 0.0, 1.0], "source": "geometry"}, "auto")
    assert pose.embed_cache_token(None, "y") == "0,1,0"
    # no pose yet under auto: nothing is cached under any key
    assert pose.embed_cache_token(None, "auto") == "unresolved"


def test_pose_cache_renames_legacy_sources(tmp_path):
    # the P2.3-A rename maps on load — no version bump, because the poses
    # themselves are unchanged and a bump would re-resolve (and re-bill) them
    pose.save_pose_cache(tmp_path, {
        "a": {"up": [0, 0, 1], "source": "heuristic", "v": pose.POSE_CACHE_VERSION},
        "b": {"up": [0, 0, 1], "source": "ensemble", "v": pose.POSE_CACHE_VERSION},
        "c": {"up": [0, 0, 1], "source": "vlm", "v": pose.POSE_CACHE_VERSION}})
    got = pose.load_pose_cache(tmp_path)
    assert [got[k]["source"] for k in "abc"] == ["geometry", "siglip", "vlm"]


def test_pose_from_cache_never_defaults_v():
    # D10: `v` is carried through, never defaulted — a default of
    # POSE_CACHE_VERSION would stamp unversioned entries as freshly resolved
    # and defeat load_pose_cache's drop rule
    p = pose.Pose.from_cache({"up": [0, 0, 1], "source": "geometry"})
    assert p.v != pose.POSE_CACHE_VERSION
    assert p.v == 0
    assert p.margin is None                      # absent from older entries


def test_pose_from_cache_treats_bare_int_front_view_as_absent():
    # D3: pre-keying entries (real on disk: embed-cache3 is nearly all
    # `front_view: 0`) carry no record of the config that produced them
    entry = {"up": [0.0, 0.0, 1.0], "confidence": 0.17, "source": "geometry",
             "margin": 1.6489, "v": 4, "front_view": 0}
    assert pose.Pose.from_cache(entry).front_view == {}
    # a per-config dict is preserved as-is
    keyed = dict(entry, front_view={"8v-e20,-20": 5})
    assert pose.Pose.from_cache(keyed).front_view == {"8v-e20,-20": 5}


def test_pose_to_cache_round_trips_a_real_entry():
    # the embed-cache2 shape (post-source-rename), all fields present
    entry = {"up": [0.0, 0.0, 1.0], "confidence": 0.17, "source": "geometry",
             "margin": 1.6512, "v": 4, "front_view": {"8v-e20,-20": 0}}
    assert pose.Pose.from_cache(entry).to_cache() == entry
    # `front_view` is omitted when nothing has been resolved, matching
    # entries that predate front-view caching
    bare = {k: v for k, v in entry.items() if k != "front_view"}
    assert pose.Pose.from_cache(bare).to_cache() == bare
    assert "front_view" not in pose.Pose.from_cache(dict(bare, front_view=0)).to_cache()


def test_front_view_is_keyed_by_view_config():
    # an index cached at 8 views is out of range at 4, and silently wrong at
    # the same count with different elevations — so a config miss is a miss
    entry = {"front_view": {"8v-e20,-20": 6}}
    assert pose.front_view(entry, "8v-e20,-20") == 6
    assert pose.front_view(entry, "4v-e20") is None


def test_legacy_front_view_int_is_treated_as_absent():
    # pre-keying entries carry no record of the config that produced them; a
    # warm classify pass regenerates them from cached embeddings
    assert pose.front_view({"front_view": 6}, "8v-e20,-20") is None
    assert pose.front_view({}, "8v-e20,-20") is None
    assert pose.front_view(None, "8v-e20,-20") is None


def test_front_view_index_picks_frontmost():
    front = np.array([[1.0, 0.0]])
    back = np.array([[0.0, 1.0]])
    views = np.array([[0.0, 1.0],    # back-facing
                      [0.7, 0.7],
                      [1.0, 0.0],    # front-facing
                      [0.7, 0.3]])
    assert pose.front_view_index(views, front, back) == 2


def test_front_prompts_defined():
    assert pose.FRONT_PROMPTS and pose.BACK_PROMPTS


def test_parse_tile_answer():
    assert pose.parse_tile_answer('{"tile": 3}', 6) == 2
    assert pose.parse_tile_answer('The answer is {"tile": 1}.', 6) == 0
    assert pose.parse_tile_answer('{"tile": 9}', 6) is None
    assert pose.parse_tile_answer('{"tile": 0}', 6) is None
    assert pose.parse_tile_answer("no json here", 6) is None
    assert pose.parse_tile_answer('{"tile": "two"}', 6) is None


def test_make_contact_sheet_grid():
    tiles = [Image.new("RGB", (512, 512), "gray") for _ in range(6)]
    sheet = pose.make_contact_sheet(tiles, thumb=100, cols=3)
    assert sheet.size == (300, 200)  # 3x2 grid of 100px tiles


requires_ollama = pytest.mark.skipif(not pose.ollama_available(),
                                     reason="ollama not running")


def test_importing_pose_does_not_import_torch_or_src_peers():
    """The import-rule table's leaf rule (interfaces.md): `pose` must not
    import torch or any other src/ module — it is the leaf both sides of the
    process boundary import, so a back-import would make it a peer and a
    torch import would hand the render child VRAM/startup cost for nothing.
    `src.identity` is the table's one sanctioned dependency (the leaf below
    the leaf). Mirrors test_messages.py's I8 guard: fresh interpreter, then
    inspect sys.modules.

    **open3d counts too**, and did not until review caught the gap
    (2026-08-19): the table permits it only inside `up_axis_scores`, and both
    functions that moved here from `renderer` are pure numpy precisely so that
    stays true. Without this clause the whole premise of the move rested on a
    manual check nobody would rerun."""
    allowed = "{'src', 'src.pose', 'src.identity'}"
    code = ("import sys; from src import pose; "
            "bad = [k for k in sys.modules if k in ('torch', 'open3d') "
            "or k.startswith(('torch.', 'open3d.'))]; "
            "bad += [k for k in sys.modules if (k == 'src' "
            f"or k.startswith('src.')) and k not in {allowed}]; "
            "print(', '.join(bad)); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"forbidden imports: {r.stdout}\n{r.stderr}"


@requires_ollama
def test_ask_vlm_up_live_transport(tmp_path):
    # exercises the real request/parse path; semantic quality isn't asserted,
    # and a missing/unpulled model must degrade to None, never raise
    tiles = [Image.new("RGB", (64, 64), "gray") for _ in range(6)]
    idx = pose.ask_vlm_up(tiles, "ollama", tmp_path)
    assert idx is None or 0 <= idx < 6
