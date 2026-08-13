import numpy as np
import open3d as o3d
import pytest
from PIL import Image

import pose


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
    assert not pose.needs_arbiter(ratio, best)


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
    assert pose.needs_arbiter(ratio, best)


def test_pose_cache_roundtrip(tmp_path):
    cache = {"some|identity": {"up": [0.0, 0.0, 1.0], "front_view": 2,
                               "confidence": 0.15, "source": "heuristic",
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

    # an arbiter that fails must never fail the run — the heuristic guess stands
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


def test_geometry_only_pose_is_a_miss_for_an_ensemble_run():
    # margin is None exactly when the ensemble did not run: one --no-up-ensemble
    # pass must not pin the pose, so an ensemble run re-resolves the entry —
    # while a second geometry-only run still gets its cache hit
    geo = {"up": [0, 0, 1], "confidence": 0.4, "source": "heuristic", "margin": None}
    assert not pose.pose_is_sufficient(geo, ensemble_available=True)
    assert pose.pose_is_sufficient(geo, ensemble_available=False)


def test_full_run_poses_stay_cached():
    # heuristic-with-margin means the ensemble ran and agreed with geometry
    assert pose.pose_is_sufficient({"source": "heuristic", "margin": 0.62}, True)
    assert pose.pose_is_sufficient({"source": "ensemble", "margin": 0.51}, True)
    # a VLM answer outranks the ensemble whichever gate escalated it, so a
    # geometry-gated arbiter call from a --no-up-ensemble run is not re-bought
    assert pose.pose_is_sufficient({"source": "vlm", "margin": None}, True)


def test_no_entry_is_never_sufficient():
    assert not pose.pose_is_sufficient(None, ensemble_available=False)
    assert not pose.pose_is_sufficient(None, ensemble_available=True)


def test_file_identity_changes_with_mtime_and_size(tmp_path):
    f = tmp_path / "a.stl"
    f.write_text("x")
    first = pose.file_identity(f, tmp_path)
    assert "a.stl" in first
    f.write_text("xy")
    assert pose.file_identity(f, tmp_path) != first


def test_embed_cache_token_keeps_legacy_key_for_heuristic():
    heur = {"up": [0.0, 0.0, 1.0], "source": "heuristic"}
    vlm = {"up": [0.0, 1.0, 0.0], "source": "vlm"}
    assert pose.embed_cache_token(heur, "auto") == "auto"
    assert pose.embed_cache_token(None, "auto") == "auto"
    assert pose.embed_cache_token({"up": [0, 0, 1], "source": "forced"}, "z") == "z"
    assert pose.embed_cache_token(vlm, "auto") == "vlm:0,1,0"


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


@requires_ollama
def test_ask_vlm_up_live_transport(tmp_path):
    # exercises the real request/parse path; semantic quality isn't asserted,
    # and a missing/unpulled model must degrade to None, never raise
    tiles = [Image.new("RGB", (64, 64), "gray") for _ in range(6)]
    idx = pose.ask_vlm_up(tiles, "ollama", tmp_path)
    assert idx is None or 0 <= idx < 6
