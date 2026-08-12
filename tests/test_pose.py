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


def test_margin_gate_escalates_only_the_unsure():
    assert pose.needs_arbiter_margin(0.1)
    assert not pose.needs_arbiter_margin(1.3)
    # the point of the change: geometry with no base at all can still be
    # confident enough that the ensemble does not need arbitrating
    assert not pose.needs_arbiter_margin(0.9, threshold=0.45)
    assert pose.needs_arbiter_margin(0.9, threshold=1.0)


def test_file_identity_changes_with_mtime_and_size(tmp_path):
    f = tmp_path / "a.stl"
    f.write_text("x")
    first = pose.file_identity(f)
    assert str(f.resolve()) in first
    f.write_text("xy")
    assert pose.file_identity(f) != first


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
