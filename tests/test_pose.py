import numpy as np
import open3d as o3d

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
                               "confidence": 0.15, "source": "heuristic"}}
    pose.save_pose_cache(tmp_path, cache)
    assert pose.load_pose_cache(tmp_path) == cache
    assert pose.load_pose_cache(tmp_path / "missing") == {}
    assert pose.load_pose_cache(None) == {}  # cache disabled


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
