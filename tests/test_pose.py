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


def test_geometry_only_pose_is_a_miss_for_an_ensemble_run():
    # margin is None exactly when the ensemble did not run: one --no-up-ensemble
    # pass must not pin the pose, so an ensemble run re-resolves the entry —
    # while a second geometry-only run still gets its cache hit
    geo = {"up": [0, 0, 1], "confidence": 0.4, "source": "geometry", "margin": None}
    assert not pose.pose_is_sufficient(geo, ensemble_available=True)
    assert pose.pose_is_sufficient(geo, ensemble_available=False)


def test_full_run_poses_stay_cached():
    # geometry-with-margin means the ensemble ran and agreed with geometry
    assert pose.pose_is_sufficient({"source": "geometry", "margin": 0.62}, True)
    assert pose.pose_is_sufficient({"source": "siglip", "margin": 0.51}, True)
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
    inspect sys.modules."""
    allowed = "{'src', 'src.pose', 'src.identity'}"
    code = ("import sys; from src import pose; "
            "bad = [k for k in sys.modules if k == 'torch' "
            "or k.startswith('torch.')]; "
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
