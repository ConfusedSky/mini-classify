"""src/renderer.py: the resident LRU (byte budget, in_flight pinning, K1
release semantics), tile/view shapes, the rotated-copy view path (I11), and
the child-owned render saving — against a fake OffscreenRenderer, because the
real one needs the GPU and, once created, must live for the process lifetime
(CLAUDE.md). What the fake pins is the *op log*: which geometry is uploaded,
shown, and removed. The pixels are eval/views_camera_rotation.py's job."""
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from src import renderer as renderer_mod
from src.loader import LoadedMesh
from src.messages import RenderConfig
from src.renderer import ROTATED_NAME, Renderer, render_key


class FakeInnerScene:
    def set_sun_light(self, direction, color, intensity):
        pass


class FakeScene:
    def __init__(self, log):
        self.scene = FakeInnerScene()
        self.log = log
        self.geoms = {}                      # name -> visible
        self.uploaded = {}                   # name -> the mesh last handed over

    def add_geometry(self, name, mesh, mat):
        assert name not in self.geoms, f"double add of {name}"
        self.geoms[name] = True              # Open3D adds visible
        self.uploaded[name] = mesh
        self.log.append(("add", name))

    def show_geometry(self, name, visible):
        assert name in self.geoms, f"show of unknown {name}"
        self.geoms[name] = visible
        self.log.append(("show", name, visible))

    def remove_geometry(self, name):
        assert name in self.geoms, f"remove of unknown {name}"
        del self.geoms[name]
        self.log.append(("remove", name))


class FakeOffscreen:
    """What Renderer touches after construction — never destroyed, like the
    real one."""

    def __init__(self, log):
        self.scene = FakeScene(log)

    def setup_camera(self, fov, center, eye, up):
        pass

    def render_to_image(self):
        return np.zeros((4, 4, 3), dtype=np.uint8)


def make_cfg(tmp_path=None, **over):
    kw = dict(render_size=64, views=8, elevations=(20.0, -20.0),
              save_renders_dir=None, render_format="png",
              budget_bytes=1_000_000, collection_root=Path("/nowhere"))
    kw.update(over)
    return RenderConfig(**kw)


@pytest.fixture
def rig(monkeypatch):
    log = []
    monkeypatch.setattr(renderer_mod, "make_offscreen",
                        lambda size: FakeOffscreen(log))

    def make(cfg=None):
        return Renderer(cfg or make_cfg()), log
    return make


def lm(nbytes=100, name="a.stl"):
    return LoadedMesh(file=Path(f"/nowhere/{name}"),
                      mesh=o3d.geometry.TriangleMesh.create_box(),
                      nbytes=nbytes)


# --- shapes ------------------------------------------------------------------

def test_pose_tiles_is_the_six_by_azimuth_grid(rig):
    r, _ = rig()
    grid = r.pose_tiles(lm(), 0)
    assert len(grid) == 6                    # one row per UP_CANDIDATE
    assert all(len(row) == renderer_mod.UP_TILE_AZIMUTHS for row in grid)
    assert all(isinstance(t, np.ndarray) for row in grid for t in row)


def test_pose_tiles_n_az_defaults_to_production_and_is_a_camera_subset(rig):
    """`n_az` exists for eval/tile_count.py, which sweeps the tile count. The
    default must stay production's, and a smaller n_az must be an exact camera
    subset of a larger one — that is what lets the sweep slice a cached grid
    instead of re-rendering per n_az."""
    r, _ = rig()
    assert all(len(row) == renderer_mod.UP_TILE_AZIMUTHS
               for row in r.pose_tiles(lm(), 0))          # default = production
    assert all(len(row) == 4 for row in r.pose_tiles(lm(), 1, n_az=4))
    assert all(len(row) == 1 for row in r.pose_tiles(lm(), 2, n_az=1))
    four = renderer_mod.view_angles(4, [renderer_mod.UP_TILE_ELEVATION])
    for n in (2, 1):
        want = renderer_mod.view_angles(n, [renderer_mod.UP_TILE_ELEVATION])
        assert all(a in four for a in want), n


def test_views_count_is_views_times_elevations(rig):
    r, _ = rig()
    views = r.views(lm(), 0, (0.0, 0.0, 1.0))
    assert len(views) == 8 * 2
    assert all(isinstance(v, np.ndarray) for v in views)


# --- residency and pinning ---------------------------------------------------

def test_pose_tiles_pins_and_views_is_the_consuming_clear(rig):
    r, _ = rig()
    r.pose_tiles(lm(), 3)
    assert r.resident["m3"].in_flight        # awaiting a pose answer
    r.views(None, 3, (0.0, 0.0, 1.0))        # the EmbedRenderTask consumes it
    assert not r.resident["m3"].in_flight    # K1: clear #1


def test_release_is_clear_number_two_and_unknown_is_a_noop(rig):
    r, _ = rig()
    r.pose_tiles(lm(), 3)
    r.release(3)
    assert not r.resident["m3"].in_flight
    r.release(3)                             # already cleared: no-op
    r.release(999)                           # unknown: no-op (K1)


def test_revisit_skips_the_loader_and_uploads_a_rotated_copy(rig):
    """The residency win after I11: the pose->embed revisit needs no
    LoadedMesh (no re-parse), and the original mesh `m5` is never re-uploaded
    — but the rotated copy is uploaded per visit, by design."""
    r, log = rig()
    r.pose_tiles(lm(), 5)
    assert r.is_resident(5)
    del log[:]
    r.views(None, 5, (0.0, 1.0, 0.0))        # residency hit: no LoadedMesh
    assert [e for e in log if e[0] == "add"] == [("add", ROTATED_NAME)]


def test_views_without_mesh_on_a_miss_raises(rig):
    r, _ = rig()
    assert not r.is_resident(7)
    with pytest.raises(ValueError):          # the child loop converts to
        r.views(None, 7, (0.0, 0.0, 1.0))    # Failure


# --- the view path: a rotated copy, never the resident original (I11) --------

def test_views_renders_a_copy_and_removes_it_again(rig):
    r, log = rig()
    r.views(lm(), 1, (0.0, 1.0, 0.0))
    assert [e for e in log if e[0] in ("add", "remove")] == [
        ("add", ROTATED_NAME), ("remove", ROTATED_NAME)]
    assert r._renderer.scene.geoms == {}      # nothing left in the scene
    assert r.is_resident(1)                   # but the mesh stays resident


def test_views_hides_the_resident_geometry_while_the_copy_shoots(rig):
    r, log = rig()
    r.pose_tiles(lm(), 1)                     # uploads and shows m1
    del log[:]
    r.views(None, 1, (0.0, 1.0, 0.0))
    assert log.index(("show", "m1", False)) < log.index(("add", ROTATED_NAME))
    assert not r._renderer.scene.geoms["m1"]
    assert r._visible is None                 # the copy is gone; nothing shows


def test_views_never_mutates_the_resident_mesh(rig):
    """The copy is what gets rotated — otherwise a second visit at another up
    would rotate an already-rotated mesh."""
    r, _ = rig()
    r.views(lm(), 1, (0.0, 1.0, 0.0))
    before = np.asarray(r.resident["m1"].mesh.vertices).copy()
    rotated = np.asarray(
        r._renderer.scene.uploaded[ROTATED_NAME].vertices).copy()
    r.views(None, 1, (0.0, 1.0, 0.0))
    after_copy = np.asarray(r._renderer.scene.uploaded[ROTATED_NAME].vertices)
    assert not np.allclose(before, rotated)   # the copy really was rotated
    assert np.allclose(np.asarray(r.resident["m1"].mesh.vertices), before)
    assert np.allclose(after_copy, rotated)   # same up, same pixels-to-be


# --- the byte-budgeted LRU ---------------------------------------------------

def test_over_budget_evicts_the_least_recently_used(rig):
    r, log = rig(make_cfg(budget_bytes=100))
    r.views(lm(60, "a.stl"), 1, (0.0, 0.0, 1.0))
    r.views(lm(60, "b.stl"), 2, (0.0, 0.0, 1.0))
    assert list(r.resident) == ["m2"]
    # an embed-only visit never uploads the original, so eviction has no
    # geometry to remove — only the rotated copies were ever in the scene
    assert not any(e[0] == "remove" and e[1] != ROTATED_NAME for e in log)


def test_eviction_removes_the_geometry_the_pose_path_uploaded(rig):
    r, log = rig(make_cfg(budget_bytes=100))
    r.pose_tiles(lm(60, "a.stl"), 1)
    r.release(1)                             # unpinned: normal LRU eligibility
    r.views(lm(60, "b.stl"), 2, (0.0, 0.0, 1.0))
    assert ("remove", "m1") in log
    assert list(r.resident) == ["m2"]


def test_lru_order_follows_touches_not_insertion(rig):
    r, _ = rig(make_cfg(budget_bytes=120))
    r.views(lm(50, "a.stl"), 1, (0.0, 0.0, 1.0))
    r.views(lm(50, "b.stl"), 2, (0.0, 0.0, 1.0))
    r.views(None, 1, (0.0, 0.0, 1.0))        # touch m1: m2 is now LRU
    r.views(lm(50, "c.stl"), 3, (0.0, 0.0, 1.0))
    assert set(r.resident) == {"m1", "m3"}   # m2 evicted, not m1


def test_in_flight_meshes_are_never_evicted(rig):
    r, log = rig(make_cfg(budget_bytes=100))
    r.pose_tiles(lm(60, "a.stl"), 1)         # pinned
    r.views(lm(60, "b.stl"), 2, (0.0, 0.0, 1.0))
    assert "m1" in r.resident                # soft bound: budget exceeded
    assert not any(e == ("remove", "m1") for e in log)
    r.release(1)                             # dropped to LRU eligibility
    r.views(lm(60, "c.stl"), 3, (0.0, 0.0, 1.0))
    assert "m1" not in r.resident


def test_evicting_the_visible_mesh_keeps_the_scene_consistent(rig):
    r, _ = rig(make_cfg(budget_bytes=100))
    r.pose_tiles(lm(60, "a.stl"), 1)         # pinned, skipped by eviction
    r.pose_tiles(lm(30, "b.stl"), 2)
    r.release(2)                             # visible, and now evictable
    r.pose_tiles(lm(60, "c.stl"), 3)         # forces m2 out
    assert set(r.resident) == {"m1", "m3"}
    assert r._visible == "m3"                # no dangling reference to m2


# --- saving (the child owns writing renders) ---------------------------------

def test_save_renders_writes_under_the_render_key(rig, tmp_path):
    root = tmp_path / "collection"
    (root / "kit").mkdir(parents=True)
    f = root / "kit" / "model.stl"
    f.write_bytes(b"")                       # rel_path resolves real paths
    cfg = make_cfg(save_renders_dir=tmp_path / "renders",
                   render_format="png", collection_root=root)
    r, _ = rig(cfg)
    images = [np.zeros((4, 4, 3), dtype=np.uint8)] * 3
    r.save_renders(f, images)
    key = render_key(f, root)
    written = sorted(p.name for p in (tmp_path / "renders").iterdir())
    assert written == [f"{key}_view{i}.png" for i in range(3)]


def test_save_renders_is_a_noop_without_a_directory(rig):
    r, _ = rig()                             # save_renders_dir=None
    r.save_renders(Path("/nowhere/a.stl"),
                   [np.zeros((4, 4, 3), dtype=np.uint8)])   # must not raise


def test_save_renders_never_fails_the_run(rig, tmp_path, capsys):
    """Parity with today's save_renders: an OSError is reported, not raised —
    these files exist for a human, and the caller still sends its ack after
    this returns (K6)."""
    blocker = tmp_path / "renders"
    blocker.write_bytes(b"not a directory")
    cfg = make_cfg(save_renders_dir=blocker, collection_root=tmp_path)
    r, _ = rig(cfg)
    r.save_renders(tmp_path / "a.stl", [np.zeros((4, 4, 3), dtype=np.uint8)])
    assert "could not save renders" in capsys.readouterr().out
