"""The render child's Renderer: pose tiles, classification views, and the
byte-budgeted resident-mesh LRU (docs/actor-refactor/interfaces.md §Render
child, data_structures.md §Renderer-child mesh residency).

Hard constraints this module is built around (CLAUDE.md):

* The `OffscreenRenderer` is created once and **never destroyed** — Filament
  throws from a destructor, so teardown aborts. The child exits via
  `os._exit(0)` (render_child.py) precisely so interpreter teardown never
  reaches it. There is no `close()` here on purpose.
* The classification views rotate **a copy of the mesh** (`mesh.rotate`),
  never the camera — I11, resolved 2026-08-17 against the interfaces note's
  draft rule (interfaces.md §Render child). The camera trick was built
  first and measured non-identical: under the production config it moved up
  to 75/255 on 54% of pixels against a 2/255 repeat floor, and with
  post-processing off (byte-stable renders) 15 of 18 model x up cases
  differed by up to 31/255 on ~14-20% of pixels — only the identity
  rotation matched (`eval/views_camera_rotation.py`,
  `docs/learnings/2026-08-17-camera-rotation-and-the-world-fixed-fill.md`).
  Mechanism: the ambient fill is a **world-fixed environment map** and this
  Open3D build exposes no `set_indirect_light_rotation`, so rotating the rig
  instead of the geometry changes what the fill illuminates; the
  indirect-light-off arm drops the residual to 1/255 on <=5.3% of pixels for
  11 of 15 (the curved mesh keeps 7-18/255 on <=3.1% across bunny's four
  side ups, silhouette resampling — counts per A-R1-2). Those pixels are what every cached embedding was computed
  from, so the copy is what keeps the caches valid.
* **The resident original is never mutated.** `views` copies it, rotates the
  copy, uploads the copy under `ROTATED_NAME`, and removes it when the shot
  is done. Residency therefore still saves the parse+load on the pose→embed
  revisit and pays the ~275 ms upload again per visit — accepted by design
  (data_structures.md §Renderer-child mesh residency).
* `pose_tiles` keeps the camera-carried rotation. Production
  (`render_up_candidate_grid`) has always rendered the pose tiles that way,
  the pose cache's provenance is those pixels, and `pose_tiles` reproduces
  the path exactly. Its *framing* is exact — resolved ups are the six signed
  axis candidates (pose.UP_CANDIDATES), and a signed axis permutation maps
  the AABB to the permuted box, so the rotated centre is `R @ center` with
  the radius unchanged.

Rendering extraction source: `classify_stls.py` (make_renderer, orbit_camera,
view_angles, render_up_candidate_grid's camera trick, save_renders,
RENDER_FORMATS). This module is the one home for all of it except
`view_angles` and `rotation_to_z_up`, which moved on to `src/pose.py`
(2026-08-19) so a consumer can name a camera angle, or the rotation a pose
implies, without importing open3d — both are pure numpy and neither ever
touched a mesh. `Renderer` calls them through `pose.` rather than importing
the names, so that `renderer.view_angles` does not resolve and this module
cannot become a re-export of them (review, 2026-08-19) — the CLI's
parallel single-process render path was deleted in the eval-debt cleanup
(2026-08-18) and `eval/rig.py` drives the `Renderer` below instead — and
`render_key` moved down to `identity` (the parent's cache checker needs it
too). The child owns saving renders (data_structures Q2): the pixels are
already in its memory. The CLI still reads `RENDER_FORMATS` for
`--render-format`'s choices, inside `main()` so that its module scope stays
open3d-free (interfaces.md's import table).
"""
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from PIL import Image

from src import pose
from src.identity import render_key
from src.loader import LoadedMesh
from src.messages import RenderConfig

SUN_INTENSITY = 90000.0
# Ambient fill. The sun is the only light Filament gives us here —
# add_directional_/point_/spot_light all return True and then render as a
# <0.1/255 no-op — so with indirect light off, every surface facing away from
# the key falls to pure black and swallows detail (an 11% of object pixels
# under 25/255 on a hat brim shading a face). The built-in environment map is
# world-fixed and does not orbit with the camera, so it is deliberately kept
# far below the key: at 10k the crushed-black fraction is 0, while the
# brightness it adds still swings ~30/255 across azimuths.
FILL_INTENSITY = 10000.0

UP_TILE_ELEVATION = 20.0  # pose contact sheet: fixed, independent of --elevations
# Views per up candidate for the ensemble; 2, not 4: measured on all 49 labels
# at production pixels, halving the azimuths flips zero ensemble picks and
# costs one extra escalation, for half of pose-embed — the run's largest GPU
# item (eval/tile_count.py, LEARNINGS 2026-08-13). 1 is not safe: it breaks
# three models including 32mm_Gate_L.
UP_TILE_AZIMUTHS = 2

# Scene name for the rotated copy `views` renders. One name, added and removed
# per visit: the copy is never resident — the mesh it was rotated from is.
ROTATED_NAME = "_rot"

# Encodings for saved renders. Written and never read back — the classifier
# always embeds the in-memory render — so a lossy format costs only what a
# human eye needs. Measured at 2048 px: jpg 0.13 s / 205 KB, png 3.83 s /
# 3.9 MB, webp 1.02 s / 100 KB; compress_level=1 is byte-identical to PIL's
# default 6 and 6.1x faster.
RENDER_FORMATS = {
    "jpg": (".jpg", {"quality": 92}),
    "png": (".png", {"compress_level": 1}),
    "webp": (".webp", {"quality": 90}),
}


def orbit_camera(center, radius, az, elev):
    """(eye, camera-up, sun direction) for one turntable position, Z-up world."""
    eye = center + radius * np.array(
        [np.cos(az) * np.cos(elev), np.sin(az) * np.cos(elev), np.sin(elev)]
    )
    # Camera 'up' is world +Z carried along the orbit, not +Z itself: the two
    # frame the image identically, but past |elev| 87.44 (|up . view| > 0.999)
    # Filament calls +Z degenerate and swaps in a fixed fallback up, which
    # freezes the image orientation so azimuth stops changing the render.
    up = np.array([-np.cos(az) * np.sin(elev), -np.sin(az) * np.sin(elev), np.cos(elev)])
    # headlight: key light shines from the camera, tilted downward in world
    # space so shading is consistent with "up" from every orbit angle
    sun = (center - eye) / np.linalg.norm(center - eye) + np.array([0, 0, -0.6])
    return eye, up, sun / np.linalg.norm(sun)


def rotated_cams(R, center, radius, angles):
    """The camera-carried rotation: each camera computed in the frame R takes
    the mesh to, then carried back with R.T so the geometry never moves.

    Exact as *framing* when R is a signed axis permutation — the rotated AABB
    is the permuted box, so its centre is R @ center and the framing radius is
    unchanged — which is why the pose path (`_shoot_rotated`) can shoot six
    candidate ups from one upload. Exact as *framing*, not as pixels: the
    ambient fill is
    world-fixed, which is why the classification views rotate a copy of the
    mesh instead (I11, module docstring)."""
    c_rot = R @ center
    cams = []
    for az, elev in angles:
        eye, cam_up, sun = orbit_camera(c_rot, radius, az, elev)
        cams.append((R.T @ c_rot, R.T @ eye, R.T @ cam_up, R.T @ sun))
    return cams


def make_offscreen(size):
    """The one `OffscreenRenderer`, scene rig included (key light from above
    plus ambient fill — meshes are rotated into Z-up world space before
    rendering, so the rig is correct for any --up-axis choice). Module-level so
    tests can substitute a fake without touching the GPU. The returned renderer
    is kept for the process lifetime and never destroyed (CLAUDE.md)."""
    renderer = rendering.OffscreenRenderer(size, size)
    scene = renderer.scene.scene
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    scene.enable_indirect_light(True)
    scene.set_indirect_light_intensity(FILL_INTENSITY)
    scene.enable_sun_light(True)  # direction is set per view in _shoot
    return renderer


@dataclass
class ResidentMesh:
    mesh: o3d.geometry.TriangleMesh  # host-side, never mutated: `views` rotates
                                     # a copy of it (I11). Holding it is what
                                     # saves the parse on a residency hit, and
                                     # what `nbytes` has always accounted for
    center: np.ndarray             # framing of the unrotated mesh, kept to avoid
    radius: float                  # recompute — the pose tiles' framing
    nbytes: int
    in_flight: bool                # awaiting a pose answer — exempt from eviction
    uploaded: bool = False         # the original is in the scene. Only the pose
                                   # path uploads it; `views` uploads a rotated
                                   # copy instead, so an embed-only visit leaves
                                   # this False and eviction has nothing to remove


class Renderer:
    """Owns the scene and residency (single-writer, invariant 3). The pose
    path's geometry is added under a unique name per admitted index and
    hidden rather than destroyed on a switch — re-show is 34 ms against
    369 ms remove+re-add (actors_proposal.md, Loader §). The view path's
    rotated copy is the exception: it is added and removed per visit,
    because it is not the resident mesh (I11, module docstring)."""

    def __init__(self, cfg: RenderConfig):
        self.cfg = cfg
        self._renderer = make_offscreen(cfg.render_size)    # never destroyed
        self._material = rendering.MaterialRecord()
        self._material.shader = "defaultLit"
        self._material.base_color = [0.7, 0.7, 0.7, 1.0]
        self.resident: OrderedDict[str, ResidentMesh] = OrderedDict()
        self._visible: str | None = None

    # --- residency -----------------------------------------------------------

    @staticmethod
    def _name(index: int) -> str:
        return f"m{index}"

    def is_resident(self, index: int) -> bool:
        """True when `views` can run without a `LoadedMesh` — the child loop
        skips `loader.get` on the pose->embed revisit, which is the residency
        win that survived I11 (no re-parse; the rotated copy is re-uploaded)."""
        return self._name(index) in self.resident

    def release(self, index: int) -> None:
        """The `Release` control message: drop the mesh to normal LRU
        eligibility. Unknown or already-cleared indices are a no-op — what
        lets the parent send it unconditionally (K1)."""
        rm = self.resident.get(self._name(index))
        if rm is not None:
            rm.in_flight = False

    def _admit(self, lm: LoadedMesh | None, index: int, pin: bool) -> ResidentMesh:
        """Register `index` in the resident LRU (evicting as needed) and set
        its pin, without touching the scene — uploading is the caller's, and
        the two callers upload different geometry (the original for
        `pose_tiles`, a rotated copy for `views`). `pin` sets `in_flight`
        (the pose pass); `pin=False` is the consuming clear — the
        `EmbedRenderTask` that uses the mesh is one of `in_flight`'s two
        clears, `release` the other (K1)."""
        name = self._name(index)
        rm = self.resident.get(name)
        if rm is None:
            if lm is None:
                raise ValueError(f"index {index} not resident and no mesh given")
            self._evict_for(lm.nbytes)
            bounds = lm.mesh.get_axis_aligned_bounding_box()
            rm = ResidentMesh(
                mesh=lm.mesh,
                center=np.asarray(bounds.get_center(), dtype=float),
                radius=float(np.linalg.norm(bounds.get_extent()) * 1.4),
                nbytes=lm.nbytes, in_flight=False)
            self.resident[name] = rm
        rm.in_flight = pin
        self.resident.move_to_end(name)                     # LRU touch (D14)
        return rm

    def _hide_visible(self) -> None:
        """Hide whatever is showing: one geometry renders at a time. Hidden,
        never removed — re-show is 34 ms against 369 ms remove+re-add."""
        if self._visible is not None:
            self._renderer.scene.show_geometry(self._visible, False)
            self._visible = None

    def _show(self, lm: LoadedMesh | None, index: int, pin: bool):
        """`_admit`, then make the mesh *as loaded* the visible geometry —
        uploading it on the first visit — and return its framing. The pose
        path only: the view path renders a rotated copy instead."""
        rm = self._admit(lm, index, pin)
        name = self._name(index)
        if not rm.uploaded:
            self._renderer.scene.add_geometry(name, rm.mesh, self._material)
            rm.uploaded = True
        if self._visible != name:
            self._hide_visible()
        self._renderer.scene.show_geometry(name, True)
        self._visible = name
        return rm.center, rm.radius

    def _evict_for(self, incoming: int) -> None:
        """Evict from the LRU front until the budget holds the incoming mesh.
        `in_flight` entries are never evicted, so `budget_bytes` is a soft
        bound — the hard worst case is the admission window x the heaviest
        mesh (D14)."""
        total = sum(rm.nbytes for rm in self.resident.values()) + incoming
        for name in list(self.resident):
            if total <= self.cfg.budget_bytes:
                break
            rm = self.resident[name]
            if rm.in_flight:
                continue
            if rm.uploaded:                                 # nothing in the scene
                self._renderer.scene.remove_geometry(name)  # for an embed-only
            del self.resident[name]                         # visit. Geometry,
                                                            # never the renderer
            total -= rm.nbytes
            if self._visible == name:
                self._visible = None

    # --- rendering -----------------------------------------------------------

    def _shoot(self, cams) -> list[np.ndarray]:
        """One image per (center, eye, up, sun), geometry as loaded. Arrays,
        not PIL: arrays are what cross the boundary (data_structures.md)."""
        images = []
        for center, eye, up, sun in cams:
            self._renderer.setup_camera(45.0, center, eye, up)
            self._renderer.scene.scene.set_sun_light(sun, [1.0, 1.0, 1.0], SUN_INTENSITY)
            images.append(np.asarray(self._renderer.render_to_image()).copy())
        return images

    def _shoot_rotated(self, R, center, radius, angles) -> list[np.ndarray]:
        """The pose path's shot (`pose_tiles`): `rotated_cams`, then fire. The
        rotation is carried by the cameras, not the geometry — one upload for
        all six candidates — and `views` deliberately does not use it (I11)."""
        return self._shoot(rotated_cams(R, center, radius, angles))

    def pose_tiles(self, lm: LoadedMesh, index: int,
                   n_az: int = UP_TILE_AZIMUTHS) -> list[list[np.ndarray]]:
        """[6][n_az] tiles — each candidate up seen from `n_az` azimuths, one
        upload total. Pins the mesh resident (`in_flight=True`): it is awaiting
        a pose answer, and the pose->embed revisit is the one revisit residency
        exists for.

        `n_az` is production's `UP_TILE_AZIMUTHS` and the child never passes
        anything else — it exists because the tile count is a *measured*
        parameter (eval/tile_count.py sweeps 4/2/1) and the harness that sweeps
        it must sweep the production call rather than a copy of it. The
        azimuths are `view_angles(n_az, [UP_TILE_ELEVATION])`, so a smaller
        n_az is an exact camera subset of a larger one."""
        center, radius = self._show(lm, index, pin=True)
        angles = pose.view_angles(n_az, [UP_TILE_ELEVATION])
        return [self._shoot_rotated(pose.rotation_to_z_up(up), center, radius, angles)
                for up in pose.UP_CANDIDATES]

    def views(self, lm: LoadedMesh | None, index: int, up) -> list[np.ndarray]:
        """The classification views: cfg.views azimuths per elevation, rendered
        from a **rotated copy** of the mesh — `mesh.rotate` on a copy, never
        the camera (I11; module docstring). The resident original is left
        alone, so a second visit at a different up rotates from the same
        unrotated geometry. `lm` may be None when the mesh is already resident
        (`is_resident`) — that hit saves the parse+load, not the upload:
        the copy is uploaded, shot, and removed each visit (~275 ms, accepted
        by design). Consuming the mesh clears `in_flight` (K1)."""
        rm = self._admit(lm, index, pin=False)
        rot = o3d.geometry.TriangleMesh(rm.mesh)            # copy, then rotate
        rot.rotate(pose.rotation_to_z_up(np.asarray(up, dtype=float)),
                   center=(0, 0, 0))
        bounds = rot.get_axis_aligned_bounding_box()        # framing from the
        center = np.asarray(bounds.get_center(), dtype=float)   # rotated copy,
        radius = float(np.linalg.norm(bounds.get_extent()) * 1.4)  # as today's
        angles = pose.view_angles(self.cfg.views, list(self.cfg.elevations))  # path
        cams = [(center, *orbit_camera(center, radius, az, elev))
                for az, elev in angles]
        self._hide_visible()                     # only the copy in the shot
        scene = self._renderer.scene
        scene.add_geometry(ROTATED_NAME, rot, self._material)
        try:
            return self._shoot(cams)
        finally:
            scene.remove_geometry(ROTATED_NAME)  # per visit; never resident

    # --- saving (the child owns writing renders — data_structures Q2) --------

    def save_renders(self, file: Path, images: list[np.ndarray]) -> None:
        """Write the debug renders under a render_key() prefix into
        cfg.save_renders_dir (already the per-config directory — the parent
        derives it, the child never reads args). Never fails the run — like
        today's save_renders, these exist for a human to look at, not for the
        classifier — but the caller must still send its ack only after this
        returns (K6)."""
        rdir = self.cfg.save_renders_dir
        if rdir is None:
            return
        key = render_key(file, self.cfg.collection_root)
        ext, opts = RENDER_FORMATS[self.cfg.render_format]
        try:
            rdir.mkdir(parents=True, exist_ok=True)
            for i, im in enumerate(images):
                Image.fromarray(im).save(rdir / f"{key}_view{i}{ext}", **opts)
        except OSError as e:
            print(f"  could not save renders for {key}: {e}")
