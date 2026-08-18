"""The production render/embed objects, in the shape a harness wants.

Every harness here used to reach into `classify_stls` for a second,
single-process arrangement of the render and embed maths — `make_renderer`,
`render_up_candidate_grid`, `render_views`, `resolve_up`, `embed_images`. Two
copies of a forward pass is one copy too many: they drifted (review §4.1), and
a number measured on the copy was only ever a statement about the copy. This
module is the adapter that removed the need for them. It constructs the
*production* objects — `src.renderer.Renderer` on a real `RenderConfig`,
`src.embedder.Embedder`, `src.loader.get` — and exposes them as the free
functions the harnesses were written against.

What it is not: a re-implementation. Every function below is a handful of
lines of plumbing over a production call. If a harness needs behaviour that
production does not have, that is a finding about production, not a reason for
a branch in here.

Two hard constraints it exists to hold (CLAUDE.md):

* **The `OffscreenRenderer` is created once and never destroyed.** Filament
  throws from a destructor, so interpreter teardown over a live renderer
  aborts the process. `rig()` therefore caches renderers per config for the
  process lifetime and hands the same one back, and a script that has built
  one must leave through `exit_without_teardown()` rather than falling off the
  end of `main()`. `eval/views_camera_rotation.py` is the worked example.
* **Rendering runs on the AMD iGPU, SigLIP on the 4060.** Nothing here
  interleaves them; a harness that renders and embeds should still do it in
  two phases, because the two towers evict each other on an 8 GB card.

Residency: the production `Renderer` keeps meshes in a byte-budgeted LRU keyed
by an integer index, because the pipeline revisits a mesh between pose and
embed. A harness visits each mesh once, so `pose_tiles`/`views` below allocate
a fresh index per call and release it immediately — the mesh then falls out of
the LRU on the next admission, which is the same add/remove churn the old
`_upload`'s `clear_geometry()` had.

Arrays, not PIL: `src.renderer` returns `np.ndarray` where the CLI's helpers
returned `PIL.Image`. A harness that saves a render needs one
`Image.fromarray`; a harness that embeds one needs nothing, because the SigLIP
processor takes either.
"""
import itertools
import os
import sys
from pathlib import Path

import numpy as np

# Importable on its own, without `common` having run first: the harnesses put
# the repo on sys.path through common.py, but tests/test_embedder.py imports
# this module directly.
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src import loader
from src.embedder import DEFAULT_MODEL, Embedder
from src.loader import LoadedMesh
from src.messages import RenderConfig
from src.renderer import UP_TILE_AZIMUTHS, UP_TILE_ELEVATION, Renderer

# Renderers, per (size, views, elevations). Never destroyed, never dropped —
# holding them here is what keeps a script from letting one reach a garbage
# collector (see the module docstring).
_RENDERERS: dict[tuple, Renderer] = {}

# Residency indices. Monotonic, so a fresh mesh is never confused with a
# resident one: `_admit` keys on the index and would hand back the *previous*
# mesh if a harness reused a number.
_INDEX = itertools.count(1)

# Big enough that a couple of meshes stay resident (the LRU is not the thing
# under test here), small enough that a 600-model sweep cannot grow without
# bound. Eviction removes the geometry the pose path uploaded, which is what
# the old `_upload`'s clear_geometry did every model.
BUDGET_BYTES = 1 << 30


def rig(size, views=1, elevations=(UP_TILE_ELEVATION,), collection_root=None,
        budget_bytes=BUDGET_BYTES) -> Renderer:
    """The production `Renderer` for one camera config, built once per process.

    `size` is the square render resolution. `views`/`elevations` only matter to
    `views()` below — the pose tiles carry their own fixed elevation
    (`UP_TILE_ELEVATION`) and take their azimuth count per call — so a harness
    that only renders pose tiles can ignore them.

    Calling this twice with the same config returns the same renderer; calling
    it with a different one builds a second, which is fine (four coexisting
    renderers measured correct, docs/reviews/2026-08-13.md §3.1). What is not
    fine is destroying either, so nothing here ever does. The cache key is
    (size, views, elevations) only — a later call's `budget_bytes` or
    `collection_root` does not rebuild an existing renderer.
    """
    elevations = tuple(float(e) for e in elevations)
    key = (int(size), int(views), elevations)
    r = _RENDERERS.get(key)
    if r is None:
        cfg = RenderConfig(render_size=int(size), views=int(views),
                           elevations=elevations, save_renders_dir=None,
                           render_format="png", budget_bytes=budget_bytes,
                           collection_root=Path(collection_root or "/"))
        r = _RENDERERS[key] = Renderer(cfg)
    return r


def load(path) -> LoadedMesh:
    """`src.loader.get` — the child's loader, numpy STL parser included. Raises
    on malformed input, exactly as the render child sees it."""
    return loader.get(path)


def as_loaded(mesh, file="<mesh>") -> LoadedMesh:
    """A `LoadedMesh` around an already-parsed Open3D mesh.

    For the harnesses whose *point* is a different loader — `parser_gate.py`
    scores `read_triangle_mesh` against the numpy parser — so they hand the
    renderer a mesh they parsed themselves rather than going through `load`.
    """
    if isinstance(mesh, LoadedMesh):
        return mesh
    return LoadedMesh(file=Path(file), mesh=mesh,
                      nbytes=loader.mesh_nbytes(mesh))


def index() -> int:
    """A fresh residency index. Pass it to `pose_tiles`/`views` when one mesh
    is visited several times — six candidate ups, say — so the LRU accounts for
    it once instead of once per visit."""
    return next(_INDEX)


def pose_tiles(r: Renderer, mesh, n_az=UP_TILE_AZIMUTHS, index=None) -> list[list[np.ndarray]]:
    """[6][n_az] up-candidate tiles — `Renderer.pose_tiles`, the pipeline's own.

    One upload for all six candidates: the rotation is carried in the cameras
    (`rotated_cams`), which is exact as *framing* because the six candidate ups
    are signed axis permutations. This is the call the render child makes, so
    these are the pixels the pose cache's provenance is.
    """
    i = next(_INDEX) if index is None else index
    try:
        return r.pose_tiles(as_loaded(mesh), i, n_az=n_az)
    finally:
        r.release(i)              # nothing is awaiting a pose answer here


def pose_sheet_tiles(r: Renderer, mesh) -> list[np.ndarray]:
    """The six tiles the VLM arbiter is shown: one azimuth per candidate."""
    return [row[0] for row in pose_tiles(r, mesh, n_az=1)]


def views(r: Renderer, mesh, up, index=None) -> list[np.ndarray]:
    """The classification views for one resolved up — `Renderer.views`.

    `r.cfg.views` azimuths per elevation off a **rotated copy** of the mesh,
    never the camera (I11 — the ambient fill is a world-fixed environment map,
    see eval/views_camera_rotation.py). `up` is a candidate up vector, not an
    index.
    """
    i = next(_INDEX) if index is None else index
    try:
        return r.views(as_loaded(mesh), i, up)
    finally:
        r.release(i)


def embedder(model_name=DEFAULT_MODEL, device=None, categories=("miniature",),
             embed_batch=0, compile_image_forward=False) -> Embedder:
    """The production `Embedder`: SigLIP, plus the four numpy prompt banks.

    `up_T`/`down_T` (the upright ensemble) and `front_T`/`back_T` (front-view
    resolution) come off it already computed, so a harness never embeds the
    probe wordings itself. `categories` only feeds `text_embeds`, which most
    harnesses do not use — one placeholder keeps the startup pass honest
    without pulling categories.txt in.
    """
    return Embedder(list(categories), model_name=model_name, device=device,
                    compile_image_forward=compile_image_forward,
                    embed_batch=embed_batch)


def embed(e: Embedder, images, batch=0) -> np.ndarray:
    """Row-normalised embeddings as float32 numpy — `Embedder.embed_images`.

    The Embedder returns a device tensor because that is what production's
    consumers want (Done scores against it on the GPU). Every harness here
    scores in numpy instead, so the `.float().cpu().numpy()` lands once, here.
    `tests/test_embedder.py` pins this against `embed_tiles`/`embed_views` —
    that assertion is what makes "the harnesses measure the production forward
    pass" a checked claim rather than a comment.
    """
    return e.embed_images(list(images), batch=batch).float().cpu().numpy()


def embed_probe_texts(e: Embedder, texts) -> np.ndarray:
    """Row-normalised float32 embeddings for arbitrary probe wordings.

    The Embedder carries the four banks production uses and no way to ask for a
    fifth, so this reaches for `Embedder._embed_raw` — one **private** name on
    the production object, deliberately, because the alternative is a second
    text forward for the probe sweep to measure instead of the real one. If
    `_embed_raw` is ever made public this call should just lose its underscore;
    if it changes shape, `eval/pose_probe_sweep.py` is the caller to fix.
    """
    return e._embed_raw(list(texts)).float().cpu().numpy()


def exit_without_teardown(code=0):
    """Leave the process without running interpreter teardown.

    Teardown would destroy a live `OffscreenRenderer` and Filament aborts from
    that destructor — the one hard constraint with no workaround (CLAUDE.md).
    Any script that has called `rig()` must end here.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
