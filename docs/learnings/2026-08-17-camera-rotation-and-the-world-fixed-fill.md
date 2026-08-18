## Camera rotation and the world-fixed fill (2026-08-17)

The actor refactor's interfaces note carried one rule as a design decision that
had never been measured: `renderer.views` must carry the up-rotation **in the
camera** (`R.T`), never `mesh.rotate`, because "residency only pays if the
resident geometry is reusable as-is". The interfaces review flagged it as a
precondition rather than a fact (I11) — `R.T` was proven pixel-identical only
for the *pose tile* grid at one elevation, while the classification views span
8 azimuths × 2 elevations. The child's `views` was built to the rule, the check
was run, and the rule lost: the camera trick does not reproduce production's
pixels, so `views` now rotates a **copy** of the resident mesh. Harness:
`eval/views_camera_rotation.py`; raw output in
`eval/out/views_camera_rotation.json`, which is gitignored — the figures here
are the record.

### The check has to be run with post-processing off

Filament's default post-processing is temporally dithered: repeats of the *same*
render differ on ~43% of pixels (`eval/render_determinism.py`, review V1), which
is more noise than any rotation question. So the measurement runs three arms —
`nopost` (post-processing off, production lighting: the real test), `noibl`
(`nopost` plus indirect light off, the attribution arm), and `default` (one
model under the production config, to put the deltas beside the noise the
pipeline already carries). Every case renders 3 STLs × 6 candidate ups × 16
views, each path twice, so repeat-stability is established before anything is
compared.

### Result: the camera trick is not the same image

| arm | camera-rotated vs `mesh.rotate` |
|---|---|
| `nopost`, production lighting | **15 of 18** cases differ, max 31/255 on 13.8–19.7% of pixels |
| `noibl`, indirect light off | 15 of 18 differ, but 11 at max **1/255** on ≤5.3% of pixels |
| `default`, production config | max **75/255 on 53.7%** of pixels, against a 2/255 repeat floor |

Only the identity rotation (up = +Z) matched in any arm, which is the tell: the
difference appears exactly when the rig is rotated relative to the world.

### Mechanism: the fill is a world-fixed environment map

The sun is the only light Filament gives us here; the ambient fill is the
built-in **indirect light**, an environment map fixed in world space
(`classify_stls.py:61-69`). Rotating the mesh turns the geometry inside that
map; rotating the camera rig instead leaves the geometry where it was and lights
it from a different side. This Open3D build exposes no
`set_indirect_light_rotation`, so there is no way to carry the map along.

The `noibl` arm is the attribution: with the fill off, eleven of the fifteen
failures collapse to a single least-significant bit on a few percent of pixels.
The four that do not are all the curved mesh — bunny at ±Y (7–8/255 on ≤0.4%)
and ±X (17–18/255 on ≤3.1%) — sub-pixel silhouette resampling, not lighting. Take the fill away and the two
paths are the same image; put it back and they are not.

### Decision: rotate a copy — the caches decide it, not the throughput

Every embedding in `embed-cache*/` was computed from `mesh.rotate` pixels
(`classify_stls.render_views`). A camera-rotated `views` would return different
images under the *same* cache keys, so a rebuilt cache and an existing one would
disagree by up to 75/255 on half the pixels with nothing to mark the change.
That is not a tradeoff against a few hundred milliseconds; it decides the
question on its own.

So `views` copies the resident mesh, rotates the copy, uploads it, shoots, and
removes it. What residency keeps and what it pays:

* **Kept**: the parse+load on the pose→embed revisit (11–66 ms with the numpy
  parser, and `read_triangle_mesh` where it falls back), plus the LRU, the byte
  budget, and the `in_flight` pinning exactly as designed.
* **Paid**: the ~275 ms upload of the rotated copy on every visit, instead of a
  34 ms re-show of a hidden geometry.
* **Not paid, as it turns out**: any revision to the throughput number. The
  roundtrip spike that produced 1.11× held meshes *host-side* and rotated them
  before rendering (`eval/overlap_spike.py:101-103`) — it never re-showed a
  resident geometry. The measured number was always measured on the design we
  have now adopted; it was the camera-rotation design that was unmeasured.

`pose_tiles` keeps the camera-carried rotation. Production's
`render_up_candidate_grid` has always rendered the tiles that way — one upload,
six candidate rotations moved into the camera — and the pose cache's provenance
is those pixels. The same argument that forces the copy on the view path forbids
it on the pose path.

### Verifying the rework: 16/18 byte-identical, 2 at the renderer's own floor

The same harness now runs a third path — the reworked `Renderer.views` — against
the same `mesh.rotate` reference. In the `nopost` arm, 16 of 18 cases are
**byte-identical**. The two that are not are `blocky_building` at up=+Y and
up=+X, differing by max 1/255 on **27 and 70 pixels** of 7,077,888 — and each is
unstable against *itself* by exactly the same 27 and 70 pixels when the identical
call is repeated. The residual is the renderer's floor under the add/remove churn
this harness does, not a difference between the paths. Under the `default` config
the reworked path differs from the reference by max 2/255 on 43.9% of pixels —
which is the repeat-noise figure to three digits, i.e. indistinguishable from
dither, against the camera path's 75/255 on 53.7%.

The repeat also guards the invariant that makes the copy safe: the resident
original is never mutated, so a second visit at a different up rotates from
unrotated geometry. A mutated original would return a double-rotated image on the
residency hit; 16 of 18 cases show the hit and the cold visit producing
identical bytes, and the other two stay within their own 27- and 70-pixel
repeat floor — a double rotation would differ massively, not by a floor's
worth (A-R1-3).

### Second-order: a re-shown geometry is not a freshly uploaded one

Worth recording because it will bite whoever reintroduces re-show into a pixel
path. In the camera arm, the fresh-upload render and the hidden-then-re-shown
render of the *same* geometry differed on 11 of 18 cases, up to 8/255 — and with
indirect light off, on 1 of 18 at 1/255. Filament's `show_geometry` round trip is
not pixel-neutral under the environment fill.

It costs nothing today: `views` uploads a fresh copy every visit, and
`pose_tiles` renders all twelve tiles from one upload, so each path is internally
consistent. But "re-show is 34 ms against 369 ms remove+re-add" is a throughput
fact, not a pixel-equivalence fact, and the two were quietly being used
interchangeably.

### What changed

`src/renderer.py` (`views` rotates a copy; `ResidentMesh` holds the host-side
mesh and an `uploaded` flag, since an embed-only visit never puts the original in
the scene), `src/render_child.py` (comment), `tests/test_renderer.py` (op-log
assertions for the add/remove-per-visit copy, plus a test that the resident mesh
is never mutated), `eval/views_camera_rotation.py`, and the `renderer.views`
bullet in `docs/actor-refactor/interfaces.md`.
