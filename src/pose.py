"""Pose resolution for STL renders: up-axis detection with confidence, a
SigLIP tie-break over the up-candidate tiles, front-view scoring, a per-file
pose cache, and a VLM arbiter for ambiguous cases. classify_stls.py
orchestrates; this module never imports classify_stls (no rendering / model
code here) — the scorers here take embeddings as arguments and never load a
model."""
import base64
import io
import json
import os
import tempfile
import time
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src import identity

UP_CANDIDATES = [np.array(u, dtype=float) for u in
                 [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]]

ABS_SCORE_FLOOR = 0.02  # best flat-base score below this = "no print base found"
SAMPLE_SEED = 0

# Escalate to the VLM when the *ensemble* is unsure, not when geometry is.
# 0.45 was picked on the `orig` set and read on the holdout (21/21 there, 43/44
# pooled, firing on ~20% of models against the old gate's ~55%). See LEARNINGS.
# Collection-scale reality check (review U4, live-margin census over 2799
# models): 44% of live margins fall below this gate (39% of cached ones) —
# the ~20% came from a 44-49 model labelled subset, and live margins run
# lower than cached (the n_az 4->2 compression). Latent while production
# runs --pose-vlm off, but whoever enables the arbiter cold should budget
# ~1227 paid calls, not ~560.
MARGIN_THRESHOLD = 0.45

# How hard geometry's vote is attenuated when it found no print base. min-max
# maps geometry's *ratio* to its margin and is blind to magnitude, so a mesh
# with no flat base anywhere still votes confidently on evidence orders of
# magnitude under ABS_SCORE_FLOOR. Squaring is not tuning for its own sake: the
# vote has to be nearly silenced before it stops overriding SigLIP (the case
# this fixes lands at w=0.14), which is why a hard switch behaves identically.
GEO_FLOOR_POWER = 2

# Bumped when a change makes previously cached poses wrong. Entries written by
# an older version are dropped on load and re-resolved: v2 = the four-view
# ensemble and the margin gate; v3 = geometry attenuated by its base evidence;
# v4 = the arbiter's contact sheet doubled to 512 px with scaled numerals.
POSE_CACHE_VERSION = 4


@dataclass(frozen=True)
class Pose:
    """A resolved pose in flight, replacing the raw cache-entry dict
    (docs/actor-refactor/data_structures.md). The on-disk pose cache stays
    JSON dicts; `from_cache`/`to_cache` are the only crossing points.

    The freeze is shallow and `Pose` is unhashable (`front_view` is a dict):
    nothing may key on a `Pose` — `index` is the identity, everywhere."""
    up: tuple[float, float, float]
    confidence: float
    source: str                    # "forced" | "geometry" | "siglip" | "vlm"
    v: int                         # no default: from_cache carries it through,
                                   # fresh resolutions pass POSE_CACHE_VERSION
                                   # explicitly (D10)
    margin: float | None = None
    arbitrated: bool | str | None = None
                                    # None = no claim (never asked, or legacy);
                                    # False = the escalation did not happen;
                                    # True = answered; "rejected" = the API
                                    # judged the request (docs/tri-state-pass-2.md,
                                    # 2026-08-21). `str` is deliberate: a clean
                                    # string enum would force load-time mapping
                                    # of every true/false written since
                                    # 2026-08-19 for no semantic gain.
    front_view: dict[str, int] = field(default_factory=dict)   # view_cfg -> index

    @classmethod
    def from_cache(cls, d):
        """Absorb legacy *shapes*, not versions: bare-int front_view entries
        carry no record of the config that produced them and are treated as
        absent (matching `front_view` below); `margin` is absent from older
        entries. `v` is carried through, never defaulted — a default of
        POSE_CACHE_VERSION would stamp unversioned entries as freshly
        resolved and defeat `load_pose_cache`'s drop rule (D10). Source
        spellings are already mapped by `load_pose_cache` (RENAMED_SOURCES);
        this constructor takes the entry as loaded."""
        fv = d.get("front_view")
        return cls(up=tuple(float(x) for x in d["up"]),
                   confidence=float(d.get("confidence", 0.0)),
                   source=d["source"],
                   v=d.get("v", 0),
                   margin=d.get("margin"),
                   # absent on every entry written before 2026-08-19 and on
                   # every model that never escalated — see `to_cache`
                   arbitrated=d.get("arbitrated"),
                   front_view=dict(fv) if isinstance(fv, dict) else {})

    def to_cache(self):
        """Main's JSON entry shape (main:classify_stls.py:1138-1141);
        `front_view` is included only once something has been resolved,
        matching entries that predate front-view caching.

        `arbitrated` is four-state, and **absence reads as `false`**
        (docs/tri-state-pass-2.md, 2026-08-21):

        * **true** — asked and answered, whether or not the answer moved the
          pose.
        * **"rejected"** — asked, and the API judged the request on its
          merits (a non-auth 4xx, or a coherent 200 refusal — the
          safety-block shape). Never re-ask; the ensemble answer stands
          permanently.
        * **false** — the escalation the margin asked for did not happen: a
          transient failure (`VLMUnavailable` — network, 5xx, auth, CLI
          timeout), a cancellation from `arbiter.shutdown()` on an abort,
          abandonment when `settle`'s wait ran out, or the gate firing in a
          run with no arbiter. Ask when possible.
        * **absent** — no claim; read as `false`. `pose_is_sufficient` is
          what makes that precise: an absent entry whose margin clears the
          gate is never touched.

        It is deliberately *not* the same fact as `source == "vlm"`, which
        means the arbiter **moved** the answer: a call that ran and confirmed
        the ensemble keeps the ensemble's label, so without this a
        confirmation and a refusal are identical on disk.

        The string passes through rather than being coerced: `bool("rejected")
        is True`, which collapsed the schema to three states on disk while
        every in-memory test passed (review 2 blocker B1)."""
        d = {"up": [float(x) for x in self.up],
             "confidence": self.confidence,
             "source": self.source,
             "margin": self.margin,
             "v": self.v}
        if self.arbitrated is not None:
            d["arbitrated"] = (self.arbitrated if isinstance(self.arbitrated, str)
                               else bool(self.arbitrated))
        if self.front_view:
            d["front_view"] = dict(self.front_view)
        return d


def up_axis_scores(mesh, n_samples=4000):
    """Flat print-base evidence per candidate up: the fraction of sampled
    surface that is both in the bottom 2% height slab and facing down.

    Seeded, because the winner can rest on ~30 of the 4000 points and an
    unseeded draw moves picks between runs on identical input — which would
    make the pose cache irreproducible and the ensemble below unstable."""
    # The only open3d in this module, and the reason it is deferred: importing
    # it at module scope charged 2596 modules to everything that imports
    # `pose`, including the read-only tools that only want the cache functions
    # — `embed_store` -> `cluster_models` paid for a rendering library to load
    # `.npy` files. Deferring an import usually just moves the cost to the
    # caller (see `DEFAULT_MODEL`, which had to move rather than defer); it is
    # genuinely free *here* because every caller of this function passes a
    # mesh, and you cannot hold an Open3D mesh without having imported open3d.
    # The argument type already implies the import.
    import open3d as o3d

    o3d.utility.random.seed(SAMPLE_SEED)
    pcd = mesh.sample_points_uniformly(n_samples, use_triangle_normal=True)
    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    scores = []
    for up in UP_CANDIDATES:
        h = pts @ up
        extent = h.max() - h.min()
        if extent <= 0:
            scores.append(0.0)
            continue
        in_bottom_slab = h < h.min() + 0.02 * extent
        facing_down = normals @ up < -0.9
        scores.append(float(np.mean(in_bottom_slab & facing_down)))
    return np.array(scores)


def rank_up_scores(scores):
    """(index, ratio, best_score) — ratio = runner_up/best, lower is more
    confident (symmetric meshes like a barrel come out ~1.0)."""
    order = np.argsort(scores)[::-1]
    best, runner = float(scores[order[0]]), float(scores[order[1]])
    return int(order[0]), (runner / best if best > 0 else 1.0), best


def detect_up_axis(mesh, n_samples=4000):
    """Geometry-only up axis. Returns (up, ratio, best_score)."""
    idx, ratio, best = rank_up_scores(up_axis_scores(mesh, n_samples))
    return UP_CANDIDATES[idx], ratio, best


def needs_arbiter_margin(margin, threshold=MARGIN_THRESHOLD):
    """The *ensemble's* doubt: how far the winning candidate leads the runner-up
    in the combined score. Geometry having no print base says nothing about
    whether the combination is unsure — that is precisely the population SigLIP
    was added to carry, and gating on it escalated models the ensemble already
    had right (17 of 18, measured)."""
    return margin < threshold


def file_identity(f, root):
    """Pose-cache key: same identity as the embedding cache.

    The path is relative to the collection root so the library can move drives
    without discarding poses that cost a paid arbiter call to resolve — see
    identity.py. mtime and size stay, because an edited file is a different
    model; the mtime is truncated to whole seconds so that a change of
    filesystem cannot move it out from under the key."""
    stat = f.stat()
    return f"{identity.rel_path(f, root)}|{identity.mtime_key(stat)}|{stat.st_size}"


def load_pose_cache(cache_dir):
    """Cached poses, minus any written by an older POSE_CACHE_VERSION.

    Stale entries are dropped rather than migrated: a pose decided under a
    different ensemble or a different escalation gate is not the pose this
    version would produce, and silently trusting it would mean the collection
    disagrees with every measurement made against it."""
    if not cache_dir:
        return {}
    p = Path(cache_dir) / "pose-cache.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    # Shape before content: the file is hand-editable, and a list-shaped or
    # null top level raised AttributeError from `raw.items()` — which no
    # caller's except clause named, so it escaped `POST /reload` as a bare
    # 500 (review, 2026-08-20). ValueError is what Collection.load already
    # converts to CacheUnusable. Non-dict *entries* are dropped like stale
    # versions: `v.get` is the next line to crash on them.
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: pose cache must be a JSON object, "
                         f"got {type(raw).__name__}")
    fresh = {k: v for k, v in raw.items()
             if isinstance(v, dict) and v.get("v") == POSE_CACHE_VERSION}
    if len(fresh) < len(raw):
        print(f"pose cache: {len(raw) - len(fresh)} of {len(raw)} entries predate "
              f"v{POSE_CACHE_VERSION} and will be re-resolved")
    for v in fresh.values():
        # The 2026-08-14 rename (review P2.3-A): same poses, honest names —
        # "geometry" = geometry's pick stood, "siglip" = SigLIP moved it off
        # that pick. Mapped on load rather than behind a version bump: the
        # poses themselves are unchanged, and a bump would re-resolve (and
        # re-bill) every entry for a spelling.
        if v.get("source") in RENAMED_SOURCES:
            v["source"] = RENAMED_SOURCES[v["source"]]
    return fresh


RENAMED_SOURCES = {"heuristic": "geometry", "ensemble": "siglip"}


def save_pose_cache(cache_dir, cache):
    if not cache_dir:
        return
    p = Path(cache_dir) / "pose-cache.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache))


def pose_is_sufficient(entry, arbiter_available, margin_threshold):
    """Is this cached pose good enough for the current run, or a miss?

    `margin` is None exactly when the SigLIP ensemble did not run — a
    geometry-only pass, written by some older `--no-up-ensemble` run. Treating
    those as hits would let one such pass pin every model to its geometry
    answer, and the ensemble — and the margin gate behind it, so the arbiter
    too — would never run again; so they read as misses and are upgraded in
    place. A VLM answer outranks the ensemble whichever gate escalated it, so
    it stands regardless of its margin.

    The four states of `arbitrated`, read against *this* run
    (docs/tri-state-pass-2.md, 2026-08-21):

    * `true` / `"rejected"` — settled. The call happened and either answered
      or was judged; a retry buys nothing.
    * `false` or **absent** — the escalation the margin asked for has not
      happened. Absence reads as `false`, and the two conditions below are
      what keep that affordable.

    Both new parameters take **no default** (the `Resolved.pose_changed`
    precedent): a default silently un-pins the W1 regression test, and every
    caller breaking loudly is the point.

    `arbiter_available` kills the laundering bug: a run with no arbiter
    (explicit `off`, a degraded `auto`, or a tripped breaker) must not
    re-render a marked entry it cannot escalate — it would re-resolve the pose
    with no gate and erase the marker, and production runs `off`.
    `margin_threshold` is this run's gate: an entry whose margin clears it is
    not owed a call at all, so a threshold change cannot launder markers
    either."""
    if entry is None:
        return False
    if entry["source"] == "vlm":
        return True
    if entry.get("arbitrated") in (True, "rejected"):
        return entry.get("margin") is not None
    if entry.get("margin") is None:
        return False
    # A miss for a *later* run only: the driver's re-route of a just-folded
    # answer passes `settled=True` to `route`, or a rate-limited call would
    # re-render and re-bill in a loop within the run that just failed to
    # arbitrate it (review, 2026-08-20).
    return not (arbiter_available
                and needs_arbiter_margin(entry["margin"], margin_threshold))


def up_str(up):
    return ",".join(f"{float(v):g}" for v in up)


def entry_up(entry):
    """The up vector of a cache entry as a 3-tuple, or None if it has none.

    The one validator, because two callers need the same answer and both used
    to assume the entry was well-formed: `embed_cache_token` below (so a
    malformed entry cannot crash a load) and `collection.pose_of` (so it
    cannot fail a whole query response). `load_pose_cache` filters on `v` and
    checks no shape at all, and pose-cache.json is hand-editable — the two
    that turn up are a missing `up` and a null one.

    Finite and non-zero as well as three floats: `rotation_to_z_up` raises on
    a zero or NaN vector (json.loads accepts a bare `NaN` literal), and both
    callers exist so that a malformed entry degrades to "no pose" instead of
    failing a load or a whole query response (review, 2026-08-20)."""
    if not entry:
        return None
    try:
        up = tuple(float(x) for x in entry["up"])
    except (KeyError, TypeError, ValueError):
        return None
    if len(up) != 3 or not all(np.isfinite(v) for v in up) or not any(up):
        return None
    return up


FORCED_UPS = {"z": (0.0, 0.0, 1.0), "y": (0.0, 1.0, 0.0)}


def embed_cache_token(entry, up_axis_arg="auto"):
    """The render identity of a pose: only `up` changes the pixels, so the
    token is the up vector and nothing else (review P2.3-B).

    `source` is not identity — it was only ever a proxy for determinism, and
    the elision it used to drive ("deterministic poses keep the legacy
    --up-axis string") filed identical pixels under two keys whenever a
    forced axis and a geometry answer agreed on the same up. A forced
    --up-axis needs no pose entry; its up is the flag. Caches keyed under
    the old tokens are re-keyed by migrate_cache_keys.py — cache-meta.json
    records which scheme a cache uses."""
    up = entry_up(entry)        # None for a malformed entry, which carries no
    if up:                      # pose — the same thing as having none at all
        return up_str(up)
    if up_axis_arg in FORCED_UPS:
        return up_str(FORCED_UPS[up_axis_arg])
    return "unresolved"     # no pose yet — nothing is cached under any key


FRONT_PROMPTS = [
    "the front of a miniature figurine, facing the camera",
    "a miniature figurine seen from the front, face and chest visible",
]
BACK_PROMPTS = [
    "the back of a miniature figurine, facing away from the camera",
    "a miniature figurine seen from behind, back of the head visible",
]


# The exact rotations for the six `UP_CANDIDATES`, transcribed float-for-float
# from the Open3D construction that used to compute them
# (`get_rotation_matrix_from_xyz((pi,0,0))` for the antiparallel case, Rodrigues
# about `up x z` otherwise). The near-zero entries are cos(pi/2) and sin(pi)
# noise and are kept rather than cleaned to 0: this table exists to be
# *bit-identical* to what the renderer drew every cached embedding with, and
# tidying it would be a change to the render recipe (OPEN_QUESTIONS) for
# cosmetic reasons. `tests/test_pose.py` asserts the equality against Open3D
# itself, so the transcription cannot drift.
_AXIS_ROTATIONS = {
    (0.0, 0.0, 1.0):  [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    (0.0, 0.0, -1.0): [[1.0, 0.0, 0.0], [0.0, -1.0, -1.2246467991473532e-16],
                       [0.0, 1.2246467991473532e-16, -1.0]],
    (0.0, 1.0, 0.0):  [[1.0, 0.0, 0.0], [0.0, 6.123233995736766e-17, -1.0],
                       [0.0, 1.0, 6.123233995736766e-17]],
    (0.0, -1.0, 0.0): [[1.0, -0.0, 0.0], [0.0, 6.123233995736766e-17, 1.0],
                       [-0.0, -1.0, 6.123233995736766e-17]],
    (1.0, 0.0, 0.0):  [[6.123233995736766e-17, -0.0, -1.0], [0.0, 1.0, -0.0],
                       [1.0, 0.0, 6.123233995736766e-17]],
    (-1.0, 0.0, 0.0): [[6.123233995736766e-17, 0.0, 1.0], [0.0, 1.0, -0.0],
                       [-1.0, 0.0, 6.123233995736766e-17]],
}


def rotation_to_z_up(up):
    """The rotation taking `up` to +Z. Pure numpy; it never touches a mesh.

    Lived in `renderer` and computed with Open3D until 2026-08-19, which meant
    a rotation matrix could not be named without a rendering library — the API
    needs one to publish `pose.azimuth_zero` (docs/api/surface.md).

    Two paths, and the split is the point. Every `up` this project resolves is
    one of `UP_CANDIDATES`, so those six are served from a table of the exact
    matrices Open3D produced, byte for byte: this function decides the pixels
    of every non-`+Z` render (`Renderer.views` rotates by it), the embedding
    key records only the up *vector*, and so a value that differed even in the
    last bit would re-pose cached models under unchanged keys. A naive
    Rodrigues rewrite is *not* bit-identical here — it differs by ~5e-17 on
    four of the six, in the entries that ought to be zero.

    Anything else falls through to Rodrigues, which agrees with Open3D to
    ~1e-15. Nothing in the pipeline reaches it, since poses are always
    axis-aligned.

    **The input is normalised first, and that is load-bearing rather than
    tidiness.** A vector collinear with Z but not unit — `[0,0,2]`, or `[0,0,1]`
    off by a rounding step — has a zero cross product with Z, so Rodrigues
    divides by zero and yields an all-NaN matrix. Open3D's version hid this: it
    took `nan` axis-angle input and returned the *identity*, so `[0,0,-2]`
    silently rendered upside down rather than failing. Normalising sends every
    such vector to the table, where the answer is right. Bit-fidelity survives
    it because the six candidates are exactly unit, and `x / 1.0 == x`.

    A zero vector raises: there is no rotation taking nothing to +Z, and a
    caller that has one is holding a bug, not an edge case."""
    v = np.asarray(up, dtype=float)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        raise ValueError(f"up must be a finite non-zero vector, got {up!r}")
    v = v / n
    exact = _AXIS_ROTATIONS.get(tuple(float(x) for x in v))
    if exact is not None:
        return np.array(exact)          # a fresh array per call, as before
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(v, z)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(v @ z, -1, 1))
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def view_angles(n_views, elevations):
    """(azimuth, elevation) radian pairs: a full turntable ring per elevation.

    Elevation-major, so views 0..n_views-1 are the first ring — a run with one
    elevation lays out exactly as it did before elevations existed, and
    view0.png keeps meaning the same camera.

    Lives here rather than in `renderer` (moved 2026-08-19) because it is pure
    numpy and it is what turns a `front_view` index into an angle: the API
    names a model's front camera without loading a rendering library
    (docs/api/surface.md §pose). `renderer` imports it back — it is still the
    function `Renderer.views` and `pose_tiles` take their cameras from, and
    there is one copy."""
    return [(2 * np.pi * i / n_views, np.deg2rad(e))
            for e in elevations for i in range(n_views)]


def front_view(entry, cfg):
    """The cached front ("hero") view index for one view configuration.

    front_view indexes into a specific run's view list, so it is stored as a
    dict keyed by `cachedir.view_config` — an index cached at 8 views is
    out of range at 4 and silently wrong under different elevations. Legacy
    integer entries carry no record of the config that produced them and are
    treated as absent; a warm classify pass regenerates them from cached
    embeddings, and consumers fall back to view 0 (a real render, just not
    necessarily the front) until it does."""
    fv = (entry or {}).get("front_view")
    return fv.get(cfg) if isinstance(fv, dict) else None


def front_view_index(view_embeds, front_embeds, back_embeds):
    """Index of the view that best faces the camera. Front is metadata: this
    never triggers a re-render, it just names one of the existing views."""
    score = ((view_embeds @ front_embeds.T).mean(1)
             - (view_embeds @ back_embeds.T).mean(1))
    return int(np.argmax(score))


# Deliberately not phrased in terms of heads and feet: roughly half the
# collection is terrain and scatter, where anatomy probes score 0/12.
UPRIGHT_PROMPTS = [
    "a 3D printed model sitting the right way up on a table",
]
TOPPLED_PROMPTS = [
    "a 3D printed model tipped onto its side",
    "a 3D printed model turned upside down",
]


def upright_scores(tile_embeds, upright_embeds, toppled_embeds):
    """Per-candidate upright score for the up-candidate tiles. The toppled
    probes are not optional padding: without something to contrast against,
    raw similarity is near-flat across the six tiles and the argmax is noise."""
    return ((tile_embeds @ upright_embeds.T).mean(1)
            - (tile_embeds @ toppled_embeds.T).mean(1))


def _unit(v):
    """Min-max to [0,1]; all-equal input collapses to zeros (no vote)."""
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def combine_up_scores(geo_scores, siglip_scores):
    """Index of the up candidate favoured by geometry and SigLIP together.

    The two are in different units — a flat-base area fraction against a
    difference of cosine similarities — so each is min-maxed before the mean.
    That is not merely a scale fix: geometry's weakest candidate is almost
    always 0, so min-max maps its runner-up to runner/best, exactly the ratio
    rank_up_scores reports. Geometry therefore votes with a ~1.0 margin when it
    has real base evidence and with a ~0.0 margin when it is guessing, handing
    those models to SigLIP without either method being thresholded."""
    return combine_up(geo_scores, siglip_scores)[0]


def geo_weight(geo_scores):
    """How loudly geometry gets to vote, from how much base evidence it has.

    1.0 once a real print base is found, falling to ~0 when the best score is
    far under ABS_SCORE_FLOOR. This is *not* the absolute-scaling scheme that
    lost at 20/23 — that one replaced min-max with clip(geo/floor), saturating
    every candidate above the floor and destroying the margin. Here min-max is
    untouched inside the vote; only its amplitude changes."""
    best = float(np.max(geo_scores))
    return min(1.0, best / ABS_SCORE_FLOOR) ** GEO_FLOOR_POWER


def combine_up(geo_scores, siglip_scores):
    """(index, margin) — the ensemble's pick and how far it leads the runner-up.

    The margin is the gate's input, and it predicts error far better than
    geometry's confidence does: measured over 44 labelled models its median is
    1.31 where the ensemble is right and 0.22 where it is wrong.

    The margin is deliberately *not* rescaled to undo the geometry weight. A
    quieter geometry vote does compress the margin, and that is correct — less
    evidence should mean less certainty. Normalising it back looks tidier and
    measurably stops escalating a model the ensemble gets wrong."""
    combined = geo_weight(geo_scores) * _unit(geo_scores) + _unit(siglip_scores)
    top = np.sort(combined)[::-1]
    return int(np.argmax(combined)), float(top[0] - top[1])


OLLAMA_URL = "http://localhost:11434"

UP_PROMPT = (
    "Each numbered tile shows the same 3D model in a different orientation. "
    "Which tile shows the model standing upright, the way it would sit on a "
    'table? Answer with JSON only: {"tile": <number>}'
)


SHEET_THUMB = 512


def sheet_font(thumb):
    """Tile numerals scaled to the tile.

    PIL's default face is a ~11 px bitmap: legible on a 768x512 sheet and
    proportionally invisible on a 1536x1024 one. A model cannot answer
    {"tile": n} about numerals it cannot read, so raising thumb without this
    measures *worse* than not raising it at all — the trap that nearly got the
    512 px result discarded."""
    size = max(11, thumb * 44 // 512)          # 44px at 512, 22px at 256
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()            # bitmap, fixed ~11px


def make_contact_sheet(tiles, thumb=SHEET_THUMB, cols=3):
    """Grid of tiles labeled 1..n (red corner numbers) for the VLM prompt.

    512 rather than 256: sonnet gains 10 of 44 across that step and gemma 3,
    for one resize of already-rendered tiles. It is not a uniform win —
    gemini-3.5-flash returns the same answer on 42 of 44 models either way —
    so the size belongs in any report of a VLM number. See LEARNINGS."""
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    draw = ImageDraw.Draw(sheet)
    font = sheet_font(thumb)
    for i, im in enumerate(tiles):
        im = im.copy()
        im.thumbnail((thumb, thumb))
        x, y = (i % cols) * thumb, (i // cols) * thumb
        sheet.paste(im, (x, y))
        draw.text((x + thumb // 36, y + thumb // 64), str(i + 1), fill="red", font=font)
    return sheet


def parse_tile_answer(text, n_tiles):
    """Extract {"tile": n} from a model reply. Returns 0-based index or None."""
    try:
        tile = json.loads(text[text.index("{"):text.rindex("}") + 1])["tile"]
    except (ValueError, KeyError, TypeError):
        return None
    if isinstance(tile, int) and 1 <= tile <= n_tiles:
        return tile - 1
    return None


def ollama_available():
    import requests
    try:
        requests.get(f"{OLLAMA_URL}/api/version", timeout=2)
        return True
    except requests.RequestException:
        return False


def _ask_ollama(png_bytes, n_tiles, model):
    import requests
    payload = {
        "model": model,
        "stream": False,
        # thinking models put every token in the thinking field and return an
        # empty content (measured: 3859 tokens / 281 s vs 7 tokens / 25 s)
        "think": False,
        "format": {"type": "object", "properties": {"tile": {"type": "integer"}},
                   "required": ["tile"]},
        "messages": [{"role": "user", "content": UP_PROMPT,
                      "images": [base64.b64encode(png_bytes).decode()]}],
    }
    resp = requests.post(f"{OLLAMA_URL}/api/chat", timeout=300, json=payload)
    if resp.status_code == 400 and "think" in resp.text:
        del payload["think"]  # older servers/models reject the field entirely
        resp = requests.post(f"{OLLAMA_URL}/api/chat", timeout=300, json=payload)
    resp.raise_for_status()
    return parse_tile_answer(resp.json()["message"]["content"], n_tiles)


def _ask_claude(sheet_path, n_tiles):
    # Failures raise VLMUnavailable rather than flattening to None: the CLI
    # never judged the request, so its timeouts, non-zero exits and a missing
    # binary are all the retry split's transient side — flattened, they were
    # indistinguishable from an unparseable answer, and a raised TimeoutExpired
    # took `_fold`'s permanent branch (review, 2026-08-20).
    try:
        out = subprocess.run(
            ["claude", "-p", f"Read the image at {sheet_path}. {UP_PROMPT}",
             "--output-format", "json", "--max-turns", "3"],
            capture_output=True, text=True, timeout=180)
    except (subprocess.TimeoutExpired, OSError) as e:
        raise VLMUnavailable(f"claude CLI: {e}") from e
    if out.returncode != 0:
        raise VLMUnavailable(
            f"claude CLI exited {out.returncode}: {out.stderr.strip()[:200]}")
    return parse_tile_answer(json.loads(out.stdout).get("result", ""), n_tiles)


GEMINI_HOST = "aiplatform.googleapis.com"
GEMINI_LOCATION = "global"
GEMINI_MODEL = "gemini-3.5-flash"
# {"tile": n} is forced by the response schema rather than asked for in prose,
# the same contract the ollama backend uses.
_GEMINI_SCHEMA = {"type": "object", "properties": {"tile": {"type": "integer"}},
                  "required": ["tile"]}
_token_cache = {}


def _run_gcloud(argv, timeout):
    """`gcloud`, with its process-level failures normalised to the RuntimeError
    the two helpers below already promise (docs/tri-state-pass-2.md,
    2026-08-21). A hung or missing binary raised `TimeoutExpired`/`OSError`
    out of helpers whose callers catch `RuntimeError`, and inside `_ask_gemini`
    that leak took `_fold`'s permanent arm — the mid-run ADC-expiry case, since
    the token cache's TTL guarantees a collection run re-mints."""
    try:
        return subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout)
    except (subprocess.SubprocessError, OSError) as e:
        raise RuntimeError(f"could not run gcloud: {e}") from e


def gcloud_project():
    """The GCP project for the Gemini backend, from the environment or gcloud."""
    for var in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"):
        if os.environ.get(var):
            return os.environ[var]
    out = _run_gcloud(["gcloud", "config", "get-value", "project"], 30)
    project = out.stdout.strip()
    if out.returncode != 0 or not project or project == "(unset)":
        raise RuntimeError("no GCP project — set GOOGLE_CLOUD_PROJECT or run "
                           "`gcloud config set project <id>`")
    return project


def gcloud_token(ttl=1800):
    """Cached ADC access token. Minting one costs ~0.5 s, which is real against
    a ~1 s arbiter call, and the token outlives a whole collection run."""
    now = time.monotonic()
    if _token_cache.get("expires", 0) > now:
        return _token_cache["token"]
    out = _run_gcloud(["gcloud", "auth", "application-default",
                       "print-access-token"], 60)
    if out.returncode != 0:
        raise RuntimeError("gcloud ADC unavailable — run "
                           f"`gcloud auth application-default login` ({out.stderr.strip()})")
    _token_cache.update(token=out.stdout.strip(), expires=now + ttl)
    return _token_cache["token"]


class VLMUnavailable(RuntimeError):
    """The call failed without the API ever judging the request: a network
    drop, a socket timeout, an HTTP 5xx, a CLI that timed out or would not
    start.

    Its own type because it is the retry rule's transient side: `_fold`
    records these `arbitrated=False` (ask again on a later run), where a
    request the API rejects on its merits leaves the key absent (a retry
    cannot succeed and would pay a call per run forever). The split used to
    be RateLimited-vs-everything, which put the most common transient
    failures — a mid-run network blip, a 502 — on the permanent side
    (review, 2026-08-20)."""


class VLMRejected(RuntimeError):
    """The API judged the request; a retry cannot succeed.

    The permanent side of the retry split, and the side that is **enumerated**
    (docs/tri-state-pass-2.md, 2026-08-21): a judged verdict can only arrive
    as a non-auth 4xx or a coherent 200 refusal, where the transient side is
    open-ended. Three review passes found transient failures pinned as
    permanent while permanent was the fallthrough — so `_fold`'s default is
    now `False` and this type is what buys `"rejected"`."""


class RateLimited(VLMUnavailable):
    """The VLM said "later" (HTTP 429/503), not "no".

    Its own subtype because the handling differs again: any `VLMUnavailable`
    is worth re-asking next run, but a quota refusal should also *wait*
    before this call's own retry, since the next call from the same pool
    will hit the same limit."""


# Waits before each retry after a rate-limit refusal. Two attempts, so one
# wait; the list documents the shape for when a third is wanted. Small on
# purpose — the Arbiter's `min_interval` is what paces the *pool*, and this
# only stops a single call's retry from being instantaneous.
VLM_BACKOFF = [5.0, 20.0]


TRANSIENT_HTTP_STATUS = (401, 403, 404, 408, 409, 425)
"""4xx statuses that are the environment, not a verdict on the request.

401/403/404 are auth, entitlement and a wrong endpoint — every model in such
a run fails identically, and a permanent record is a wrong pin the next run
cannot clear. 408/409/425 are a request timeout, a conflict and a
too-early retry from an intermediary: transient by definition, and mapping
them permanent is the leak this pass exists to end (review 2)."""

REJECTED_FINISH_REASONS = ("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT",
                           "RECITATION", "SPII", "IMAGE_SAFETY")
"""`finishReason`s that say the API judged the request. Enumerated, like the
rest of the permanent side: `MAX_TOKENS`, `OTHER` and a missing reason are
transient, because a body that merely carries no answer is not a verdict."""


def _ask_gemini(png_bytes, n_tiles, model, project=None):
    """Vertex AI arbiter. Raw HTTPS rather than an SDK: one POST, no dependency,
    and the same call the eval harness measured at 43/44 standalone."""
    import http.client
    import urllib.error
    import urllib.request

    # The environment being broken is not the request being judged: a missing
    # project or an expired ADC token is transient, and the token cache's
    # 1800 s TTL guarantees a collection run re-mints mid-run, where the
    # startup probe cannot see it (docs/tri-state-pass-2.md, 2026-08-21).
    # `gcloud_project()` is unreachable from the pipeline — resolve_pose_vlm
    # always populates args.gemini_project — so the live path is the token
    # half; both are wrapped anyway.
    try:
        project = project or gcloud_project()
        token = gcloud_token()
    except RuntimeError as e:
        raise VLMUnavailable(f"gcloud: {e}") from e
    url = (f"https://{GEMINI_HOST}/v1/projects/{project}/locations/{GEMINI_LOCATION}"
           f"/publishers/google/models/{model}:generateContent")
    body = json.dumps({
        "contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": "image/png",
                            "data": base64.b64encode(png_bytes).decode()}},
            {"text": UP_PROMPT}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                             "responseSchema": _GEMINI_SCHEMA},
    }).encode()
    req = urllib.request.Request(url, body, {"Authorization": f"Bearer {token}",
                                             "Content-Type": "application/json"})
    # The `HTTPError` clause stays ABOVE the `OSError` one — HTTPError
    # subclasses OSError — while `.read()`'s IncompleteRead is an
    # HTTPException and lands transient below.
    try:
        raw = urllib.request.urlopen(req, timeout=300).read()
    except urllib.error.HTTPError as e:
        detail = f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"
        # 429/503 are "come back later", not "this request is wrong", and they
        # return in milliseconds — so an immediate retry is a second failure
        # and a freed worker starts a third. Distinguished so `ask_vlm_up` can
        # back off instead (2026-08-19: a --rescan at collection scale hit
        # Vertex quota and the un-paced pool turned it into a storm).
        if e.code in (429, 503):
            raise RateLimited(detail) from e
        # Auth/entitlement and intermediary-timeout statuses are the
        # environment, not a verdict on the request, and both are discovered
        # only mid-run — the startup probe never makes a Vertex call
        # (docs/tri-state-pass-2.md, 2026-08-21). Any 5xx is the server
        # failing, transient like a network drop, without 429/503's backoff.
        if e.code in TRANSIENT_HTTP_STATUS or e.code >= 500:
            raise VLMUnavailable(detail) from e
        raise VLMRejected(detail) from e
    except (OSError, http.client.HTTPException) as e:
        # URLError and socket timeouts are OSErrors; a mid-read protocol error
        # is HTTPException. None of them is the API saying "no", so none may
        # land on the permanent side of the retry split (review, 2026-08-20).
        raise VLMUnavailable(f"network failure: {e}") from e
    # Read split from parse, and `"rejected"` inferred from the API's stated
    # verdict rather than from a KeyError (review 2 blocker B2): a
    # `finishReason: MAX_TOKENS` with no parts — the thinking-token exhaustion
    # measured twice in this repo — is deterministic per model *config* and
    # would otherwise pin the whole collection permanently.
    try:
        d = json.loads(raw)
    except ValueError as e:
        raise VLMUnavailable(f"unparseable 200 body: {raw[:200]!r}") from e
    cand = (d.get("candidates") or [{}])[0]
    feedback = d.get("promptFeedback") or {}
    reason = cand.get("finishReason") or feedback.get("blockReason")
    parts = (cand.get("content") or {}).get("parts")
    if not parts:
        if reason in REJECTED_FINISH_REASONS or "blockReason" in feedback:
            raise VLMRejected(f"blocked ({reason}): {raw[:200]!r}")
        raise VLMUnavailable(
            f"200 with no answer (finishReason={reason!r}): {raw[:200]!r}")
    return parse_tile_answer("".join(p.get("text", "") for p in parts), n_tiles)


DEFAULT_VLM_MODELS = {"ollama": "gemma4:26b", "gemini": GEMINI_MODEL, "claude": None}


def ask_vlm_up(tiles, backend, scratch_dir, vlm_model="gemma4:26b", save_to=None,
               project=None, sleep=time.sleep, raise_failures=False):
    """Ask the VLM which candidate orientation is upright. One retry on a
    bad/failed answer, then None — the caller keeps the geometry guess.
    The pipeline never hard-fails because of the VLM.

    A rate-limit refusal waits before the retry (`VLM_BACKOFF`); anything else
    retries at once, as before. `sleep` is an injection seam for tests.

    `raise_failures` re-raises **the last attempt's exception**, whatever its
    type, instead of flattening every failure to None. The pipeline passes it
    because the failures deserve different records, and the *type* is the only
    thing that distinguishes them by the time the answer is folded:

    * `VLMUnavailable` (including `RateLimited`, which additionally backs off
      here) — the API never judged the request: worth asking again on a
      later run;
    * `VLMRejected` — the API judged the request, which cannot succeed on a
      retry and would pay a call per run forever. It gets no arm of its own
      here: the generic arm below already retries once and re-raises under
      `raise_failures`, and a judged rejection rarely differs on attempt 2
      (docs/tri-state-pass-2.md, 2026-08-21);
    * anything else — an unknown failure, which `_fold` records retryable
      since three passes found transient failures on the permanent side.

    An earlier version raised only `RateLimited`, which left every hard
    failure returning None and therefore indistinguishable from an answer of
    None — so the branch meant to catch it was unreachable from production and
    a 400 re-escalated forever anyway (review, 2026-08-19).

    Note what still returns None with no exception: **an answer that would not
    parse, twice**. That is deliberate rather than left over — a VLM that
    garbled its output may well answer next run — but it does mean the parse
    failure is recorded as retryable.

    Default False so `eval/gemini_sheet_fill.py` — which maps this over a
    thread pool, where a raise loses every result in the sweep — keeps
    today's behaviour.

    save_to keeps a per-model copy of the sheet next to the saved renders. It
    is the same image the VLM was shown, written whether or not the answer
    parses, so a wrong pose can be read back off disk."""
    try:
        sheet = make_contact_sheet(tiles)
    except Exception as e:
        print(f"  pose VLM error ({backend}): {e}")
        return None
    if save_to is not None:
        try:
            save_to = Path(save_to)
            save_to.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(save_to)
        except OSError as e:  # a debug artifact must never fail the run
            print(f"  could not save pose sheet {save_to}: {e}")
    backoff = 0.0                      # local: this runs on 4+ arbiter threads
    last_error = None                  # the LAST attempt's, not any attempt's
    for attempt in range(2):
        if attempt and backoff:
            sleep(backoff)
        # cleared per attempt, or an attempt that *returns* an unparseable
        # answer leaves the previous attempt's exception standing and that one
        # decides the record: "400 then unparseable" raised the stale 400 and
        # cached the model once, where the last attempt did not raise at all
        # and an unparseable answer is retryable (review, 2026-08-19)
        last_error = None
        try:
            if backend == "ollama":
                buf = io.BytesIO()
                sheet.save(buf, format="PNG")
                idx = _ask_ollama(buf.getvalue(), len(tiles), vlm_model)
            elif backend == "gemini":
                buf = io.BytesIO()
                sheet.save(buf, format="PNG")
                idx = _ask_gemini(buf.getvalue(), len(tiles), vlm_model, project)
            else:  # claude
                # A unique name per call: the arbiter pool runs 4+ of these
                # concurrently against one scratch_dir, and a shared filename
                # let worker B's save land between A's save and A's subprocess
                # read — A's model silently judged on B's renders, stamped
                # `source: vlm` and never revisited (review, 2026-08-20).
                # gemini/ollama hand bytes over in memory and never had the
                # window.
                fd, name = tempfile.mkstemp(prefix="pose-sheet-", suffix=".png",
                                            dir=scratch_dir)
                os.close(fd)
                sheet_path = Path(name)
                try:
                    sheet.save(sheet_path)
                    idx = _ask_claude(sheet_path, len(tiles))
                finally:
                    sheet_path.unlink(missing_ok=True)
        except RateLimited as e:
            backoff = VLM_BACKOFF[min(attempt, len(VLM_BACKOFF) - 1)]
            print(f"  pose VLM rate-limited ({backend}), waiting {backoff:g}s: {e}")
            last_error, idx = e, None
        except VLMUnavailable as e:
            # transient but not a quota signal: retry at once, and carry the
            # type so a last-attempt failure is recorded as retryable
            print(f"  pose VLM unavailable ({backend}): {e}")
            last_error, idx = e, None
        except Exception as e:
            print(f"  pose VLM error ({backend}): {e}")
            last_error, idx = e, None
        if idx is not None:
            return idx
    if last_error is not None and raise_failures:
        raise last_error
    return None
