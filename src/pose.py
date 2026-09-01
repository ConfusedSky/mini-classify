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
import threading
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
                                    # judged the request (docs/archive/tri-state-pass-2.md,
                                    # 2026-08-21). `str` is deliberate: a clean
                                    # string enum would force load-time mapping
                                    # of every true/false written since
                                    # 2026-08-19 for no semantic gain.
    arbiter: str | None = None      # which judge answered: "backend/model"
                                    # (`arbiter_id`). Only ever compared as a
                                    # whole string, so the model half being an
                                    # OpenRouter org/model id — a second "/" —
                                    # needs no escaping and no parsing.
    front_view: dict[str, int] = field(default_factory=dict)   # view_cfg -> index

    @classmethod
    def from_cache(cls, d):
        """A plain constructor over an entry `load_pose_cache` returned.

        It absorbs no legacy shapes — they were deleted with the 2026-08-31
        rebuild (docs/cache-rebuild.md §3). What makes reading `up`, `source`,
        `v`, `margin` and `front_view` straight out of the dict safe is one
        thing and not five: `_readable`, the loader's predicate, requires
        every one of them, because it is derived from these accesses. In
        particular `margin` is required to be **present** and may be `None` —
        the sufficiency check is not what guarantees it, and never was: a
        `source == "vlm"` entry is a hit one line before sufficiency looks at
        `margin` at all, so a vlm entry with the key missing used to pass
        sufficiency and raise `KeyError` here, once per run, forever
        (adversarial review pass 3, 2026-08-31).

        `arbitrated` absent is not a legacy shape but a live state: no claim,
        read as `false` (see `to_cache`). `arbiter` is `.get` for the same
        reason and one more: it was introduced without a version bump
        (2026-08-31), so an entry written before it is at the *current*
        version and absent — legal, not a shape the rebuild deleted. The
        plain-constructor doctrine above forbids absorbing dead shapes, not
        reading a field the live schema makes optional."""
        return cls(up=tuple(float(x) for x in d["up"]),
                   confidence=float(d.get("confidence", 0.0)),
                   source=d["source"],
                   v=d["v"],
                   margin=d["margin"],
                   arbitrated=d.get("arbitrated"),
                   arbiter=d.get("arbiter"),
                   front_view=dict(d.get("front_view", {})))

    def to_cache(self):
        """The pose cache's JSON entry shape.

        `front_view` is included only once something has been resolved,
        matching entries that predate front-view caching.

        `arbitrated` is four-state, and **absence reads as `false`**
        (docs/archive/tri-state-pass-2.md, 2026-08-21):

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
        every in-memory test passed (review 2 blocker B1).

        `arbiter` — "backend/model", `arbiter_id` — rides along with the two
        settled states and **only** those: it is the record of who judged, and
        `false`/absent carry no judgment to attribute. Written through with no
        `POSE_CACHE_VERSION` bump, the way `arbitrated` itself was introduced
        (docs/archive/tri-state-pass-2.md): older readers ignore the key, and a
        bump would instead wipe every one of the four states it annotates.
        `--repose` is its one reader — see `pose_is_sufficient`."""
        d = {"up": [float(x) for x in self.up],
             "confidence": self.confidence,
             "source": self.source,
             "margin": self.margin,
             "v": self.v}
        if self.arbitrated is not None:
            d["arbitrated"] = (self.arbitrated if isinstance(self.arbitrated, str)
                               else bool(self.arbitrated))
        if self.arbiter and self.arbitrated in (True, "rejected"):
            d["arbiter"] = self.arbiter
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


def _number(x):
    """A JSON number — and not the `bool` that `isinstance(x, int)` admits.

    `_readable`'s numeric clauses exist to stop a crash, and `True` crashes
    nothing: it would be read as a confidence or a margin of 1.0, and a
    silently wrong number outlives a dropped entry."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _readable(entry):
    """Can the contract process this entry? The loader admits exactly what it
    can, and drops the rest — the principle 9d39745 and c3adf6c each applied
    to one shape at a time, and adversarial review pass 3 (2026-08-31) turned
    into one predicate after finding four more admitted shapes that crashed
    downstream.

    Every clause below is an **access** `Pose.from_cache` or
    `pose_is_sufficient` actually makes, not a schema wish. That is the whole
    derivation, and it is why the clauses may not be weakened one at a time:

    * a `dict` — `pose_is_sufficient`'s `entry["source"]` and every `.get`
      call in both functions;
    * `v == POSE_CACHE_VERSION` — `from_cache` reads `d["v"]` with no
      default, and a pose decided under a different ensemble or escalation
      gate is not the pose this version would produce anyway (the original
      reason to drop, `load_pose_cache` below);
    * `source` a `str` — `pose_is_sufficient` subscripts `entry["source"]`
      before any `.get`, so an entry without one raises inside *sufficiency*,
      not in `from_cache`: the check meant to protect the constructor is the
      thing that crashes;
    * `up` a usable vector, through `entry_up` so the rule has one home — it
      is `tuple(float(x) for x in d["up"])` in `from_cache`, which raises on
      a missing or null `up`, and `entry_up` additionally rejects the zero
      and NaN vectors `rotation_to_z_up` raises on downstream;
    * `confidence` absent or a real number — `from_cache` reads
      `float(d.get("confidence", 0.0))`, which raises on a null
      (`TypeError`) and on a non-numeric string (`ValueError`). Absent is
      legal because the default *is* the access;
    * the `margin` **key present**, its value `None` or a real number —
      `from_cache` reads `d["margin"]` with no default, and a non-null margin
      goes on to `needs_arbiter_margin`, whose `margin < threshold` raises
      `TypeError` on a string: that one crashes inside *sufficiency*, so once
      again the check meant to protect the constructor is the thing that
      crashes. `None` has to survive: it is the geometry-only pass and C3's
      `arbitrated: false` marker, and the ensemble upgrades both in place;
    * `arbitrated` absent, `None`, or exactly `True`, `False` or
      `"rejected"` — the one clause that is not a crash but an ambiguity. A
      hand-edited JSON `1` is `== True`, so `pose_is_sufficient`'s
      `entry.get("arbitrated") in (True, "rejected")` reads it as a judgment
      while `Done.record_pose`'s refusals, which test `was is True`, do not:
      the same entry lands in different cells of that truth table depending
      on which reader is looking at it. Dropping it at the door is what keeps
      the table's rows exhaustive over what the loader admits (pass 4);
    * `front_view` absent or a dict — `dict(d.get("front_view", {}))` raises
      on the legacy bare int, which is stamped v4 and so clears the version
      test (9d39745);
    * and **not** the sufficiency/guard deadlock (c3adf6c): a judgment
      (`arbitrated` true or `"rejected"`) on a non-`vlm` source with no
      margin. `pose_is_sufficient`'s `arbitrated` branch answers
      `margin is not None`, so such an entry is insufficient in *every* run
      and `route` re-poses and re-renders it in every run; and
      `Done.record_pose`'s guard refuses every falsy-`arbitrated` record over
      a stored judgment, so no re-resolution can heal it. Nothing converges
      and nothing writes.

    The numeric clauses reject `bool` as well as `str` and `None`, because
    `float(True)` and `True < 0.45` do not raise: a `confidence: true` would
    be admitted and read as 1.0, a wrong number rather than a dropped entry,
    and the drop is the cheaper mistake.

    No production writer emits any of these shapes — `Poser._make_pose` fills
    every required key with a float, and `_fold` stamps a judgment only onto
    a pose that parked, which ran `needs_arbiter_margin` against a float — so
    they arrive from a hand-edited or foreign pose-cache.json. Each one that
    got through became a permanent per-file `Failure` (`route` raises, the
    driver's J3 boundary converts) in every run, never healed."""
    if not isinstance(entry, dict):
        return False
    if entry.get("v") != POSE_CACHE_VERSION:
        return False
    if not isinstance(entry.get("source"), str):
        return False
    if entry_up(entry) is None:
        return False
    if "confidence" in entry and not _number(entry["confidence"]):
        return False
    if "margin" not in entry or not (entry["margin"] is None
                                     or _number(entry["margin"])):
        return False
    if not isinstance(entry.get("front_view", {}), dict):
        return False
    arbitrated = entry.get("arbitrated")
    if not (arbitrated is None or arbitrated is True or arbitrated is False
            or arbitrated == "rejected"):
        return False
    return not (arbitrated in (True, "rejected")
                and entry["source"] != "vlm"
                and entry["margin"] is None)


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
    # One predicate, `_readable`, which says which accesses each clause comes
    # from. It replaced a list of inline clauses that had accreted one shape
    # at a time and was still admitting four more that crashed downstream
    # (adversarial review pass 3, 2026-08-31): the rule is the contract's
    # accesses, so it belongs somewhere it can be stated and read.
    #
    # Dropping rather than repairing costs a re-pose, so it was priced: 53 of
    # embed-cache512's 3540 entries, in the cache being rebuilt from scratch
    # that same night, and 0 of embed-cache-test's 2508 (census 2026-08-31,
    # after embed-cache2/3/4 were deleted — embed-cache3 had been nearly all
    # bare ints, and while it existed this had to repair instead of drop).
    fresh = {k: v for k, v in raw.items() if _readable(v)}
    if len(fresh) < len(raw):
        print(f"pose cache: {len(raw) - len(fresh)} of {len(raw)} entries predate "
              f"v{POSE_CACHE_VERSION} or carry a shape this code cannot process, "
              f"and will be re-resolved")
    return fresh


def save_pose_cache(cache_dir, cache):
    """The evals' writer, and theirs alone (J7).

    The bare `write_text` is deliberate — `pose` is the leaf both sides of the
    process boundary import and may not reach for `cachedir` to get an atomic
    one (interfaces.md's import-rule table). Anything that writes a
    pose-cache.json a later run will read goes through `cachedir.write_atomic`
    at its own call site instead, as `Done.flush` and `migrate_cache_keys.main`
    do; a tool reaching for this function is about to leave a torn cache
    behind on the one file whose loss costs money."""
    if not cache_dir:
        return
    p = Path(cache_dir) / "pose-cache.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache))


def repose_reopens(entry, arbiter_available, repose_arbiter):
    """Did `--repose` re-open this settled entry — is *that* why it is a miss?

    A predicate rather than a clause inside `pose_is_sufficient` because two
    callers need the same answer and a second copy would drift: sufficiency
    asks it to return False, and `cache_checker.route` asks it to decide
    `PoseRenderTask.force_escalate`. The Poser has to be *told*, because
    "re-opened by --repose" is invisible from where it stands: it sees a fresh
    ensemble margin, and a re-opened entry whose fresh margin clears this
    run's gate would otherwise take the ungated arm, record
    `arbitrated=None`, be refused by `Done.record_pose`'s guard, and be
    re-opened again by the next `--repose` run — re-rendered forever, never
    re-judged, and silent about it (adversarial review pass 3, 2026-08-31).

    False whenever `--repose` is off (`repose_arbiter is None`), so an
    ordinary miss — a cold entry, a geometry-only pass, an owed escalation —
    never forces a call."""
    return bool(repose_arbiter is not None and arbiter_available
                and entry is not None
                and entry.get("arbitrated") in (True, "rejected")
                and entry.get("arbiter") != repose_arbiter)


def pose_is_sufficient(entry, arbiter_available, margin_threshold,
                       *, repose_arbiter=None):
    """Is this cached pose good enough for the current run, or a miss?

    `margin` is None exactly when the SigLIP ensemble did not run — a
    geometry-only pass, written by some older `--no-up-ensemble` run. Treating
    those as hits would let one such pass pin every model to its geometry
    answer, and the ensemble — and the margin gate behind it, so the arbiter
    too — would never run again; so they read as misses and are upgraded in
    place. A VLM answer outranks the ensemble whichever gate escalated it, so
    it stands regardless of its margin.

    The four states of `arbitrated`, read against *this* run
    (docs/archive/tri-state-pass-2.md, 2026-08-21):

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
    either.

    `repose_arbiter` (`--repose`, 2026-08-31) is the run's own `arbiter_id`,
    and the one thing that re-opens a *settled* entry: a judgment recorded by
    a different judge is a miss, so the model re-poses and re-escalates to
    this run's arbiter. It fires on the `arbitrated` field alone, whatever the
    entry's margin does against today's gate — the point is re-judging a
    judgment, not re-checking the gate that bought it. Four notes:

    * it deliberately re-opens `"rejected"` too. A rejection is one API's
      verdict on one request, not a fact about the model: GLM refusing a sheet
      says nothing about whether gemini will. Under the *same* arbiter
      `"rejected"` stays as permanent as it has always been — the equality
      check is what makes that hold.
    * an entry with `arbitrated` truthy and no `arbiter` key compares as
      `None != repose_arbiter` and so is a miss. That is the deliberate
      backfill path: every judgment written before provenance existed gets
      re-bought once, by the first `--repose` run, and carries a stamp
      afterwards.
    * the `arbiter_available` guard is the same C3 doctrine as above — a
      degraded run must not re-render entries it cannot re-judge. Without it,
      `--repose` on a run whose arbiter probe failed would re-pose the
      collection and re-record it under no judge at all.
    * re-opening is only half the transaction, and the clause lives in
      `repose_reopens` above so `route` can carry the other half: the
      re-opened file must actually reach the arbiter, whatever its *fresh*
      margin does against the gate, or the flag never converges."""
    if entry is None:
        return False
    if repose_reopens(entry, arbiter_available, repose_arbiter):
        # Above the `source == "vlm"` hit deliberately: an answer a *different*
        # arbiter MOVED is the most valuable thing --repose re-buys, since the
        # measured backends disagree (glm +3 -> 41/44 where gemini is +4 ->
        # 42/44, LEARNINGS 2026-08-30) exactly on the poses one of them moved.
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

    The one validator, because three callers need the same answer and the
    first two used to assume the entry was well-formed: `embed_cache_token`
    below (so a malformed entry cannot crash a load), `collection.pose_of` (so
    it cannot fail a whole query response), and — since pass 3, 2026-08-31 —
    `_readable`, the loader's own predicate, which is what stops such an entry
    reaching `Pose.from_cache` in the first place. pose-cache.json is
    hand-editable; the two that turn up are a missing `up` and a null one.

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


# The exact rotations for the six `UP_CANDIDATES`: each takes its candidate to
# +Z by a quarter or half turn, so every entry is 0 or +/-1 and nothing here is
# approximate.
#
# Until the 2026-08-31 rebuild these were transcribed float-for-float from the
# Open3D construction that computed them, cos(pi/2) and sin(pi) noise included
# (`6.123233995736766e-17` where this table now reads `0.0`), because the table
# had to stay bit-identical to what drew every cached embedding: the key records
# only the up *vector*, so a last-bit difference would have re-posed cached
# models under unchanged keys. The rebuild regenerates every one of those
# embeddings, which is the only moment the noise can be dropped — see
# docs/cache-rebuild.md §1, and RECIPE_VERSION in identity.py, which is what
# makes any future change here visible in the key instead of silent.
_AXIS_ROTATIONS = {
    (0.0, 0.0, 1.0):  [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    (0.0, 0.0, -1.0): [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    (0.0, 1.0, 0.0):  [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    (0.0, -1.0, 0.0): [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    (1.0, 0.0, 0.0):  [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    (-1.0, 0.0, 0.0): [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
}


def rotation_to_z_up(up):
    """The rotation taking `up` to +Z. Pure numpy; it never touches a mesh.

    Lived in `renderer` and computed with Open3D until 2026-08-19, which meant
    a rotation matrix could not be named without a rendering library — the API
    needs one to publish `pose.azimuth_zero` (docs/api/surface.md).

    Two paths, and the split is the point. Every `up` this project resolves is
    one of `UP_CANDIDATES`, and each of those six is a quarter or half turn
    whose matrix is exactly 0 and +/-1 — so they are served from `_AXIS_ROTATIONS`
    rather than computed. Rodrigues on the same input lands ~5e-17 off in the
    entries that ought to be zero, and this function decides the pixels of every
    non-`+Z` render (`Renderer.views` rotates by it), so the table is what keeps
    a rotation the pipeline uses constantly from being a floating-point result.

    Anything else falls through to Rodrigues, which agrees with Open3D to
    ~1e-15. Nothing in the pipeline reaches it, since poses are always
    axis-aligned.

    **The input is normalised first, and that is load-bearing rather than
    tidiness.** A vector collinear with Z but not unit — `[0,0,2]`, or `[0,0,1]`
    off by a rounding step — has a zero cross product with Z, so Rodrigues
    divides by zero and yields an all-NaN matrix. Open3D's version hid this: it
    took `nan` axis-angle input and returned the *identity*, so `[0,0,-2]`
    silently rendered upside down rather than failing. Normalising sends every
    such vector to the table, where the answer is right, and it cannot miss the
    table on a rounding step: the six candidates are exactly unit, so the
    division is `x / 1.0 == x`.

    A zero vector raises: there is no rotation taking nothing to +Z, and a
    caller that has one is holding a bug, not an edge case."""
    v = np.asarray(up, dtype=float)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        raise ValueError(f"up must be a finite non-zero vector, got {up!r}")
    v = v / n
    exact = _AXIS_ROTATIONS.get(tuple(float(x) for x in v))
    if exact is not None:
        # a fresh array per call: a caller that mutates what it gets back must
        # not be mutating the table every later call reads
        return np.array(exact)
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
    out of range at 4 and silently wrong under different elevations. A config
    nothing has resolved yet is simply absent; a warm classify pass fills it
    in from cached embeddings, and consumers fall back to view 0 (a real
    render, just not necessarily the front) until it does.

    The shape is trusted because every caller — `Done._score`,
    `Collection.pose_of`, `cluster_models.py`, `test_categories.py` — reads
    entries that came through `load_pose_cache`, which drops the ones carrying
    the legacy spelling (a bare int) rather than passing them on."""
    return (entry or {}).get("front_view", {}).get(cfg)


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
    # bool subclasses int, so an unguarded check answers tile 1 for `true`
    if (isinstance(tile, int) and not isinstance(tile, bool)
            and 1 <= tile <= n_tiles):
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
    the two helpers below already promise (docs/archive/tri-state-pass-2.md,
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
    (docs/archive/tri-state-pass-2.md, 2026-08-21): a judged verdict can only arrive
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


class DeadlineExceeded(VLMUnavailable):
    """The wall clock ran out before the provider answered. Transient across
    runs, final within the call.

    Transient because the API never judged the request: `_fold` records
    `arbitrated=False` and the next run asks again, which is the whole reason
    it is a `VLMUnavailable` rather than a type of its own — the permanent
    side of the retry split is the enumerated one
    (docs/archive/tri-state-pass-2.md, 2026-08-21) and a deadline is not a
    verdict.

    Final because the loop that blew the deadline is *reproducible*: the
    provider keeps generating server-side after the client abandons the
    socket, so a retry re-runs the same reasoning loop and pays for it twice
    (measured 2026-08-30: `Floor` at max effort exhausted six 120 s attempts;
    separately, a solo call on `Body` abandoned at 120 s ran 142 s to
    completion server-side and billed its 17,584 output tokens). `ask_vlm_up`
    therefore breaks out of its retry loop on this type instead of spending
    the second attempt on it."""


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


def _split_transport(e):
    """Map a transport-layer failure onto the retry split. **Always raises.**

    Both arbiters face the same status space and must map it identically, and
    three review passes have each found a transient failure pinned permanent
    in one copy or the other — so the mapping gets one home. The 200-body
    splits stay with their callers: Vertex's `finishReason` and OpenRouter's
    embedded error envelope genuinely differ."""
    import urllib.error

    # HTTPError first: it subclasses OSError, so the order is the whole
    # classification, not a style choice.
    if isinstance(e, urllib.error.HTTPError):
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
        # only mid-run — the startup probe never makes a provider call
        # (docs/archive/tri-state-pass-2.md, 2026-08-21). Any 5xx is the server
        # failing, transient like a network drop, without 429/503's backoff.
        if e.code in TRANSIENT_HTTP_STATUS or e.code >= 500:
            raise VLMUnavailable(detail) from e
        raise VLMRejected(detail) from e
    # URLError and socket timeouts are OSErrors; a mid-read protocol error is
    # an HTTPException. None of them is the API saying "no", so none may land
    # on the permanent side of the retry split (review, 2026-08-20).
    raise VLMUnavailable(f"network failure: {e}") from e


def _ask_gemini(png_bytes, n_tiles, model, project=None):
    """Vertex AI arbiter. Raw HTTPS rather than an SDK: one POST, no dependency,
    and the same call the eval harness measured at 43/44 standalone."""
    import http.client
    import urllib.error
    import urllib.request

    # The environment being broken is not the request being judged: a missing
    # project or an expired ADC token is transient, and the token cache's
    # 1800 s TTL guarantees a collection run re-mints mid-run, where the
    # startup probe cannot see it (docs/archive/tri-state-pass-2.md, 2026-08-21).
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
    try:
        raw = urllib.request.urlopen(req, timeout=300).read()
    except (urllib.error.HTTPError, OSError, http.client.HTTPException) as e:
        _split_transport(e)
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


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GLM_MODEL = "z-ai/glm-5.3-flash"
GLM_PROVIDER = "DeepInfra"
"""The one standard-price OpenRouter provider that serves this request shape.

Pins are request-shape specific (measured 2026-08-30): with image + strict
JSON schema + reasoning, Relace, Z.AI and Novita all answer `404 No endpoints
found` though each advertises structured outputs — only DeepInfra ($0.075/M)
and Modal ($0.15/M) serve it. Auto-routing is worse than a bad pin: it sent
vision requests to providers that never finished them and to one that returned
7 output tokens at `effort: max`. Provider health flips hour to hour, so the
pin is overridable per run (`OPENROUTER_PROVIDER`)."""

GLM_DEADLINE_S = 120.0
"""Wall clock per call, enforced from outside the socket read — see
`_fetch_with_deadline` for why no socket timeout can do it. 120 s is the cap
the 2026-08-30 sweeps ran under; at `low` effort the median call is 3 s, so
only the pathological shapes ever approach it."""

_GLM_SCHEMA = {"type": "object", "properties": {"tile": {"type": "integer"}},
               "required": ["tile"], "additionalProperties": False}


def openrouter_key_path():
    """Where the OpenRouter bearer token lives. `OPENROUTER_KEY_FILE` overrides."""
    env = os.environ.get("OPENROUTER_KEY_FILE")
    return Path(env) if env else Path.home() / ".config/openrouter/key"


def openrouter_key():
    """The bearer token, read per call — and the backend's startup probe.

    A missing, unreadable or empty key file is `VLMUnavailable` for the same
    reason a missing gcloud token is: the environment being broken is not the
    request being judged, so it must not land on the permanent side of the
    retry split (docs/archive/tri-state-pass-2.md, 2026-08-21)."""
    path = openrouter_key_path()
    try:
        key = path.read_text().strip()
    except (OSError, ValueError) as e:    # ValueError: undecodable bytes
        raise VLMUnavailable(f"no OpenRouter key at {path}: {e}") from e
    if not key:
        raise VLMUnavailable(f"OpenRouter key file is empty: {path}")
    return key


def openrouter_provider():
    return os.environ.get("OPENROUTER_PROVIDER") or GLM_PROVIDER


def _fetch_with_deadline(req, deadline=None):
    """`urlopen(...).read()` on a daemon thread, abandoned at the deadline.

    Two things the code cannot say, both measured 2026-08-30:

    * **No socket timeout can do this job.** OpenRouter keeps a request alive
      with heartbeat bytes — ~1 packet every 150 ms — while the upstream
      provider hangs, so the socket is never idle long enough for `urlopen`'s
      own timeout to fire: workers sat idle 46 minutes on connections that
      were, as far as the socket knew, perfectly healthy. The wall clock has
      to be enforced from outside the read.
    * **`ThreadPoolExecutor` cannot host it.** Its workers are non-daemon and
      `concurrent.futures` registers an *untimed* atexit join of every one of
      them (CPython 3.9+), so an abandoned reader holds the interpreter open
      until its read finally ends — measured on 3.12.13, one 10 s abandoned
      call delayed process exit by 10.037 s, and under the heartbeat
      pathology above it would not end at all. That is CLAUDE.md's "every
      join on the render child is bounded" wearing another hat. A daemon
      thread is simply never joined at exit.

    The inner timeout is the deadline itself rather than something longer, so
    an ordinary stall reaps its own thread and being un-joinable at exit stays
    the backstop rather than the mechanism."""
    import urllib.request

    deadline = GLM_DEADLINE_S if deadline is None else deadline
    slot = []

    def read():
        try:
            slot.append((urllib.request.urlopen(req, timeout=deadline).read(), None))
        except BaseException as e:   # carried out and re-raised, never swallowed
            slot.append((None, e))

    worker = threading.Thread(target=read, name="or-deadline", daemon=True)
    worker.start()
    worker.join(timeout=deadline)
    if not slot:                     # still reading: abandon it and say so
        raise DeadlineExceeded(
            f"no answer within {deadline:g}s — unanswered, not retried")
    raw, err = slot[0]
    if err is not None:
        raise err
    return raw


def _ask_glm(tile_pngs, n_tiles, model):
    """OpenRouter arbiter (GLM-5.3-Flash), the measured fallback for a run with
    no gcloud (LEARNINGS, "Arbiter backends, sheet sizes, presentations and
    effort", 2026-08-30: +3 -> 41/44 against gemini's +4 -> 42/44, $0.06 a run
    against $2.7, 3 s median, and no GPU where the gemma/ollama fallback would
    evict SigLIP).

    Takes the tiles **separately**, not a contact sheet: presentation is worth
    nothing to gemini (±1 either way) and up to +3 to GLM, whose arbiter tier
    goes from +0/+2 on a sheet to +3 on six captioned images. Effort stays
    `low` — `max` scores no better standalone, nets +0 as the tier, and is
    where the deadline problem was measured."""
    import http.client
    import urllib.error
    import urllib.request

    key = openrouter_key()
    content = []
    for i, png in enumerate(tile_pngs, 1):
        content.append({"type": "text", "text": f"Tile {i}:"})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(png).decode()}})
    content.append({"type": "text", "text": UP_PROMPT})
    body = json.dumps({
        "model": model,
        "temperature": 0,
        "reasoning": {"effort": "low"},
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "tile", "strict": True, "schema": _GLM_SCHEMA}},
        "provider": {"order": [openrouter_provider()], "allow_fallbacks": False},
        "messages": [{"role": "user", "content": content}],
    }).encode()
    req = urllib.request.Request(OPENROUTER_URL, body,
                                 {"Authorization": f"Bearer {key}",
                                  "Content-Type": "application/json"})
    # `DeadlineExceeded` is a RuntimeError and matches none of these, so it
    # reaches `ask_vlm_up` unmapped — which is what breaks its retry loop.
    try:
        raw = _fetch_with_deadline(req)
    except (urllib.error.HTTPError, OSError, http.client.HTTPException) as e:
        _split_transport(e)
    try:
        d = json.loads(raw)
    except ValueError as e:
        raise VLMUnavailable(f"unparseable 200 body: {raw[:200]!r}") from e
    # OpenRouter reports upstream failures as an HTTP **200** carrying an
    # error envelope, so the same split has to be run twice — once on the
    # transport status and once on this. A code it does not state, or one that
    # is not a verdict, is transient: permanent stays the enumerated side.
    err = d.get("error")
    if err is not None:
        code = err.get("code") if isinstance(err, dict) else None
        detail = f"200 with embedded error {code!r}: {raw[:200]!r}"
        if code in (429, 503):
            raise RateLimited(detail)
        if (isinstance(code, int) and 400 <= code < 500
                and code not in TRANSIENT_HTTP_STATUS):
            raise VLMRejected(detail)
        raise VLMUnavailable(detail)
    choice = (d.get("choices") or [{}])[0]
    reason = choice.get("finish_reason")
    if reason == "content_filter":       # the only verdict this envelope states
        raise VLMRejected(f"blocked ({reason}): {raw[:200]!r}")
    text = (choice.get("message") or {}).get("content")
    if not text:
        raise VLMUnavailable(
            f"200 with no answer (finish_reason={reason!r}): {raw[:200]!r}")
    return parse_tile_answer(text, n_tiles)


DEFAULT_VLM_MODELS = {"ollama": "gemma4:26b", "gemini": GEMINI_MODEL,
                      "glm": GLM_MODEL, "claude": None}


def arbiter_id(backend, model):
    """Who judged, as one string: `"gemini/gemini-3.5-flash"`,
    `"glm/z-ai/glm-5.3-flash"`. None when there is no arbiter at all.

    One function because two callers must produce byte-identical strings or
    `--repose` re-poses the whole collection every run: `poser.Poser` stamps
    what it writes, `classify_stls.main` computes what to compare against.
    The model half can itself contain "/" (OpenRouter ids are org/model),
    which does not matter: nothing ever splits this string, and the only
    operation on it is whole-string equality. A backend that has no model id
    at all — `claude`, which is a CLI — stamps the bare backend name; the
    alternative asserts a model literally called "None"."""
    if not backend:
        return None
    model = model or DEFAULT_VLM_MODELS.get(backend)
    return f"{backend}/{model}" if model else backend


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
    * `DeadlineExceeded` — a `VLMUnavailable`, so it is asked again next run,
      but it is the one type that does **not** get this call's second
      attempt: the reasoning loop that ran out the clock is reproducible and
      is still generating server-side, so the retry pays for the same
      non-answer twice (see the type's docstring for the measurement);
    * `VLMRejected` — the API judged the request, which cannot succeed on a
      retry and would pay a call per run forever. It gets no arm of its own
      here: the generic arm below already retries once and re-raises under
      `raise_failures`, and a judged rejection rarely differs on attempt 2
      (docs/archive/tri-state-pass-2.md, 2026-08-21);
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

    save_to keeps a per-model copy of the sheet next to the saved renders,
    written whether or not the answer parses, so a wrong pose can be read back
    off disk. It is the image the VLM was shown for every backend except
    `glm`, which is sent the same six tiles separately — the sheet is the
    human's view of the call, not a transcript of it."""
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
            elif backend == "glm":
                # The sheet above is built and saved, and then not sent: GLM
                # answers +3 as the arbiter tier on six captioned tiles and
                # +0/+2 on the same tiles as one sheet (measured 2026-08-30).
                # The sheet is what a human reads back off disk; the tiles are
                # what the model is asked about.
                pngs = []
                for im in tiles:
                    # Capped like the sheet's own tiles: above SHEET_THUMB the
                    # per-tile token cost is unmeasured (6-12x at a
                    # --render-size of 1024+). thumbnail never enlarges, so a
                    # default-size run sends exactly the bytes it always did.
                    im = im.copy()
                    im.thumbnail((SHEET_THUMB, SHEET_THUMB))
                    buf = io.BytesIO()
                    im.save(buf, format="PNG")
                    pngs.append(buf.getvalue())
                idx = _ask_glm(pngs, len(tiles), vlm_model)
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
        except DeadlineExceeded as e:
            # The one failure that is not worth a second attempt: the loop
            # that blew the deadline is reproducible and still running (and
            # billing) server-side, so the retry buys another 120 s and
            # another bill for the same non-answer. Still recorded retryable
            # across runs — it is a VLMUnavailable.
            print(f"  pose VLM deadline ({backend}): {e}")
            last_error = e
            break
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
