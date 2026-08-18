"""`route()` — the whole decision is in the return value (interfaces.md I14).

Table-driven over the cache-state cube: pose cached (absent / geometry-only /
siglip / vlm / forced axis) x embedding `.npy` present or not x saved renders
present/partial/absent x `--skip-embed` / `--save-renders` / `--up-axis`
combinations. Cache states are fabricated on tmp_path; no GPU, no models, no
renderer.
"""
import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from src import cache_checker, pose
from src.cache_checker import route
from src.messages import (
    CacheContext,
    CachedHit,
    EmbedRenderTask,
    PoseRenderTask,
    Redraw,
    Retired,
)

REPO = Path(__file__).resolve().parent.parent

# Entry shapes as load_pose_cache leaves them (post-rename, current version).
ENTRIES = {
    "geo": {"up": [0.0, 0.0, 1.0], "confidence": 0.91, "source": "geometry",
            "margin": None, "v": pose.POSE_CACHE_VERSION},
    "siglip": {"up": [0.0, 1.0, 0.0], "confidence": 0.83, "source": "siglip",
               "margin": 0.61, "v": pose.POSE_CACHE_VERSION},
    "vlm": {"up": [1.0, 0.0, 0.0], "confidence": 1.0, "source": "vlm",
            "margin": None, "v": pose.POSE_CACHE_VERSION},
}


class ForbiddenPoses(dict):
    """A pose store that fails the test if consulted — pins the forced
    --up-axis shortcut (actors_proposal.md migration notes)."""

    def get(self, *a, **kw):
        raise AssertionError("pose cache consulted on forced --up-axis")

    def __getitem__(self, k):
        raise AssertionError("pose cache consulted on forced --up-axis")


def make_args(**over):
    d = dict(views=4, elevations=[20.0], render_size=512, model="test-model",
             compile=False, up_axis="auto", up_ensemble=True, skip_embed=False,
             save_renders=False, cache_dir="embed-cache")
    d.update(over)
    return argparse.Namespace(**d)


@dataclass
class Case:
    id: str
    expect: type
    up_axis: str = "auto"
    up_ensemble: bool = True
    skip_embed: bool = False
    save_renders: bool = False
    cache_dir: bool = True          # False: caching off entirely (embeds_dir None)
    pose_state: str = "siglip"      # key into ENTRIES | "absent" | "forced"
    embed_cached: bool = False
    renders: str = "none"           # "none" | "partial" | "all"
    needs_embed: bool | None = None  # asserted on EmbedRenderTask / Redraw.task
    retires: bool | None = None      # asserted on CachedHit


CASES = [
    # --- pose resolution owed: PoseRenderTask, whatever else is cached ------
    Case("pose-miss-cold", PoseRenderTask, pose_state="absent"),
    Case("pose-miss-under-skip-embed", PoseRenderTask, pose_state="absent",
         skip_embed=True),
    Case("geo-only-upgraded-by-ensemble", PoseRenderTask, pose_state="geo"),
    # --- pose sufficient, embedding owed: EmbedRenderTask(needs_embed=True) -
    Case("geo-accepted-without-ensemble", EmbedRenderTask, pose_state="geo",
         up_ensemble=False, needs_embed=True),
    Case("vlm-sufficient-despite-ensemble", EmbedRenderTask, pose_state="vlm",
         needs_embed=True),
    Case("cold-embed", EmbedRenderTask, needs_embed=True),
    # renders missing too: the fresh render covers saving — no Redraw shape
    Case("cold-embed-renders-missing", EmbedRenderTask, save_renders=True,
         renders="none", needs_embed=True),
    Case("caching-off", EmbedRenderTask, cache_dir=False, needs_embed=True),
    # --- warm: CachedHit -----------------------------------------------------
    Case("warm-hit", CachedHit, embed_cached=True, retires=True),
    Case("warm-hit-renders-complete", CachedHit, embed_cached=True,
         save_renders=True, renders="all", retires=True),
    # --- redraw: embedding cached, a saved render missing --------------------
    Case("redraw-partial", Redraw, embed_cached=True, save_renders=True,
         renders="partial", needs_embed=False, retires=False),
    Case("redraw-none", Redraw, embed_cached=True, save_renders=True,
         renders="none", needs_embed=False, retires=False),
    # --- --skip-embed warm paths (interfaces.md J1/Q2) ------------------------
    Case("skip-embed-nothing-wanted", Retired, skip_embed=True),
    Case("skip-embed-ignores-cached-npy", Retired, skip_embed=True,
         embed_cached=True),
    Case("skip-embed-renders-complete", Retired, skip_embed=True,
         save_renders=True, renders="all"),
    Case("skip-embed-renders-missing", EmbedRenderTask, skip_embed=True,
         save_renders=True, renders="partial", needs_embed=False),
    Case("skip-embed-save-renders-without-cache-dir", Retired, skip_embed=True,
         save_renders=True, cache_dir=False, renders="none"),
    # --- forced --up-axis: the pose-cache lookup is skipped ------------------
    Case("forced-z-cold", EmbedRenderTask, up_axis="z", pose_state="forced",
         needs_embed=True),
    Case("forced-z-warm", CachedHit, up_axis="z", pose_state="forced",
         embed_cached=True, retires=True),
    Case("forced-y-warm", CachedHit, up_axis="y", pose_state="forced",
         embed_cached=True, retires=True),
    Case("forced-z-skip-embed", Retired, up_axis="z", pose_state="forced",
         skip_embed=True),
    Case("forced-z-redraw", Redraw, up_axis="z", pose_state="forced",
         embed_cached=True, save_renders=True, renders="partial",
         needs_embed=False, retires=False),
]


def build(tmp_path, case):
    """Fabricate the cache state a Case describes; returns (f, ctx, expected)
    where expected carries the pose and cache_file route should emit."""
    root = tmp_path / "collection"
    (root / "kit").mkdir(parents=True)
    f = root / "kit" / "Baal_Flaming_Sword_L.stl"
    f.write_bytes(b"solid not-really\n")

    args = make_args(up_axis=case.up_axis, up_ensemble=case.up_ensemble,
                     skip_embed=case.skip_embed, save_renders=case.save_renders,
                     cache_dir=str(tmp_path / "cache") if case.cache_dir else None)

    if case.pose_state == "forced":
        poses = ForbiddenPoses()
        entry = None
        expected_pose = pose.Pose(up=pose.FORCED_UPS[case.up_axis],
                                  confidence=0.0, source="forced",
                                  v=pose.POSE_CACHE_VERSION)
    elif case.pose_state == "absent":
        poses, entry, expected_pose = {}, None, None
    else:
        entry = dict(ENTRIES[case.pose_state])
        poses = {pose.file_identity(f, root): entry}
        expected_pose = pose.Pose.from_cache(entry)

    embeds = None
    cache_file = None
    if case.cache_dir:
        embeds = tmp_path / "cache" / "embeds"
        embeds.mkdir(parents=True)
        token = pose.embed_cache_token(entry, case.up_axis)
        if token != "unresolved":
            ident = pose.file_identity(f, root)
            key = cache_checker.cache_key_from_identity(ident, args, token)
            cache_file = embeds / f"{key}.npy"
            if case.embed_cached:
                cache_file.touch()

    n_views = args.views * len(args.elevations)
    rkey = cache_checker.render_key(f, root)
    present = {"all": range(n_views),
               "partial": [i for i in range(n_views) if i != 2],
               "none": ()}[case.renders]
    render_index = {f"{rkey}_view{i}": tmp_path / f"r{i}.jpg" for i in present}

    ctx = CacheContext(poses=poses, embeds_dir=embeds,
                       render_index=render_index, args=args, root=root)
    return f, ctx, expected_pose, cache_file


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_route_decision_table(tmp_path, case):
    f, ctx, expected_pose, cache_file = build(tmp_path, case)
    out = route(f, 7, ctx)

    assert type(out) is case.expect
    if isinstance(out, Redraw):
        task, hit = out.task, out.hit
        assert (task.file, task.index) == (f, 7)
        assert (hit.file, hit.index) == (f, 7)
        assert task.needs_embed is case.needs_embed
        assert hit.retires is case.retires
        assert task.pose == expected_pose and hit.pose == expected_pose
        assert hit.cache_file == cache_file
    else:
        assert (out.file, out.index) == (f, 7)
        if isinstance(out, EmbedRenderTask):
            assert out.needs_embed is case.needs_embed
            assert out.pose == expected_pose
        elif isinstance(out, CachedHit):
            assert out.retires is case.retires
            assert out.pose == expected_pose
            assert out.cache_file == cache_file


def test_route_keys_embed_cache_on_the_pose_up(tmp_path):
    """The embedding token is the up vector (P2.3-B): a warm hit's cache_file
    must sit under up_str(entry['up']), and a forced axis under the flag's
    vector — the same key when they agree on the same up."""
    case = Case("k", CachedHit, embed_cached=True, retires=True)
    f, ctx, _, cache_file = build(tmp_path, case)
    ident = pose.file_identity(f, ctx.root)
    expected = cache_checker.cache_key_from_identity(ident, ctx.args, "0,1,0")
    assert cache_file.name == f"{expected}.npy"
    assert route(f, 0, ctx).cache_file == cache_file


def test_route_raises_on_vanished_file(tmp_path):
    """J3: the error boundary is the driver's — route raises, never guards."""
    case = Case("gone", PoseRenderTask, pose_state="absent")
    f, ctx, _, _ = build(tmp_path, case)
    f.unlink()
    with pytest.raises(OSError):
        route(f, 0, ctx)
    ctx.args.up_axis = "z"          # the forced path stats too (embed key)
    ctx.poses = ForbiddenPoses()
    with pytest.raises(OSError):
        route(f, 0, ctx)


def test_key_composition_matches_classify_stls(tmp_path):
    """The extracted key builders stay byte-identical to the originals in
    classify_stls.py (which route cannot import: it pulls in torch)."""
    classify_stls = pytest.importorskip("classify_stls")
    ident = "kit/model.stl|1723600000|4096"
    for over in (dict(),
                 dict(elevations=[20.0, -10.0]),
                 dict(compile=True),
                 dict(views=8, render_size=384)):
        args = make_args(**over)
        assert cache_checker.cache_key_from_identity(ident, args, "0,0,1") \
            == classify_stls.cache_key_from_identity(ident, args, "0,0,1")
    root = tmp_path
    f = tmp_path / "Baal_Flaming_Sword_L.stl"
    f.write_bytes(b"x")
    assert cache_checker.render_key(f, root) == classify_stls.render_key(f, root)
    assert cache_checker.EMBED_CACHE_VERSION == classify_stls.EMBED_CACHE_VERSION
    assert cache_checker.DEFAULT_ELEVATIONS == classify_stls.DEFAULT_ELEVATIONS


def test_import_pulls_no_torch_and_no_renderer():
    """Fresh interpreter: importing route must not import torch, and must not
    touch open3d's rendering module (no renderer parent-side)."""
    code = (
        "import sys\n"
        "import src.cache_checker\n"
        "assert 'torch' not in sys.modules, 'torch imported'\n"
        "assert 'classify_stls' not in sys.modules, 'classify_stls imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True)
