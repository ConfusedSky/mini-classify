"""Done tests (interfaces.md §"Done — the only writer, and the owner of
retirement").

Every `on()` arm; J2 double-retirement idempotence; the retires=False
CachedHit (row, no retirement, no Release); Release exactly once per
retirement on a fake transport; flush idempotence (two calls, one atomic
replace each, identical bytes); and byte-shape parity of the CSV and
pose-cache output with what classify_stls.py writes today — the success-row
oracle replicates the score block (classify_stls.py:1197-1217) line for line,
so the numbers and the row dict are pinned against the production original's
*shape*: the ordering, the rounding, the field names. Its `pool_sims` is the
shared one (there is only one now), so what the oracle pins is the block
around it, not the pooling itself.

CPU torch throughout; no GPU, no renderer.
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from src import pose
from src.cache_checker import route
from src.cachedir import cache_key, view_config
from src.identity import cache_key_from_identity
from src.done import CSV_FIELDS, Done
from src.query import pool_sims
from src.messages import (
    CacheContext,
    CachedHit,
    Embedded,
    Failure,
    Release,
    Rendered,
    ResultRow,
    Retired,
)
from src.pose import Pose

DIM = 8
CATEGORIES = ["dragon", "terrain", "vehicle", "spaceship"]
N_VIEWS = 2      # views=2 x elevations=[20.0]


class FakeTransport:
    """Records every send — the Release assertions read `sent`."""

    def __init__(self):
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)

    def recv_nowait(self):
        return None

    def recv(self, timeout=None):
        return None

    def close(self):
        pass


@dataclass
class Admission:
    """Stand-in for the driver's counter (data_structures.md §Supervisor
    accounting): admitted is the driver's field, retired is Done's."""
    admitted: int = 0
    retired: int = 0

    def in_flight(self) -> int:
        return self.admitted - self.retired


def make_args(tmp_path, **over):
    d = dict(pool="mean", up_axis="auto", cache_dir=str(tmp_path / "cache"),
             out=str(tmp_path / "results.csv"), views=2, elevations=[20.0],
             render_size=384, model="google/siglip-so400m-patch14-384",
             skip_embed=False, up_ensemble=True, save_renders=None)
    d.update(over)
    return argparse.Namespace(**d)


@dataclass
class Rig:
    done: Done
    ctx: CacheContext
    admission: Admission
    tasks: FakeTransport
    text_embeds: torch.Tensor
    front_T: np.ndarray
    back_T: np.ndarray
    root: Path


def make_rig(tmp_path, **args_over) -> Rig:
    args = make_args(tmp_path, **args_over)
    root = tmp_path / "models"
    root.mkdir(exist_ok=True)
    embeds_dir = Path(args.cache_dir) / "embeds" if args.cache_dir else None
    if embeds_dir is not None:
        embeds_dir.mkdir(parents=True, exist_ok=True)
    ctx = CacheContext(poses={}, embeds_dir=embeds_dir, render_index={},
                       args=args, root=root)
    rng = np.random.default_rng(7)
    text_embeds = torch.from_numpy(
        rng.standard_normal((len(CATEGORIES), DIM)).astype(np.float32))
    front_T = rng.standard_normal((2, DIM)).astype(np.float32)
    back_T = rng.standard_normal((2, DIM)).astype(np.float32)
    admission, tasks = Admission(), FakeTransport()
    done = Done(admission, text_embeds, ctx, tasks, categories=CATEGORIES,
                front_embeds=front_T, back_embeds=back_T)
    return Rig(done, ctx, admission, tasks, text_embeds, front_T, back_T, root)


def stl(rig: Rig, name="a.stl") -> Path:
    f = rig.root / name
    if not f.exists():
        f.write_bytes(b"solid fake\nendsolid\n" + name.encode())
    return f


def img_embeds(seed=3) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.standard_normal((N_VIEWS, DIM)).astype(np.float32))


def a_pose(up=(0.0, 0.0, 1.0), source="geometry", conf=0.1234, margin=0.9) -> Pose:
    return Pose(up=up, confidence=conf, source=source, margin=margin,
                v=pose.POSE_CACHE_VERSION)


def today_row(f, p: Pose, fv, img, text, mode="mean"):
    """The score block exactly as classify_stls.process wrote it (:1197-1217)
    — the parity oracle for the row's shape, ordering and rounding."""
    view_sims = (img @ text.T).float().cpu().numpy()
    sims = torch.from_numpy(pool_sims(view_sims, mode))
    order = sims.argsort(descending=True)
    row = {"file": str(f), "up": pose.up_str(p.up), "pose_conf": p.confidence,
           "pose_source": p.source, "front_view": fv}
    for rank in range(min(3, len(CATEGORIES))):
        idx = order[rank]
        row[f"top{rank + 1}"] = CATEGORIES[idx]
        row[f"score{rank + 1}"] = round(sims[idx].item(), 4)
    return row


def expected_fv(rig: Rig, img) -> int:
    return pose.front_view_index(img.float().cpu().numpy(), rig.front_T, rig.back_T)


def save_hit_npy(rig: Rig, img) -> Path:
    p = rig.ctx.embeds_dir / "deadbeef.npy"
    np.save(p, img.numpy())
    return p


# --- on(): every arm ---------------------------------------------------------

def test_cached_hit_scores_retires_and_releases(tmp_path):
    rig = make_rig(tmp_path)
    f, img, p = stl(rig), img_embeds(), a_pose()
    rig.done.on(CachedHit(file=f, index=0, pose=p, cache_file=save_hit_npy(rig, img)))
    row = rig.done.rows[0]
    assert isinstance(row, ResultRow)
    assert row.to_csv() == today_row(f, p, expected_fv(rig, img), img, rig.text_embeds)
    assert rig.admission.retired == 1
    assert rig.tasks.sent == [Release(file=f, index=0)]


def test_cached_hit_retires_false_writes_row_only(tmp_path):
    rig = make_rig(tmp_path)
    f, img = stl(rig), img_embeds()
    rig.done.on(CachedHit(file=f, index=4, pose=a_pose(),
                          cache_file=save_hit_npy(rig, img), retires=False))
    assert isinstance(rig.done.rows[4], ResultRow)
    assert rig.admission.retired == 0
    assert rig.done.retired_ids == set()
    assert rig.tasks.sent == []          # no retirement, no Release


def test_embedded_saves_npy_where_todays_reader_looks(tmp_path):
    rig = make_rig(tmp_path)
    f, img, p = stl(rig), img_embeds(11), a_pose(source="siglip")
    rig.done.record_pose(f, 2, p)        # the Poser records before dispatching
    rig.done.on(Embedded(file=f, index=2, pose=p, embeds=img))
    # The write must land exactly where `cachedir.cache_key` (the production
    # original, main:classify_stls.py:639-645) derives the hit path from the
    # store.
    token = pose.embed_cache_token(rig.ctx.poses[pose.file_identity(f, rig.root)],
                                   rig.ctx.args.up_axis)
    expect = rig.ctx.embeds_dir / \
        f"{cache_key(f, rig.ctx.args, token, rig.root)}.npy"
    assert expect.exists()
    np.testing.assert_array_equal(np.load(expect), img.float().cpu().numpy())
    assert isinstance(rig.done.rows[2], ResultRow)
    assert rig.admission.retired == 1
    assert rig.tasks.sent == [Release(file=f, index=2)]


def test_embedded_without_embeds_dir_still_scores(tmp_path):
    rig = make_rig(tmp_path, cache_dir=None)
    assert rig.ctx.embeds_dir is None
    f, p = stl(rig), a_pose()
    rig.done.on(Embedded(file=f, index=0, pose=p, embeds=img_embeds()))
    assert isinstance(rig.done.rows[0], ResultRow)
    assert rig.admission.retired == 1


def test_failure_row_and_retirement(tmp_path):
    rig = make_rig(tmp_path)
    f = stl(rig)
    rig.done.on(Failure(file=f, index=9, error="mesh exploded"))
    assert rig.done.rows[9].to_csv() == \
        {"file": str(f), "top1": "RENDER_ERROR: mesh exploded"}  # :1127, :1169
    assert rig.admission.retired == 1
    assert rig.tasks.sent == [Release(file=f, index=9)]


def test_failure_overwrites_redraw_hit_row(tmp_path):
    # K5: on the redraw-failure path the Failure overwrites the hit's row —
    # parity with today, where a render failure reports RENDER_ERROR rather
    # than the cached score. One retirement total.
    rig = make_rig(tmp_path)
    f, img = stl(rig), img_embeds()
    rig.done.on(CachedHit(file=f, index=1, pose=a_pose(),
                          cache_file=save_hit_npy(rig, img), retires=False))
    rig.done.on(Failure(file=f, index=1, error="redraw died"))
    assert isinstance(rig.done.rows[1], Failure)
    assert rig.admission.retired == 1
    assert rig.tasks.sent == [Release(file=f, index=1)]


def test_retired_and_rendered_retire_without_rows(tmp_path):
    rig = make_rig(tmp_path)
    f, g = stl(rig, "a.stl"), stl(rig, "b.stl")
    rig.done.on(Retired(file=f, index=0))
    rig.done.on(Rendered(file=g, index=1))
    assert rig.done.rows == {}
    assert rig.admission.retired == 2
    assert rig.tasks.sent == [Release(file=f, index=0), Release(file=g, index=1)]


def test_unknown_message_raises(tmp_path):
    with pytest.raises(TypeError):
        make_rig(tmp_path).done.on(object())


# --- J2: retirement is idempotent, Release exactly once ----------------------

def test_double_retirement_is_ignored(tmp_path):
    rig = make_rig(tmp_path)
    rig.admission.admitted = 1
    f = stl(rig)
    rig.done.on(Retired(file=f, index=0))
    rig.done.on(Retired(file=f, index=0))          # repeat: no count, no Release
    rig.done.on(Rendered(file=f, index=0))         # cross-arm repeat too
    assert rig.admission.retired == 1
    assert rig.admission.in_flight() == 0          # never negative
    assert rig.tasks.sent == [Release(file=f, index=0)]


def test_release_exactly_once_per_retirement_across_arms(tmp_path):
    rig = make_rig(tmp_path)
    rig.admission.admitted = 5
    files = [stl(rig, f"m{i}.stl") for i in range(5)]
    img, p = img_embeds(), a_pose()
    rig.done.on(CachedHit(file=files[0], index=0, pose=p,
                          cache_file=save_hit_npy(rig, img)))
    rig.done.record_pose(files[1], 1, p)
    rig.done.on(Embedded(file=files[1], index=1, pose=p, embeds=img))
    rig.done.on(Failure(file=files[2], index=2, error="x"))
    rig.done.on(Retired(file=files[3], index=3))
    rig.done.on(Rendered(file=files[4], index=4))
    rig.done.on(Failure(file=files[2], index=2, error="again"))   # dup retire
    assert rig.admission.retired == 5
    assert rig.admission.in_flight() == 0
    assert rig.tasks.sent == [Release(file=files[i], index=i) for i in range(5)]


# --- record_pose and the front_view write (I9, D9) ---------------------------

def test_record_pose_writes_the_canonical_store(tmp_path):
    rig = make_rig(tmp_path)
    f, p = stl(rig), a_pose(source="vlm", margin=0.12)
    rig.done.record_pose(f, 0, p)
    ident = pose.file_identity(f, rig.root)
    assert rig.ctx.poses[ident] == p.to_cache()    # same object route() reads
    assert rig.done.poses is rig.ctx.poses


def test_front_view_resolved_once_and_merged_into_entry(tmp_path):
    rig = make_rig(tmp_path)
    f, img, p = stl(rig), img_embeds(), a_pose()
    rig.done.record_pose(f, 0, p)
    rig.done.on(CachedHit(file=f, index=0, pose=p, cache_file=save_hit_npy(rig, img)))
    entry = rig.ctx.poses[pose.file_identity(f, rig.root)]
    cfg = view_config(rig.ctx.args)
    assert entry["front_view"] == {cfg: expected_fv(rig, img)}   # :1203-1206


def test_cached_front_view_short_circuits(tmp_path):
    rig = make_rig(tmp_path)
    f, img, p = stl(rig), img_embeds(), a_pose()
    rig.done.record_pose(f, 0, p)
    entry = rig.ctx.poses[pose.file_identity(f, rig.root)]
    cfg = view_config(rig.ctx.args)
    entry["front_view"] = {cfg: 1}                 # pre-resolved for this cfg
    rig.done.on(CachedHit(file=f, index=0, pose=p, cache_file=save_hit_npy(rig, img)))
    assert rig.done.rows[0].front_view == 1        # cache wins, no recompute
    assert entry["front_view"] == {cfg: 1}


# --- E-R1-1: a forced --up-axis never reads the pose store -------------------

def test_forced_axis_ignores_warm_auto_entry(tmp_path):
    """A forced run over a cache warmed by an auto run (the case the old
    empty-store test hid). The store's up is a different axis, so its
    embeddings and its front_view describe different pixels: the fresh .npy
    must land under the *forced* token — where route() looks — and the row's
    front_view must be recomputed, with the auto entry left untouched."""
    rig = make_rig(tmp_path, up_axis="z")
    f, img = stl(rig), img_embeds(5)
    ident = pose.file_identity(f, rig.root)
    cfg, fv = view_config(rig.ctx.args), expected_fv(rig, img)
    # last run's auto resolution: a different up, front_view cached for this
    # very view config and deliberately not the answer this run computes
    warm = a_pose(up=(0.0, 1.0, 0.0), source="siglip").to_cache()
    warm["front_view"] = {cfg: 1 - fv}
    rig.ctx.poses[ident] = warm
    before = copy.deepcopy(warm)

    forced = Pose(up=pose.FORCED_UPS["z"], confidence=0.0, source="forced",
                  v=pose.POSE_CACHE_VERSION)
    rig.done.on(Embedded(file=f, index=0, pose=forced, embeds=img))

    # the .npy is exactly where route() looks for it on the forced path
    forced_npy = rig.ctx.embeds_dir / (
        cache_key_from_identity(ident, rig.ctx.args,
                                pose.up_str(pose.FORCED_UPS["z"])) + ".npy")
    hit = route(f, 0, rig.ctx)
    assert isinstance(hit, CachedHit)
    assert hit.cache_file == forced_npy and forced_npy.exists()
    np.testing.assert_array_equal(np.load(forced_npy), img.float().cpu().numpy())
    # and nothing was filed under the stale auto token
    assert not (rig.ctx.embeds_dir / (
        cache_key_from_identity(ident, rig.ctx.args,
                                pose.up_str(warm["up"])) + ".npy")).exists()
    # front_view recomputed for the forced pose; the auto entry unmodified
    assert rig.done.rows[0].front_view == fv
    assert rig.ctx.poses[ident] == before


def test_forced_axis_cached_hit_does_not_merge_into_auto_entry(tmp_path):
    """The CachedHit arm of the same rule: scoring a forced hit neither reads
    nor writes the store's front_view."""
    rig = make_rig(tmp_path, up_axis="y")
    f, img = stl(rig), img_embeds(5)
    ident = pose.file_identity(f, rig.root)
    cfg, fv = view_config(rig.ctx.args), expected_fv(rig, img)
    warm = a_pose(up=(0.0, 0.0, 1.0)).to_cache()
    warm["front_view"] = {cfg: 1 - fv}
    rig.ctx.poses[ident] = warm
    before = copy.deepcopy(warm)
    forced = Pose(up=pose.FORCED_UPS["y"], confidence=0.0, source="forced",
                  v=pose.POSE_CACHE_VERSION)
    rig.done.on(CachedHit(file=f, index=0, pose=forced,
                          cache_file=save_hit_npy(rig, img)))
    assert rig.done.rows[0].front_view == fv       # recomputed, not the entry's
    assert rig.ctx.poses[ident] == before


def test_auto_token_matches_the_store_round_trip(tmp_path):
    """The other half of E-R1-1: deriving the token from `m.pose.up` is
    equivalent to the store lookup on the auto path — record_pose round-trips
    `up` exactly, so the key is unchanged from what route() computed."""
    rig = make_rig(tmp_path)
    f, img, p = stl(rig), img_embeds(11), a_pose(up=(0.0, -1.0, 0.0),
                                                 source="vlm")
    rig.done.record_pose(f, 0, p)
    ident = pose.file_identity(f, rig.root)
    rig.done.on(Embedded(file=f, index=0, pose=p, embeds=img))
    from_store = pose.embed_cache_token(rig.ctx.poses[ident], "auto")
    assert from_store == pose.up_str(p.up)
    assert (rig.ctx.embeds_dir /
            (cache_key_from_identity(ident, rig.ctx.args, from_store) + ".npy")
            ).exists()


# --- flush: idempotent, atomic, byte-compatible ------------------------------

def flush_bytes(rig: Rig):
    csv_p = Path(rig.ctx.args.out)
    pose_p = Path(rig.ctx.args.cache_dir) / "pose-cache.json"
    return csv_p.read_bytes(), pose_p.read_bytes() if pose_p.exists() else None


def populated_rig(tmp_path, **args_over) -> Rig:
    rig = make_rig(tmp_path, **args_over)
    f, g, img, p = stl(rig, "a.stl"), stl(rig, "b.stl"), img_embeds(), a_pose()
    rig.done.record_pose(f, 0, p)
    rig.done.on(CachedHit(file=f, index=0, pose=p, cache_file=save_hit_npy(rig, img)))
    rig.done.on(Failure(file=g, index=1, error="boom"))
    return rig


def test_flush_idempotent_one_replace_each(tmp_path, monkeypatch):
    rig = populated_rig(tmp_path)
    real_replace, calls = os.replace, []
    monkeypatch.setattr(os, "replace",
                        lambda src, dst: (calls.append(dst), real_replace(src, dst))[1])
    rig.done.flush()
    first = flush_bytes(rig)
    assert len(calls) == 1                         # pose cache via one os.replace
    rig.done.flush()
    assert len(calls) == 2                         # idempotent: replayed, not skipped
    assert flush_bytes(rig) == first               # identical bytes both times
    assert not (Path(rig.ctx.args.cache_dir) / "pose-cache.json.tmp").exists()


def test_flush_pose_cache_byte_parity_with_save_pose_cache(tmp_path):
    rig = populated_rig(tmp_path)
    rig.done.flush()
    other = tmp_path / "oracle"
    other.mkdir()
    pose.save_pose_cache(other, rig.ctx.poses)     # today's writer (src/pose.py:200)
    assert (Path(rig.ctx.args.cache_dir) / "pose-cache.json").read_bytes() == \
        (other / "pose-cache.json").read_bytes()
    # and it round-trips through today's loader
    assert pose.load_pose_cache(rig.ctx.args.cache_dir) == rig.ctx.poses


def test_flush_csv_byte_parity_with_today(tmp_path):
    rig = make_rig(tmp_path)
    f, g, img, p = stl(rig, "a.stl"), stl(rig, "b.stl"), img_embeds(), a_pose()
    # out-of-index-order arrival: the flush sorts (actors_proposal.md §Done)
    rig.done.on(Failure(file=g, index=7, error="boom"))
    rig.done.record_pose(f, 2, p)
    rig.done.on(CachedHit(file=f, index=2, pose=p, cache_file=save_hit_npy(rig, img)))
    rig.done.flush()
    # Today's epilogue, verbatim shape (classify_stls.py:1261-1266): the
    # literal field list, DictWriter filling the error row's holes.
    buf = io.StringIO()
    fields = ["file", "top1", "score1", "top2", "score2", "top3", "score3",
              "up", "pose_conf", "pose_source", "front_view"]
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows([today_row(f, p, expected_fv(rig, img), img, rig.text_embeds),
                      {"file": str(g), "top1": "RENDER_ERROR: boom"}])
    assert Path(rig.ctx.args.out).read_bytes() == buf.getvalue().encode()
    assert CSV_FIELDS == fields


def test_flush_skips_pose_cache_when_up_axis_forced(tmp_path):
    rig = make_rig(tmp_path, up_axis="z")
    f, img = stl(rig), img_embeds()
    forced = Pose(up=(0.0, 0.0, 1.0), confidence=0.0, source="forced",
                  v=pose.POSE_CACHE_VERSION)
    rig.done.on(CachedHit(file=f, index=0, pose=forced,
                          cache_file=save_hit_npy(rig, img)))
    rig.done.flush()                               # classify_stls.py:1255 guard
    assert not (Path(rig.ctx.args.cache_dir) / "pose-cache.json").exists()
    assert Path(rig.ctx.args.out).exists()


def test_flush_writes_csv_even_when_pose_cache_fails(tmp_path, monkeypatch):
    """E-R1-2/E-R1-3 (the full-disk incident, classify_stls.py:1249-1268): a
    failing pose-cache write must not also cost the rows, must still
    propagate, and must not strand the .tmp."""
    rig = populated_rig(tmp_path)

    def full_disk(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", full_disk)
    with pytest.raises(OSError) as ei:
        rig.done.flush()
    assert ei.value.errno == 28                    # the pose failure propagates
    cache = Path(rig.ctx.args.cache_dir)
    assert not (cache / "pose-cache.json").exists()
    assert not (cache / "pose-cache.json.tmp").exists()          # E-R1-3
    # every finished row is on disk regardless
    with open(rig.ctx.args.out, newline="") as fh:
        written = list(csv.DictReader(fh))
    assert [r["file"] for r in written] == [str(rig.root / "a.stl"),
                                            str(rig.root / "b.stl")]


def test_flush_last_failure_reraises_after_both_writes_tried(tmp_path, monkeypatch):
    """Both writes fail: the last failure is what propagates, with the
    earlier one kept visible as __context__."""
    rig = populated_rig(tmp_path)
    Path(rig.ctx.args.out).mkdir(parents=True)     # the CSV open() fails too

    def full_disk(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", full_disk)
    with pytest.raises(IsADirectoryError) as ei:
        rig.done.flush()
    assert isinstance(ei.value.__context__, OSError)
    assert ei.value.__context__.errno == 28
    assert not (Path(rig.ctx.args.cache_dir) / "pose-cache.json.tmp").exists()


def test_flush_empty_rows_writes_header_only(tmp_path):
    rig = make_rig(tmp_path, skip_embed=True)
    rig.done.on(Retired(file=stl(rig), index=0))   # rows legitimately has holes
    rig.done.flush()
    assert Path(rig.ctx.args.out).read_bytes() == \
        (",".join(CSV_FIELDS) + "\r\n").encode()


# The two parity pins that lived here (`pool_sims` and `view_config` against
# classify_stls' copies) are retired with the copies: each has exactly one home
# now (`src/query.py` and `src/cachedir.py`), so both assertions had become a
# function compared with itself. `pool_sims` is still exercised through
# `today_row` on every success-row test, and `view_config` through the
# front_view keying below.
