import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from src.cachedir import (RUN_PARAMS_FILE, RUN_PARAMS_KEYS, apply_run_params,
                          cache_key)
from src.identity import DEFAULT_ELEVATIONS

REPO = Path(__file__).resolve().parent.parent


def apply(tmp_path, monkeypatch, recorded, parser):
    (tmp_path / RUN_PARAMS_FILE).write_text(json.dumps(recorded))
    parser.add_argument("--cache-dir", default=str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["prog"])
    return apply_run_params(parser)


def test_only_manifest_keys_flow_between_tools(tmp_path, monkeypatch):
    # "pool" was removed from RUN_PARAMS_KEYS: the classifier's scoring
    # default must not override test_categories' own softmax default — but a
    # run-params.json written back when pool *was* recorded still contains it,
    # so the read has to be gated by the manifest too, not just the write
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="softmax")
    parser.add_argument("--views", type=int, default=4)
    args = apply(tmp_path, monkeypatch, {"pool": "mean", "views": 8}, parser)
    assert args.pool == "softmax"   # the tool's own default wins
    assert args.views == 8          # cache-identity keys still flow


def test_command_line_still_beats_the_manifest(tmp_path, monkeypatch):
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--cache-dir", default=str(tmp_path))
    (tmp_path / RUN_PARAMS_FILE).write_text(json.dumps({"views": 8}))
    monkeypatch.setattr(sys, "argv", ["prog", "--views", "6"])
    assert apply_run_params(parser).views == 6


def test_pool_is_not_in_the_manifest():
    assert "pool" not in RUN_PARAMS_KEYS


def test_compile_is_cache_identity():
    # compiled embeddings drift ~1e-03 from eager, so the regime must travel
    # with the cache like any other identity key
    assert "compile" in RUN_PARAMS_KEYS


def test_compile_keys_a_separate_regime(tmp_path):
    f = tmp_path / "a.stl"
    f.write_bytes(b"x")

    def key(**kw):
        return cache_key(f, argparse.Namespace(
            views=4, render_size=512, elevations=DEFAULT_ELEVATIONS,
            model="m", **kw), "auto", tmp_path)

    legacy = key()               # a parser that predates the flag entirely
    eager = key(compile=False)
    compiled = key(compile=True)
    assert eager == legacy       # pre---compile caches survive byte-identical
    assert compiled != eager     # the two numeric regimes never share a key


def test_the_manifest_is_written_before_the_model_load(tmp_path):
    """A cache that is mid-build must already describe itself.

    The manifest used to be written when the run *exited*, so a cache being
    filled by a live run — or one whose run died to kill -9 or an OOM — held
    real .npy files under keys nobody could reconstruct: every reader fell
    back to parser defaults (`--model`, `--views`), computed keys that missed
    every entry, and reported a half-built cache as unusable.

    Driven as a subprocess, because nothing may import `classify_stls.py`
    (CLAUDE.md) — and stopped inside the Embedder, the same way
    test_pose_vlm_prompt.py stops it: `--model` names a repository that does
    not exist and the environment is pinned offline, so the run dies exactly
    where a real OOM would, with `loading <model> ...` as the proof it got
    that far. `--pose-vlm off` keeps the arbiter out of it. No GPU, no
    network, no renders, and nothing outside tmp_path."""
    stls = tmp_path / "stls"
    stls.mkdir()
    (stls / "a.stl").write_bytes(b"solid x\nendsolid x\n")
    cats = tmp_path / "categories.txt"
    cats.write_text("a knight\na tree\n")
    cache = tmp_path / "cache"

    out = subprocess.run(
        [sys.executable, "classify_stls.py", str(stls),
         "--cache-dir", str(cache), "--categories", str(cats),
         "--model", "no-such-org/no-such-model", "--pose-vlm", "off",
         # not the parser defaults: the point is that the *run's* values are
         # what a reader finds, not what argparse would have guessed
         "--views", "6", "--render-size", "128"],
        cwd=REPO, env=dict(os.environ, HF_HUB_OFFLINE="1",
                           TRANSFORMERS_OFFLINE="1"),
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=300)

    assert out.returncode != 0                                  # died loading
    assert "loading no-such-org/no-such-model" in out.stdout     # got that far
    params = json.loads((cache / RUN_PARAMS_FILE).read_text())
    assert params["views"] == 6
    assert params["render_size"] == 128
    assert params["model"] == "no-such-org/no-such-model"
    assert params["input"] == str(stls.resolve())
    assert params["collection_root"] == str(stls.resolve())
