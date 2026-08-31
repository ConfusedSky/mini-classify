import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from src.cachedir import (RUN_PARAMS_FILE, RUN_PARAMS_KEYS, apply_run_params,
                          cache_key, save_run_params, stamp_cache_version)
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


def scratch(tmp_path):
    """The smallest input a run will accept: one STL and a category file."""
    stls, cats = tmp_path / "stls", tmp_path / "categories.txt"
    stls.mkdir(exist_ok=True)
    (stls / "a.stl").write_bytes(b"solid x\nendsolid x\n")
    cats.write_text("a knight\na tree\n")
    return stls, cats


def a_run_that_dies_in_the_model_load(tmp_path, cache):
    """Drive the real CLI as a subprocess — nothing may import
    `classify_stls.py` (CLAUDE.md) — and stop it inside the Embedder the way
    test_pose_vlm_prompt.py stops it: `--model` names a repository that does
    not exist and the environment is pinned offline, so the run dies exactly
    where a real OOM would, with `loading <model> ...` as the proof it got
    that far. `--pose-vlm off` keeps the arbiter out of it. No GPU, no
    network, no renders, and nothing outside tmp_path."""
    stls, cats = scratch(tmp_path)
    out = subprocess.run(
        [sys.executable, "classify_stls.py", str(stls),
         "--cache-dir", str(cache), "--categories", str(cats),
         "--model", "no-such-org/no-such-model", "--pose-vlm", "off",
         # explicit, and divergent from both the parser defaults and the
         # seeded manifest below: these are the values a clobbering write
         # would leave behind, and they describe zero entries
         "--views", "6", "--render-size", "128"],
        cwd=REPO, env=dict(os.environ, HF_HUB_OFFLINE="1",
                           TRANSFORMERS_OFFLINE="1"),
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=300)
    assert out.returncode != 0                                  # died loading
    assert "loading no-such-org/no-such-model" in out.stdout     # got that far
    return out


def test_a_preflight_death_leaves_the_manifest_alone(tmp_path):
    """A manifest describes the run that is about to fill the cache, so a run
    that never gets to fill anything must not rewrite it.

    The repro (adversarial review, 2026-08-31): a good cache described
    views:4 and the 512 checkpoint; a rerun with a typo'd `--model` died in
    the model load, and a manifest written at startup was left saying views:6
    and a model that does not exist — so every reader then computed keys that
    missed every entry of a *fully built* cache, the same CacheUnusable the
    early write was introduced to prevent, aimed at a worse cache.

    Byte-compared rather than key-compared: the merge drops None, so a
    clobbering write is not required to change every field to be a
    catastrophe."""
    stls, cats = scratch(tmp_path)
    cache = tmp_path / "cache"
    # seeded through the production writer, like every cache fixture here
    stamp_cache_version(cache)
    save_run_params(argparse.Namespace(
        cache_dir=str(cache), input=str(stls), collection_root=str(stls.resolve()),
        views=4, elevations=list(DEFAULT_ELEVATIONS), render_size=512,
        model="google/siglip2-so400m-patch16-512", compile=False,
        up_axis="auto", categories=str(cats), render_format="jpg"))
    before = (cache / RUN_PARAMS_FILE).read_bytes()

    a_run_that_dies_in_the_model_load(tmp_path, cache)

    assert (cache / RUN_PARAMS_FILE).read_bytes() == before


def test_a_preflight_death_writes_no_manifest_at_all(tmp_path):
    """The same rule where there is nothing to clobber: a cache directory a
    dead run only touched must not end up describing entries it never wrote —
    a reader would take the description at face value."""
    cache = tmp_path / "cache"

    a_run_that_dies_in_the_model_load(tmp_path, cache)

    assert not (cache / RUN_PARAMS_FILE).exists()
