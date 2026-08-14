import argparse
import json
import sys

from classify_stls import RUN_PARAMS_FILE, RUN_PARAMS_KEYS, apply_run_params


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
