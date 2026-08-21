"""src/cachedir.py: `write_atomic`, the one copy of temp-then-replace, and the
`cache_version` totality guard.

The walk-cache and stamp *consumers* of write_atomic are pinned where their
semantics live (tests/test_collection.py, tests/test_migrate_cache_keys.py);
here is the idiom itself, because both hand-rolled copies of it got half
wrong: a fixed `.tmp` name that two concurrent writers tore, and no unlink on
a failed replace (review, 2026-08-20).
"""
import json
import os
from pathlib import Path

import pytest

from src import cachedir
from src.cachedir import cache_version, write_atomic


def test_write_atomic_publishes_and_leaves_no_tmp(tmp_path):
    target = tmp_path / "walk.json"
    write_atomic(target, json.dumps({"files": []}))
    assert json.loads(target.read_text()) == {"files": []}
    assert list(tmp_path.iterdir()) == [target]          # nothing stranded


def test_write_atomic_overwrites_without_a_window(tmp_path):
    target = tmp_path / "walk.json"
    write_atomic(target, "old")
    write_atomic(target, "new")
    assert target.read_text() == "new"
    assert list(tmp_path.iterdir()) == [target]


def test_write_atomic_temp_names_are_unique_per_writer(tmp_path, monkeypatch):
    """The concurrent-writer failure: `classify_stls.py` and a
    `POST /reload {"rescan": true}` derive the identical walk-cache path, and
    a shared `.tmp` let the second `write_text` truncate the first's live
    handle — publishing NUL-padded JSON, with the loser's replace raising
    FileNotFoundError. mkstemp's uniqueness is the fix; asserted by holding
    the first write's temp open while the second runs."""
    target = tmp_path / "walk.json"
    real_replace, held = os.replace, {}

    def stalled_first_replace(src, dst):
        if not held:
            held["src"] = src                # writer A parked before replace
            write_atomic(target, "B")        # writer B runs start-to-finish
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", stalled_first_replace)
    write_atomic(target, "A")
    # B wrote its own temp file, not A's; A's replace still lands last
    assert target.read_text() == "A"
    assert list(tmp_path.iterdir()) == [target]


def test_write_atomic_unlinks_the_tmp_when_replace_fails(tmp_path, monkeypatch):
    def full_disk(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", full_disk)
    with pytest.raises(OSError):
        write_atomic(tmp_path / "walk.json", "data")
    assert list(tmp_path.iterdir()) == []                # ENOSPC strands nothing


def test_write_atomic_keeps_write_text_permissions(tmp_path):
    """mkstemp opens 0600; the published file must match what a plain
    `write_text` produces under this process's umask — asserted against a
    write_text oracle rather than a literal 0644, which was only true under
    umask 022 (review follow-up, 2026-08-21)."""
    target = tmp_path / "meta.json"
    write_atomic(target, "{}")
    oracle = tmp_path / "oracle.json"
    oracle.write_text("{}")
    assert (target.stat().st_mode & 0o777) == (oracle.stat().st_mode & 0o777)


def test_cache_version_with_caching_disabled_reads_nothing(tmp_path, monkeypatch):
    """`--cache-dir ''` disables caching; `Path("")` made this read
    `./cache-meta.json` from whatever directory the process ran in
    (review, 2026-08-20)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / cachedir.CACHE_META_FILE).write_text(
        json.dumps({"cache_version": 7}))
    assert cache_version("") == 0
    assert cache_version(None) == 0
