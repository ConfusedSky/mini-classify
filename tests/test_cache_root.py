import json

import pytest

import identity
from classify_stls import RUN_PARAMS_FILE, cache_root


def cache_with_root(tmp_path, recorded):
    """A cache dir whose run-params records a collection root."""
    d = tmp_path / "cache"
    d.mkdir(exist_ok=True)
    (d / RUN_PARAMS_FILE).write_text(json.dumps(
        {"collection_root": str(recorded)} if recorded else {}))
    return d


# --- identity.resolve_root: the decision, without the I/O --------------------

def test_no_recorded_root_takes_the_input(tmp_path):
    (tmp_path / "STL").mkdir()
    assert identity.resolve_root(tmp_path / "STL", None) == \
        ((tmp_path / "STL").resolve(), None)


def test_the_same_root_is_not_worth_reporting(tmp_path):
    stl = tmp_path / "STL"
    stl.mkdir()
    assert identity.resolve_root(stl, stl.resolve()) == (stl.resolve(), None)


def test_a_subdirectory_run_keeps_the_librarys_anchor(tmp_path):
    # the regression this exists for: running on one kit must key the way the
    # whole-library run did, or the same file is indexed twice
    stl = tmp_path / "STL"
    (stl / "Loot Studios").mkdir(parents=True)
    root, note = identity.resolve_root(stl / "Loot Studios", stl.resolve())
    assert (root, note) == (stl.resolve(), "subdir")


def test_somewhere_else_is_a_mismatch(tmp_path):
    stl, other = tmp_path / "STL", tmp_path / "Other"
    stl.mkdir()
    other.mkdir()
    root, note = identity.resolve_root(other, stl.resolve())
    assert (root, note) == (other.resolve(), "mismatch")


def test_a_root_that_no_longer_exists_is_a_mismatch(tmp_path):
    # the library moved: same shape as pointing at the wrong collection, which
    # is exactly why the caller has to ask rather than guess
    new = tmp_path / "new" / "STL"
    new.mkdir(parents=True)
    root, note = identity.resolve_root(new, tmp_path / "gone" / "STL")
    assert (root, note) == (new.resolve(), "mismatch")


# --- cache_root: what the decision does at the edge --------------------------

def test_a_subdirectory_run_is_announced_not_blocked(tmp_path, capsys):
    stl = tmp_path / "STL"
    (stl / "Kit").mkdir(parents=True)
    got = cache_root(stl / "Kit", cache_with_root(tmp_path, stl.resolve()))
    assert got == stl.resolve()
    assert "stay anchored" in capsys.readouterr().out


def test_a_mismatch_stops_a_non_interactive_run(tmp_path, monkeypatch):
    stl, other = tmp_path / "STL", tmp_path / "Other"
    stl.mkdir()
    other.mkdir()
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit) as e:
        cache_root(other, cache_with_root(tmp_path, stl.resolve()))
    assert "--reanchor" in str(e.value)


def test_reanchor_accepts_the_new_root(tmp_path, capsys):
    stl, other = tmp_path / "STL", tmp_path / "Other"
    stl.mkdir()
    other.mkdir()
    got = cache_root(other, cache_with_root(tmp_path, stl.resolve()), reanchor=True)
    assert got == other.resolve()
    assert "--reanchor given" in capsys.readouterr().out


def test_declining_the_prompt_stops_the_run(tmp_path, monkeypatch):
    stl, other = tmp_path / "STL", tmp_path / "Other"
    stl.mkdir()
    other.mkdir()
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    with pytest.raises(SystemExit):
        cache_root(other, cache_with_root(tmp_path, stl.resolve()))


def test_accepting_the_prompt_re_keys(tmp_path, monkeypatch):
    stl, other = tmp_path / "STL", tmp_path / "Other"
    stl.mkdir()
    other.mkdir()
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert cache_root(other, cache_with_root(tmp_path, stl.resolve())) == other.resolve()


def test_a_read_only_tool_warns_and_carries_on(tmp_path, capsys):
    # cluster_models/test_categories write nothing; blocking a REPL on a
    # prompt helps no one, but the miss has to be visible
    stl, other = tmp_path / "STL", tmp_path / "Other"
    stl.mkdir()
    other.mkdir()
    got = cache_root(other, cache_with_root(tmp_path, stl.resolve()), confirm=False)
    assert got == other.resolve()
    assert "read-only" in capsys.readouterr().out


def test_a_moved_library_says_so_in_the_prompt(tmp_path, capsys):
    new = tmp_path / "new" / "STL"
    new.mkdir(parents=True)
    cache_root(new, cache_with_root(tmp_path, tmp_path / "gone" / "STL"),
               reanchor=True)
    assert "no longer exists" in capsys.readouterr().out


def test_an_empty_cache_asks_nothing(tmp_path, capsys):
    stl = tmp_path / "STL"
    stl.mkdir()
    assert cache_root(stl, cache_with_root(tmp_path, None)) == stl.resolve()
    assert capsys.readouterr().out == ""
