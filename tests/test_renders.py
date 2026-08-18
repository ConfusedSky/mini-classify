"""The renders half of the cache: where they go (`renders_dir`/`render_subdir`),
what they are called (`render_key`), and that the two ends agree — what the
child writes (`Renderer.save_renders`) is exactly what `render_index` finds
again next run. That round trip is the only thing keeping a redraw decision
honest, and it crosses a process boundary, so it is pinned here from both
sides."""
import argparse
import os

import numpy as np
import pytest
from PIL import Image

from src import renderer as renderer_mod
from src.cachedir import embeds_dir, render_index, render_subdir, renders_dir
from src.identity import render_key
from src.messages import RenderConfig
from src.renderer import RENDER_FORMATS, Renderer


def args(render_size=2048, views=8, elevations=(20.0, -20.0)):
    return argparse.Namespace(render_size=render_size, views=views,
                              elevations=list(elevations))


def test_subdir_names_the_camera_config():
    assert render_subdir(args()) == "2048px-8v-e20,-20"
    assert render_subdir(args(512, 4, [20.0])) == "512px-4v-e20"


def test_subdir_formats_elevations_like_the_cache_key():
    # cache_key writes elevations as f"{e:g}" — the two must never disagree
    # about what one config is, or a config change silently reuses a directory
    assert render_subdir(args(512, 4, [20.0, -10.5])) == "512px-4v-e20,-10.5"


def test_subdir_separates_configs_that_share_filenames():
    # same view count, different cameras: identical <stem>_view<i> names, so the
    # directory is the only thing keeping the two apart
    assert render_subdir(args(512, 4, [20.0])) != render_subdir(args(512, 4, [40.0]))


def tile(path, size=(8, 8), colour="white"):
    Image.new("RGB", size, colour).save(path)


def test_key_separates_models_that_share_a_filename(tmp_path):
    # the real case: one Baal_Flaming_Sword_L per kit, both rendered into one
    # flat directory. Keyed by stem alone the second overwrote the first.
    a, b = tmp_path / "Kit I", tmp_path / "Kit II"
    for d in (a, b):
        d.mkdir()
        (d / "Baal_Flaming_Sword_L.stl").touch()
    ka, kb = render_key(a / "Baal_Flaming_Sword_L.stl", tmp_path), \
        render_key(b / "Baal_Flaming_Sword_L.stl", tmp_path)
    assert ka != kb
    assert ka.startswith("Baal_Flaming_Sword_L_")  # stem stays searchable


def test_key_is_stable_across_edits_to_the_file(tmp_path):
    # only the path is hashed: re-rendering replaces a model's own images
    # rather than leaving a stale set behind under a new name
    f = tmp_path / "bunny.stl"
    f.write_bytes(b"one")
    before = render_key(f, tmp_path)
    f.write_bytes(b"two different bytes")
    assert render_key(f, tmp_path) == before


def test_key_follows_the_file_not_the_spelling_of_the_path(tmp_path):
    # render_index keys come from filenames, so writer and readers must agree
    # even when one of them was handed a relative or unnormalised path
    (tmp_path / "sub").mkdir()
    f = tmp_path / "sub" / "bunny.stl"
    f.touch()
    assert render_key(tmp_path / "sub" / ".." / "sub" / "bunny.stl", tmp_path) \
        == render_key(f, tmp_path)


def test_the_cache_holds_both_derived_directories(tmp_path):
    # one --cache-dir is the whole cache: a renders directory paired with the
    # wrong cache would show one run's images beside another's embeddings
    assert embeds_dir(tmp_path) == tmp_path / "embeds"
    assert renders_dir(tmp_path, args(384, 8, [20.0, -20.0])) == \
        tmp_path / "renders" / "384px-8v-e20,-20"


def test_a_camera_config_change_is_a_different_render_directory(tmp_path):
    a = renders_dir(tmp_path, args(384, 8, [20.0]))
    b = renders_dir(tmp_path, args(512, 8, [20.0]))
    assert a != b


def test_caching_off_derives_no_directories(tmp_path):
    # --cache-dir '' disables the cache; --save-renders then has nowhere to go
    assert embeds_dir("") is None
    assert renders_dir("", args()) is None


def test_index_resolves_whatever_format_was_written(tmp_path):
    tile(tmp_path / "bunny_view0.png")
    tile(tmp_path / "bunny_view1.jpg")
    index = render_index(tmp_path)
    assert index["bunny_view0"].suffix == ".png"
    assert index["bunny_view1"].suffix == ".jpg"


def test_index_prefers_the_newest_when_a_view_exists_twice(tmp_path):
    # a format switch can leave both behind; the newer one is the current pose
    tile(tmp_path / "bunny_view0.png")
    tile(tmp_path / "bunny_view0.jpg")
    os.utime(tmp_path / "bunny_view0.png", (1, 1))
    assert render_index(tmp_path)["bunny_view0"].suffix == ".jpg"


def test_index_omits_missing_views_and_missing_dirs(tmp_path):
    tile(tmp_path / "bunny_view0.jpg")
    assert "bunny_view1" not in render_index(tmp_path)
    assert render_index(tmp_path / "never-rendered") == {}
    assert render_index(None) == {}


def test_index_does_not_confuse_stems_sharing_a_prefix(tmp_path):
    tile(tmp_path / "bunny_view0.jpg")
    tile(tmp_path / "bunnyhop_view0.jpg")
    index = render_index(tmp_path)
    assert index["bunny_view0"].name == "bunny_view0.jpg"
    assert index["bunnyhop_view0"].name == "bunnyhop_view0.jpg"


# --- what the child writes, and what this module reads back (F-4) ------------
#
# The production writer is `Renderer.save_renders` in the render child; the
# module-level `classify_stls.save_renders` these tests used to call was a
# second copy with no production caller, so a divergence between the names it
# wrote and the names `render_index` parses could not have shown up here. The
# fixture stubs `make_offscreen` only because the real one needs the GPU and,
# once created, must live for the process lifetime (CLAUDE.md) — `save_renders`
# never touches it.

@pytest.fixture
def saver(monkeypatch, tmp_path):
    monkeypatch.setattr(renderer_mod, "make_offscreen", lambda size: None)

    def make(rdir, fmt="jpg", root=tmp_path):
        return Renderer(RenderConfig(
            render_size=64, views=2, elevations=(20.0,), save_renders_dir=rdir,
            render_format=fmt, budget_bytes=1 << 20, collection_root=root))
    return make


def frames(n=2):
    return [np.zeros((8, 8, 3), dtype=np.uint8)] * n


@pytest.mark.parametrize("fmt", sorted(RENDER_FORMATS))
def test_save_renders_writes_every_view_in_the_chosen_format(tmp_path, saver, fmt):
    ext, _ = RENDER_FORMATS[fmt]
    f = tmp_path / "bunny.stl"
    f.touch()
    saver(tmp_path / "cfg", fmt).save_renders(f, frames())
    key = render_key(f, tmp_path)
    assert sorted(p.name for p in (tmp_path / "cfg").iterdir()) == \
        [f"{key}_view0{ext}", f"{key}_view1{ext}"]


@pytest.mark.parametrize("fmt", sorted(RENDER_FORMATS))
def test_what_the_child_writes_is_what_render_index_finds(tmp_path, saver, fmt):
    """The round trip, in every format: `render_index` keys are
    '<render_key>_view<i>', and that string is what `route`'s redraw check
    looks up. A writer that spelled the name differently would silently
    re-render the whole collection every run."""
    rdir = tmp_path / "cfg"
    f = tmp_path / "kit" / "bunny.stl"
    f.parent.mkdir()
    f.touch()
    saver(rdir, fmt).save_renders(f, frames(3))
    key = render_key(f, tmp_path)
    index = render_index(rdir)
    assert [f"{key}_view{i}" in index for i in range(3)] == [True] * 3
    assert index[f"{key}_view0"].suffix == RENDER_FORMATS[fmt][0]


def test_save_renders_never_fails_the_run(tmp_path, saver, capsys):
    # these files exist for a human to look at, so a write failure is a warning
    blocked = tmp_path / "ro"
    blocked.mkdir(mode=0o500)
    f = tmp_path / "bunny.stl"
    f.touch()
    try:
        saver(blocked / "cfg").save_renders(f, frames(1))
    finally:
        blocked.chmod(0o700)
    assert "could not save renders" in capsys.readouterr().out
