import argparse
import os

import pytest
from PIL import Image

from classify_stls import RENDER_FORMATS, render_index, render_subdir, save_renders


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


@pytest.mark.parametrize("fmt", sorted(RENDER_FORMATS))
def test_save_renders_writes_every_view_in_the_chosen_format(tmp_path, fmt):
    ext, _ = RENDER_FORMATS[fmt]
    save_renders(tmp_path / "cfg", "bunny", [Image.new("RGB", (8, 8))] * 2, fmt)
    assert sorted(p.name for p in (tmp_path / "cfg").iterdir()) == \
        [f"bunny_view0{ext}", f"bunny_view1{ext}"]


def test_save_renders_never_fails_the_run(tmp_path, capsys):
    # these files exist for a human to look at, so a write failure is a warning
    blocked = tmp_path / "ro"
    blocked.mkdir(mode=0o500)
    try:
        save_renders(blocked / "cfg", "bunny", [Image.new("RGB", (8, 8))], "jpg")
    finally:
        blocked.chmod(0o700)
    assert "could not save renders for bunny" in capsys.readouterr().out
