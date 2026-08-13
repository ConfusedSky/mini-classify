from PIL import Image

from classify_stls import render_key
from migrate_renders import plan_dir


def collection(tmp_path, *rel_paths):
    """STLs at the given relative paths; returns them and a stem->files map."""
    files = []
    for rel in rel_paths:
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.touch()
        files.append(f)
    by_stem = {}
    for f in files:
        by_stem.setdefault(f.stem, []).append(f)
    return files, by_stem


def tile(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(path)


def plan(rdir, files, by_stem):
    return plan_dir(rdir, by_stem, {render_key(f) for f in files})


def test_renames_when_one_model_owns_the_filename(tmp_path):
    files, by_stem = collection(tmp_path, "stl/bunny.stl")
    rdir = tmp_path / "renders"
    tile(rdir / "bunny_view0.jpg")
    tile(rdir / "bunny_pose.png")
    renames, deletes, already, orphans = plan(rdir, files, by_stem)
    assert not deletes and not already and not orphans
    assert sorted(d.name for _, d in renames) == \
        [f"{render_key(files[0])}_pose.png", f"{render_key(files[0])}_view0.jpg"]


def test_deletes_the_images_of_a_shared_filename(tmp_path):
    # only one model's renders survived the overwrite and nothing records which
    files, by_stem = collection(tmp_path, "kit1/sword.stl", "kit2/sword.stl")
    rdir = tmp_path / "renders"
    tile(rdir / "sword_view0.jpg")
    renames, deletes, _, _ = plan(rdir, files, by_stem)
    assert not renames
    assert [p.name for p in deletes] == ["sword_view0.jpg"]


def test_leaves_files_it_cannot_account_for(tmp_path):
    # a render whose STL is gone, and something that is not a view at all
    files, by_stem = collection(tmp_path, "stl/bunny.stl")
    rdir = tmp_path / "renders"
    tile(rdir / "vanished_view0.jpg")
    tile(rdir / "contact-sheet.png")
    renames, deletes, _, orphans = plan(rdir, files, by_stem)
    assert not renames and not deletes and orphans == 2


def test_rerunning_it_changes_nothing(tmp_path):
    files, by_stem = collection(tmp_path, "stl/bunny.stl")
    rdir = tmp_path / "renders"
    tile(rdir / f"{render_key(files[0])}_view0.jpg")
    renames, deletes, already, orphans = plan(rdir, files, by_stem)
    assert not renames and not deletes and not orphans and already == 1


def test_drops_an_old_file_whose_migrated_copy_exists(tmp_path):
    # a half-finished --apply, or a rerun after new renders were saved
    files, by_stem = collection(tmp_path, "stl/bunny.stl")
    rdir = tmp_path / "renders"
    tile(rdir / "bunny_view0.jpg")
    tile(rdir / f"{render_key(files[0])}_view0.jpg")
    renames, deletes, _, _ = plan(rdir, files, by_stem)
    assert not renames
    assert [p.name for p in deletes] == ["bunny_view0.jpg"]


def test_a_stem_ending_in_view_digits_is_split_at_the_right_end(tmp_path):
    # "Sword_view2" is a legal model name; its render is "Sword_view2_view0"
    files, by_stem = collection(tmp_path, "stl/Sword_view2.stl")
    rdir = tmp_path / "renders"
    tile(rdir / "Sword_view2_view0.jpg")
    renames, deletes, _, orphans = plan(rdir, files, by_stem)
    assert not deletes and not orphans
    assert renames[0][1].name == f"{render_key(files[0])}_view0.jpg"
