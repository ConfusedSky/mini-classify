import zipfile

import pytest

from unpack_models import destination, extract, plan_zip


def make_zip(path, entries):
    """A zip at `path` holding {name: text}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        for name, text in entries.items():
            z.writestr(name, text)
    return path


def leaf(tmp_path, name="Barrier_NoSupports"):
    """The shape the real sets ship: a zip carrying its own top-level folder."""
    return make_zip(tmp_path / "Barrier" / f"{name}.zip",
                    {f"{name}/32mm_Barrier.stl": "solid\n"})


def test_destination_is_beside_the_zip_when_it_brings_its_own_root(tmp_path):
    z = leaf(tmp_path)
    with zipfile.ZipFile(z) as zf:
        dest, owns_root = destination(zf.namelist(), z)
    assert owns_root and dest == z.parent / "Barrier_NoSupports"


def test_destination_is_named_after_the_zip_when_it_is_a_bare_pile(tmp_path):
    # two such zips in one directory would otherwise interleave their files
    z = make_zip(tmp_path / "m" / "loose.zip", {"a.stl": "", "b.stl": ""})
    with zipfile.ZipFile(z) as zf:
        dest, owns_root = destination(zf.namelist(), z)
    assert not owns_root and dest == z.parent / "loose"


def test_extract_puts_the_stl_where_the_walk_will_find_it(tmp_path):
    z = leaf(tmp_path)
    _, dest, _ = plan_zip(z)
    extract(z, dest)
    assert (dest / "32mm_Barrier.stl").read_text() == "solid\n"
    assert z.exists()  # the archive is kept
    assert not list(z.parent.glob("*.partial"))


def test_a_second_run_on_an_intact_destination_is_done(tmp_path):
    z = leaf(tmp_path)
    _, dest, _ = plan_zip(z)
    extract(z, dest)
    action, again, _ = plan_zip(z)
    assert action == "done" and again == dest


def test_a_destination_that_no_longer_matches_the_archive_is_repaired(tmp_path):
    # the archive is the authority on what a finished extraction looks like.
    # The cost of that: a hand-edited file is treated as damage and replaced.
    z = leaf(tmp_path)
    _, dest, _ = plan_zip(z)
    extract(z, dest)
    (dest / "32mm_Barrier.stl").write_text("edited by hand\n")
    assert plan_zip(z)[0] == "repair"


def test_a_zero_length_file_is_repaired_not_called_done(tmp_path):
    # exactly what a drive going read-only mid-extraction leaves behind: the
    # tree in place, the files inside it empty. The old check — directory
    # exists and is non-empty — called this finished.
    z = leaf(tmp_path)
    _, dest, _ = plan_zip(z)
    extract(z, dest)
    (dest / "32mm_Barrier.stl").write_bytes(b"")
    assert plan_zip(z)[0] == "repair"
    extract(z, dest)
    assert (dest / "32mm_Barrier.stl").read_text() == "solid\n"
    assert plan_zip(z)[0] == "done"
    assert not list(z.parent.glob("*.partial"))


def test_a_missing_file_is_repaired(tmp_path):
    z = leaf(tmp_path)
    _, dest, _ = plan_zip(z)
    extract(z, dest)
    (dest / "32mm_Barrier.stl").unlink()
    (dest / "leftover.txt").write_text("keeps the directory non-empty")
    assert plan_zip(z)[0] == "repair"


def test_a_failed_repair_leaves_the_damaged_copy_in_place(tmp_path):
    # the swap happens only after the replacement is staged, so a mid-repair
    # failure must not cost the files that were already there
    z = leaf(tmp_path)
    _, dest, _ = plan_zip(z)
    extract(z, dest)
    (dest / "32mm_Barrier.stl").write_bytes(b"")
    z.write_bytes(z.read_bytes()[:-40])          # truncate the source
    with pytest.raises(Exception):
        extract(z, dest)
    assert dest.is_dir() and (dest / "32mm_Barrier.stl").exists()
    assert not list(z.parent.glob("*.partial"))


def test_an_empty_leftover_directory_is_not_mistaken_for_a_finished_one(tmp_path):
    z = leaf(tmp_path)
    (z.parent / "Barrier_NoSupports").mkdir()
    assert plan_zip(z)[0] == "extract"


def test_a_tagged_variant_is_skipped_without_reading_the_zip(tmp_path):
    # naming.skip decides this, so the archive need not exist
    z = tmp_path / "Barrier_Supported_LYCHEE.zip"
    assert plan_zip(z)[0] == "skip-tagged"


def test_the_walk_and_the_unpacker_agree_on_the_slicer_exports(tmp_path):
    # the case the shared filter exists for: no "supported" anywhere in the name
    assert plan_zip(tmp_path / "AlkaMyastan_32mm_LYCHEE.zip")[0] == "skip-tagged"


def test_all_unpacks_the_tagged_ones_too(tmp_path):
    z = leaf(tmp_path, name="Barrier_Supported")
    assert plan_zip(z, unpack_all=True)[0] == "extract"


def test_a_zip_that_would_escape_its_directory_is_refused(tmp_path):
    z = make_zip(tmp_path / "m" / "evil.zip", {"../../outside.stl": "no"})
    assert plan_zip(z, unpack_all=True)[0] == "unsafe"


def test_a_corrupt_zip_is_reported_not_raised(tmp_path):
    z = tmp_path / "m" / "broken_NoSupports.zip"
    z.parent.mkdir(parents=True)
    z.write_bytes(b"PK\x03\x04 and then nonsense")
    assert plan_zip(z)[0].startswith("unreadable")


def test_a_failed_extraction_leaves_no_half_model_behind(tmp_path):
    z = leaf(tmp_path)
    _, dest, _ = plan_zip(z)
    z.write_bytes(z.read_bytes()[:-40])  # truncate after planning
    with pytest.raises(Exception):
        extract(z, dest)
    assert not dest.exists()
    assert not list(z.parent.glob("*.partial"))
