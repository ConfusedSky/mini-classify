import shutil
import zipfile
from pathlib import Path

import pytest

import unpack_models
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
    extract(z, dest, replace=True)
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
        extract(z, dest, replace=True)
    assert dest.is_dir() and (dest / "32mm_Barrier.stl").exists()
    assert not list(z.parent.glob("*.partial"))


def test_a_flat_zip_with_one_file_extracts_into_a_stem_dir(tmp_path):
    # review S1: one top-level entry that is a FILE is not a root. Treating it
    # as one misnested the model under a directory named after itself in one
    # revision and refused to extract in the next — broken differently in
    # every revision until now.
    z = make_zip(tmp_path / "Ranger.zip", {"32mm_Ranger.stl": "solid ranger\n"})
    dest, owns_root = destination(zipfile.ZipFile(z).namelist(), z)
    assert (dest.name, owns_root) == ("Ranger", False)
    action, dest, _ = plan_zip(z)
    assert action == "extract"
    extract(z, dest)
    assert (tmp_path / "Ranger" / "32mm_Ranger.stl").read_text() == "solid ranger\n"
    assert plan_zip(z)[0] == "done"      # not 'extract' forever


def test_zips_sharing_a_root_get_their_own_destinations(tmp_path):
    # fifteen thingiverse zips all carry the author's name as their root; the
    # one-destination rule would keep only whichever extracted last. Diverted,
    # every one extracts and a re-run finds them all done.
    a = make_zip(tmp_path / "war" / "snek.zip", {"j4roid/snek.stl": "solid snek\n"})
    b = make_zip(tmp_path / "war" / "deer.zip", {"j4roid/deer.stl": "solid deer\n"})
    overrides, shared = unpack_models.divert_collisions([a, b])
    assert overrides == {a: tmp_path / "war" / "snek", b: tmp_path / "war" / "deer"}
    [(d, group)] = shared
    assert d.name == "j4roid" and set(group) == {a, b}
    for z in (a, b):
        action, dest, _ = plan_zip(z, dest=overrides[z])
        assert action == "extract"
        extract(z, dest)
    assert (tmp_path / "war" / "snek" / "snek.stl").read_text() == "solid snek\n"
    assert (tmp_path / "war" / "deer" / "deer.stl").read_text() == "solid deer\n"
    assert plan_zip(a, dest=overrides[a])[0] == "done"
    assert plan_zip(b, dest=overrides[b])[0] == "done"


def test_destinations_do_not_depend_on_the_runs_flags(tmp_path):
    # review T1: a skip-tagged variant sharing the author root must divert the
    # group identically whether or not --all selects it — or the same zip
    # extracts to j4roid/ on one run and snek/ on the next, and both copies
    # then report done under their respective flags
    a = make_zip(tmp_path / "war" / "snek.zip", {"j4roid/snek.stl": "s\n"})
    b = make_zip(tmp_path / "war" / "snek_supported.zip", {"j4roid/snek_s.stl": "t\n"})
    overrides, _ = unpack_models.divert_collisions([a, b])
    assert overrides == {a: tmp_path / "war" / "snek",
                         b: tmp_path / "war" / "snek_supported"}


def test_a_diversion_target_colliding_with_a_derived_dest_also_diverts(tmp_path):
    # review T2: third.zip's own root IS snek/, snek.zip's diversion target —
    # without the fixed point it refused with a misleading repair message
    a = make_zip(tmp_path / "war" / "snek.zip", {"j4roid/snek.stl": "s\n"})
    b = make_zip(tmp_path / "war" / "deer.zip", {"j4roid/deer.stl": "d\n"})
    c = make_zip(tmp_path / "war" / "third.zip", {"snek/third.stl": "x\n"})
    overrides, shared = unpack_models.divert_collisions([a, b, c])
    assert overrides == {a: tmp_path / "war" / "snek",
                         b: tmp_path / "war" / "deer",
                         c: tmp_path / "war" / "third"}
    # review V1: snek/ was contested mid-diversion but ends as snek.zip's live
    # home — only j4roid/, nobody's final destination, is a stale leftover
    assert [d.name for d, _ in shared] == ["j4roid"]
    for z in (a, b, c):
        extract(z, overrides[z])
        assert plan_zip(z, dest=overrides[z])[0] == "done"


def test_a_zips_live_home_is_never_reported_as_a_stale_leftover(tmp_path):
    # review V1: snek.zip's root IS its own stem, so it never moves and snek/
    # is its correct destination — the advisory must not send a human to
    # delete a good extraction
    a = make_zip(tmp_path / "war" / "snek.zip", {"snek/a.stl": "solid a\n"})
    b = make_zip(tmp_path / "war" / "other.zip", {"snek/b.stl": "solid b\n"})
    overrides, shared = unpack_models.divert_collisions([a, b])
    assert overrides == {b: tmp_path / "war" / "other"}
    assert shared == []


def test_ignore_elsewhere_is_the_override(tmp_path):
    # review U1: the redundancy check is a strong heuristic, not proof — a
    # deliberate backup copy makes real zips look redundant, so there must be
    # a way past the check that is not "edit the module"
    z = curated(tmp_path)
    assert plan_zip(z)[0] == "elsewhere"
    assert plan_zip(z, check_elsewhere=False)[0] == "extract"


def test_a_diverted_zip_ignores_the_shared_leftover(tmp_path):
    # the shared dir holds one group member's content from the old rule —
    # byte-identical, so 'elsewhere' would wrongly declare that zip redundant
    # and it would never reach its own directory
    a = make_zip(tmp_path / "war" / "snek.zip", {"j4roid/snek.stl": "solid snek\n"})
    b = make_zip(tmp_path / "war" / "deer.zip", {"j4roid/deer.stl": "solid deer\n"})
    leftover = tmp_path / "war" / "j4roid"
    leftover.mkdir()
    (leftover / "snek.stl").write_text("solid snek\n")   # a's content, old rule
    overrides, _ = unpack_models.divert_collisions([a, b])
    assert plan_zip(a, dest=overrides[a])[0] == "extract"


def test_an_existing_destination_is_never_replaced_without_authorization(tmp_path):
    # the j4roid case: fifteen thingiverse zips share their author's name as
    # the root dir, so a root-level --apply would have each one destroy its
    # predecessor. Unauthorized, the second extraction refuses instead.
    z = leaf(tmp_path)
    _, dest, _ = plan_zip(z)
    extract(z, dest)
    (dest / "32mm_Barrier.stl").write_bytes(b"different")
    stray = dest.with_name(dest.name + ".partial")
    stray.mkdir()                               # someone else's interrupted run
    with pytest.raises(RuntimeError, match="opt-in"):
        extract(z, dest)                        # no replace: refuse
    assert (dest / "32mm_Barrier.stl").read_bytes() == b"different"
    assert stray.is_dir()                       # refusal touches nothing (N5)


def test_repair_authorization_is_per_zip_or_directory(tmp_path):
    z = leaf(tmp_path)
    assert not unpack_models.repair_authorized(z, [])
    assert unpack_models.repair_authorized(z, [str(z)])
    assert unpack_models.repair_authorized(z, [str(tmp_path)])
    assert not unpack_models.repair_authorized(z, [str(tmp_path / "other")])


def curated(tmp_path, text="solid\n"):
    """A zip plus a hand-curated copy of its model under another dir name."""
    z = make_zip(tmp_path / "Set" / "Kit.zip",
                 {"Kit/mini.stl": "solid\n", "Kit/render.jpg": "not kept"})
    other = tmp_path / "Set" / "My Curated Name"
    other.mkdir(parents=True)
    (other / "mini.stl").write_text(text)
    return z


def test_content_extracted_by_hand_elsewhere_is_recognised(tmp_path):
    # a set someone unzipped manually and curated into their own layout: every
    # model exists under the zip's parent by name, size and bytes, junk
    # pruned. The zip is redundant and must not re-extract a duplicate tree.
    assert plan_zip(curated(tmp_path))[0] == "elsewhere"


def test_a_name_and_size_coincidence_is_not_elsewhere(tmp_path):
    # review N2: same basename, same byte count, different mesh. The CRC
    # sample must catch it — a false 'elsewhere' means a zip is silently
    # never unpacked, the exact failure this script exists to fix.
    assert plan_zip(curated(tmp_path, text="SOLID\n"))[0] == "extract"


def test_all_widens_selection_but_keeps_the_redundancy_check(tmp_path):
    # --all means "include the skip-tagged variants", not "re-extract 19 GB
    # of content that is verifiably already on disk". The safety checks are
    # not what the flag bypasses.
    assert plan_zip(curated(tmp_path), unpack_all=True)[0] == "elsewhere"


def test_an_extractor_decoding_names_differently_still_lands(tmp_path, monkeypatch):
    # review N1: 7z decodes no-UTF-8-flag entry names as UTF-8 where zipfile
    # used cp437, so the staged root can carry a name destination() never
    # predicted. The swap adopts whatever single root was actually written.
    z = make_zip(tmp_path / "Kit.zip", {"Kit/mini.stl": "solid\n"})
    real = unpack_models.unzip_into

    def divergent(zpath, tmp):
        real(zpath, tmp)
        (tmp / "Kit").rename(tmp / "K┤ít")
    monkeypatch.setattr(unpack_models, "unzip_into", divergent)
    _, dest, _ = plan_zip(z)
    extract(z, dest)
    assert (dest / "mini.stl").read_text() == "solid\n"
    assert not list(z.parent.glob("*.partial"))


def test_a_kill_mid_swap_is_not_mistaken_for_redundancy(tmp_path):
    # review R1: a hard kill between the two renames leaves dest missing and
    # the original in .replaced — whose content is byte-identical, so the CRC
    # sample would happily confirm a false 'elsewhere' for an absent model
    z = leaf(tmp_path)
    _, dest, _ = plan_zip(z)
    extract(z, dest)
    dest.rename(dest.with_name(dest.name + ".replaced"))
    assert plan_zip(z)[0] == "extract"


def test_extract_restores_an_interrupted_swap_before_anything_else(tmp_path):
    # review R1: the orphaned aside copy is the original — put it back, and
    # stop so the stale plan gets remade against the restored tree
    z = leaf(tmp_path)
    _, dest, _ = plan_zip(z)
    extract(z, dest)
    (dest / "32mm_Barrier.stl").write_bytes(b"the original")
    dest.rename(dest.with_name(dest.name + ".replaced"))
    dest.with_name(dest.name + ".partial").mkdir()   # the killed run's staging
    with pytest.raises(RuntimeError, match="interrupted swap"):
        extract(z, dest)
    assert (dest / "32mm_Barrier.stl").read_bytes() == b"the original"
    assert not list(z.parent.glob("*.replaced"))
    # review S2: the rerun plans 'done' and never visits again, so the
    # untrustworthy staging must be swept now, not left on the drive forever
    assert not list(z.parent.glob("*.partial"))


def test_unexpected_staging_contents_raise_rather_than_misnest(tmp_path, monkeypatch):
    # review R2: with owns_root, anything but exactly one staged root dir used
    # to fall back to renaming tmp itself — the tree landed one level too deep
    # and the collection walk quietly found nothing. A loud stop beats that.
    z = make_zip(tmp_path / "Kit.zip", {"Kit/mini.stl": "solid\n"})
    real = unpack_models.unzip_into

    def extra(zpath, tmp):
        real(zpath, tmp)
        (tmp / "__MACOSX_leftover").write_text("")
    monkeypatch.setattr(unpack_models, "unzip_into", extra)
    _, dest, _ = plan_zip(z)
    with pytest.raises(RuntimeError, match="exactly the archive's root"):
        extract(z, dest)
    assert not dest.exists()
    assert not list(z.parent.glob("*.partial"))


def test_a_failed_swap_restores_the_original(tmp_path, monkeypatch):
    # review N1: the swap must never have a moment where neither copy exists —
    # a rename failing mid-swap puts the original back
    z = leaf(tmp_path)
    _, dest, _ = plan_zip(z)
    extract(z, dest)
    (dest / "32mm_Barrier.stl").write_bytes(b"damaged but mine")
    real = Path.rename

    def boom(self, target):
        if Path(target) == dest and ".partial" in str(self):
            raise OSError("injected mid-swap")
        return real(self, target)
    monkeypatch.setattr(Path, "rename", boom)
    with pytest.raises(OSError, match="injected"):
        extract(z, dest, replace=True)
    assert (dest / "32mm_Barrier.stl").read_bytes() == b"damaged but mine"
    assert not list(z.parent.glob("*.partial"))
    assert not list(z.parent.glob("*.replaced"))


@pytest.mark.skipif(not any(shutil.which(n) for n in ("7z", "7zz", "7za")),
                    reason="needs a 7z binary")
def test_a_method_python_cannot_decompress_goes_through_7z(tmp_path, monkeypatch):
    # Windows Explorer writes Deflate64 into large archives, which zipfile
    # lists but cannot extract. Force the same dispatch on an ordinary zip and
    # let the real 7z do the work end to end — staging and swap included.
    monkeypatch.setattr(unpack_models, "PY_METHODS", set())
    z = leaf(tmp_path)
    _, dest, _ = plan_zip(z)
    extract(z, dest)
    assert (dest / "32mm_Barrier.stl").read_text() == "solid\n"
    assert plan_zip(z)[0] == "done"
    assert not list(z.parent.glob("*.partial"))


def test_a_missing_7z_names_the_problem_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr(unpack_models, "PY_METHODS", set())
    monkeypatch.setattr(shutil, "which", lambda _: None)
    z = leaf(tmp_path)
    _, dest, _ = plan_zip(z)
    with pytest.raises(RuntimeError, match="needs 7z"):
        extract(z, dest)
    assert not dest.exists()
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
