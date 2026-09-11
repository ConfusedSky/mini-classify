import json
from pathlib import Path

import pytest

from src.naming import SKIP_TAGS, skip

REPO = Path(__file__).resolve().parent.parent


# every spelling that actually appears across the collection and its archives
@pytest.mark.parametrize("name", [
    "Barrier_NoSupports",
    "ArmoredDragonCultist_NoHeadInHand_32mm_No Supports",
    "Aimar - Unsupported",
    "32_Unsupported_Aimar_BodyMask",
    "32mm_PitFiend",
    "AjaxFighter_32mm_NoSupports",
])
def test_keeps_the_plain_model(name):
    assert not skip(name)


@pytest.mark.parametrize("name", [
    "Barrier_Supported",
    "Barrier_Supported_LYCHEE",
    "Bonus_32mm_Supported_CHITUBOX",
    "Remorhaz_32mm_Supported_Solid",
    "Igloo_Supported_Hollow",
    "Rebels_32mm_PreSupported",
    "Thing_pre-supported",
    "Thing_pre_supported",
    "32mm_Bashir_Base",
    "75mm_PitFiend",
])
def test_drops_the_variants(name):
    assert skip(name)


@pytest.mark.parametrize("name", [
    "AlkaMyastan_32mm_LYCHEE",   # a real archive: slicer named, "supported" not
    "Thing_CHITUBOX",
])
def test_drops_slicer_exports_that_never_say_supported(name):
    # the drift this module exists to close — the walk used to keep these
    assert skip(name)


def test_unsupported_is_not_read_as_supported():
    # "unsupported" contains "supported"; removing the word is what prevents it
    assert not skip("Unsupported")
    assert skip("Supported")


# Only "NoSupports", "No Supports" and "Unsupported" are confirmed present in the
# collection; the rest are defensive — these names come from many sculptors and
# nothing enforces a spelling. Every one of them fails *open* if missed, keeping
# a supported model, so breadth is the safe side to err on.
@pytest.mark.parametrize("name", [
    "Barrier_NoSupports",           # observed
    "Barrier_No Supports",          # observed
    "Barrier_Unsupported",          # observed
    "Barrier_No_Supports",
    "Barrier_No-Supports",
    "Barrier_NoSupport",
    "Barrier_no supports",
    "Barrier_Un-Supported",         # separator forms: the old .replace() missed
    "Barrier_Un_Supported",
    "Barrier_Un Supported",
    "Barrier_NonSupported",
    "Barrier_Non-Supported",
    "Barrier_Not Supported",
    "NoSupports_Barrier",           # leading, no separator before it
])
def test_every_way_of_saying_no_supports_is_kept(name):
    assert not skip(name)


@pytest.mark.parametrize("name", [
    "Casino_Supported",   # contains "no_supported"
    "Piano_Supported",
    "Uno_Supported",
    "Volcano Supported",
])
def test_a_negation_inside_an_ordinary_word_is_not_a_negation(name):
    # the dangerous direction: reading these as unsupported fails open and puts
    # a scaffolded model into the collection
    assert skip(name)


def test_unsupported_does_not_mask_a_second_tag():
    # removing the word must not smuggle the rest of the name past the check
    assert skip("Aimar_Unsupported_Base")
    assert skip("75mm_Unsupported_Hero")
    assert skip("Igloo_No Supports_Hollow")


def test_matching_ignores_case():
    assert skip("BARRIER_SUPPORTED") and skip("barrier_supported")


def test_the_two_groups_make_up_the_tag_list():
    from src.naming import NON_MODEL_TAGS, SUPPORT_TAGS
    assert SKIP_TAGS == SUPPORT_TAGS + NON_MODEL_TAGS


@pytest.mark.parametrize("name", [
    "Concrete Chunk Sprue A.stl",         # a plate of the loose chunks beside it
    "Medium Pipe Straight Long 5x Sprue.stl",   # five copies of one pipe
    "Grass Tufts Sprue.stl",
    "Damaged Metal Containers Sprue.stl",
])
def test_a_sprue_is_the_packing_not_a_model(name):
    # A sprue lays several pieces (or five of one) on a raft, supports printed
    # in. All 21 in the 2026-09-09 walk sat in a directory that also indexed
    # the individual pieces they carry, 3-16 of them.
    assert skip(name)


@pytest.mark.parametrize("name", [
    "Concrete Chunk (6).stl",             # the loose piece the sprue carries
    "Medium Pipe Straight Long.stl",
    "Spruce Tree.stl",                    # "spruce" is not "sprue"
])
def test_the_pieces_a_sprue_carries_stay_indexed(name):
    assert not skip(name)


@pytest.mark.parametrize("name", [
    "75_Unsupported_AlphaAlm_Body.stl",   # DM Stash's scale spelling, file
    "75_Unsupported_Aimar_BodyMask.stl",
])
def test_a_75_prefix_is_a_scale_duplicate(name):
    # DM Stash writes the 75mm duplicate as a bare "75_" prefix; every such
    # file in the 2026-09-03 census had an exact 32_ sibling in the walk
    assert skip(name)


@pytest.mark.parametrize("name", [
    "32_Unsupported_AlphaAlm_Body.stl",   # the twin that stays
    "1975_Tank.stl",                      # contains "75_", not a scale variant
    "Beast Slayer_75 mm scale",           # spaced spelling: unique kit, kept
    "UndeadKnight_Bust 150mm_STL_V1",     # 150mm: unique kit, kept
])
def test_the_75_prefix_is_anchored_and_scale_kits_without_twins_stay(name):
    assert not skip(name)


@pytest.mark.parametrize("name", [
    "Standalone Weapons & Hands",         # the five directory spellings...
    "Standalone Weapons and Hands",
    "Minoc Standalone Weapons",
    "Standalone Hands & Weapons & Spells",
    "Standalone_Draugr_Shield.stl",       # ...and the loose accessory files
    "Standalone_Barbarian_2H_Axe (repaired).stl",
])
def test_standalone_accessories_are_not_models(name):
    assert skip(name)


def test_no_labelled_model_is_cut_by_the_vocabulary():
    """Ground truth must describe the collection the walk actually indexes.

    This is the guard the `75_` cut needed and did not have: commit ea1ef04
    stopped indexing DM Stash's `75_` prefix, and five entries in
    `up_axis_labels.json` quietly stopped being measurable — the holdout that
    "21/21" is quoted against had become 18 models, and nothing said so until
    somebody re-derived the walk by hand (2026-09-09). A label the vocabulary
    excludes is not a hard failure anywhere; it just silently leaves the
    denominator.

    Every path segment is checked, not only the filename, because `skip` is
    asked of directory names too while walking — a label under a pruned
    directory is just as unreachable."""
    labels = json.loads((REPO / "labels" / "up_axis_labels.json").read_text())["labels"]
    assert labels, "the labels file is the point of this test"
    cut = [l["path"] for l in labels
           if any(skip(part) for part in Path(l["path"]).parts)]
    assert not cut, (
        "these labelled models would not survive the collection walk:\n  "
        + "\n  ".join(cut))
