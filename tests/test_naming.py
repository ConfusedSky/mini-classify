import pytest

from naming import SKIP_TAGS, skip


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
    from naming import NON_MODEL_TAGS, SUPPORT_TAGS
    assert SKIP_TAGS == SUPPORT_TAGS + NON_MODEL_TAGS
