"""What a model's filename says about it.

One vocabulary, shared by the collection walk (`src.cachedir.find_stls`) and
the archive unpacker (`unpack_models.py`), because they ask the same question
about the same names — "is this the plain model, or a variant we do not
classify?" — and had already drifted apart answering it separately.

The drift that prompted this: the walk's tag list carried no entry for LYCHEE or
CHITUBOX, catching those exports only when the name *also* said "supported".
`AlkaMyastan_32mm_LYCHEE.zip` is a real archive that does not, so its contents
would have entered the collection as a pre-supported duplicate of the model
beside it.

Matching is substring-based on the lowercased name, so a tag hits inside a
longer word. That is deliberate for `presupported`/`pre_supported`, and the
reason `unsupported` needs the special handling in `_searchable`.
"""
import re

# Pre-supported and slicer-specific exports: the same geometry with scaffolding
# printed in. LYCHEE and CHITUBOX name the slicer rather than the support, and
# do not reliably carry "supported" as well.
SUPPORT_TAGS = ("presupported", "pre-supported", "pre_supported", "supported",
                "lychee", "chitubox")

# Not a model in its own right: a bare base disc, a hollowed print variant,
# the 75mm duplicate of a model we already have at 32mm, a standalone
# accessory piece, or a sprue.
#
# A sprue is a print-convenience plate — several pieces (or five copies of
# one) laid out on a raft, usually with the supports printed in. It is the
# packing, not a model: of the 21 in the walk on 2026-09-09, every one sits in
# a directory that also indexes the individual pieces it carries, between 3
# and 16 of them ("Concrete Chunk Sprue A" beside ten loose Concrete Chunks;
# "Metal Containers Sprue" beside all ten drums and canisters; the "5x" pipe
# plates beside their single pipe). Classifying one asks SigLIP to describe a
# tray, and its embedding is a blend of pieces that are each already in the
# collection on their own.
#
# "standalone" is Artisan Guild's own word for the class —
# five directory spellings ("Standalone Weapons & Hands", "Minoc Standalone
# Weapons", "Standalone Hands & Weapons & Spells", ...) plus 80 loose
# "Standalone_<Weapon>.stl" files outside those dirs, and the 2026-09-03
# census found no occurrence anywhere in the library that is not an
# accessory. Its 202-file directory class was also the largest source of
# byte-identical duplicates (98 md5-identical groups collection-wide).
NON_MODEL_TAGS = ("base", "hollow", "75mm", "standalone", "sprue")

SKIP_TAGS = SUPPORT_TAGS + NON_MODEL_TAGS

# DM Stash spells the 75mm scale duplicate as a bare "75_" prefix —
# "75_Unsupported_AlphaAlm_Body.stl" beside its "32_" twin; 57 files in the
# 2026-09-03 census, every one with an exact 32_ sibling in the walk. A
# prefix match rather than a substring tag, deliberately: "1975_Tank.stl"
# would contain "75_" and is not a scale variant. (The spaced "75 mm scale"
# and "150mm" kits stay indexed on purpose — they have no 32mm twin, so the
# NON_MODEL_TAGS rationale does not cover them.)
SCALE_PREFIX = re.compile(r"^75_")


# Every way this collection says "no supports", which all contain the thing they
# negate and must be removed before the tags are matched. Observed spellings:
# "NoSupports", "No Supports" (with the space), "Unsupported", and the
# "/No Supports" path segment. The rest of the shape — "un"/"non"/"not", any
# separator or none, "support"/"supports"/"supported" — is covered because these
# names come from a dozen different sculptors and the spelling is not a standard.
#
# The lookbehind is what keeps the negation from being found inside an ordinary
# word: without it "Casino_Supported" contains "no_supported" and would be read
# as unsupported, which fails open and puts a supported model in the collection.
# A plain \b cannot do this job — underscore is a word character, so \b never
# fires at the "_u" in "32mm_Unsupported".
NOT_SUPPORTED = re.compile(r"(?<![a-z])(?:un|no[nt]?)[\s_-]*support(?:ed|s)?")


def _searchable(name):
    """Lowercased, with every spelling of "no supports" removed.

    Removing the negation wholesale, rather than special-casing each tag, means
    a new tag can be added to the vocabulary without having to reason about
    whether some way of saying "not that" contains it."""
    return NOT_SUPPORTED.sub("", name.lower())


def skip(name):
    """Should this file or directory stay out of the collection?

    Asked of STL filenames and of directory names while walking (pruning a
    directory is a large win on slow media), and of archive names when deciding
    what is worth unpacking. Passing an archive its own name works because these
    sets name a zip after the folder it contains."""
    return bool(SCALE_PREFIX.match(name)) or \
        any(t in _searchable(name) for t in SKIP_TAGS)
