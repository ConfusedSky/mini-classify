# Widening the labelled set: 49 to 206, and what a name can and cannot say

*2026-09-09. `up_axis_labels.json`, `eval/pose_label_calibration.py`,
`eval/pose_label_expand.py`. Everything below is measured against
embed-cache512's walk (3,091 files, 3,070 after the sprue cut) and against
labels a human adjudicated.*

The labelled set was 49, of which 5 were hand-picked as hard, so every accuracy
figure in this repo rested on an honest holdout of 20. It is now **206**, with
a scoring population of **179** — about 9× — and the error bars roughly 3×
tighter.

## Two ways the set had already shrunk without saying so

**The walk stopped indexing five of them.** The `^75_` rule (commit `ea1ef04`)
cut DM Stash's scale duplicates; five labels pointed at those files, four of
them in the holdout. So "21/21 on the frozen holdout" had named an 18-model set
since 2026-09-03 and nothing said so. Four were re-pointed at their `32_`
twins, verified as the same mesh at a different scale — identical vertex
counts, identical scale-normalised bounding boxes, identical geometry pick, and
a contact sheet that reads the same.

**One was a duplicate.** `75_Unsupported_Wajoski_Wolf2`'s `32_` twin was
already labelled, so `orig` had carried the same wolf twice: every pooled
figure quoted on 44 models was really 43 distinct ones. The two meshes differ
by 27 vertices in 2.7M — invisible to byte-dedupe, exactly the scale-twin case
the 2026-09-03 census named.

`tests/test_naming.py::test_no_labelled_model_is_cut_by_the_vocabulary` is the
guard. It checks every path segment, since `skip` prunes directories too, and
it caught the third instance on its first run: adding `sprue` to the vocabulary
orphaned `Damaged Metal Sheet Sprue` out of the holdout the moment it landed.

## Calibrate the labeller before its proposals seed anything

`pose_label_calibration.py` renders the *already-labelled* models as the same
six-tile sheets the VLM arbiter sees, into a manifest carrying no label and no
geometry column, with the answers in a separate file only `--score` opens.

It exists because the arbiter tables make the spread impossible to ignore: on
these same 44 sheets gemini-3.5-flash reads 43 and haiku reads 20. "A model
looked at it" is not a quality claim.

This session's labeller scored **47/48 blind** — the miss a winged wyvern in
the `hard` set. That number is what justified pre-selecting its proposals at
all; at haiku's rate the right answer would have been to show the sheets bare.

## Roughly half of this collection cannot carry an up-axis label

The convention excludes models whose upright is genuinely undefined — loose
hands, wings, swords, pipes, pins. Against a human triage of 247 drawn models:
**123 parts, 124 models.** A 50% part rate in an unfiltered draw, well above
the 36% a filename scan predicted and the ~41% `part_probe.py` measured for
part-ness generally.

That is the number that governs how many sheets a target set costs, and it was
the single biggest surprise of the session.

## What a name can say, measured

Each signal scored on both sides of that triage, against the labels that
survived review:

| signal | catches parts | wrongly cuts models |
|---|---|---|
| `standalone` (already in the vocabulary) | **0**/128 | 0/119 |
| `_L`/`_R` token, or `part`+digits | 51/128 (40%) | 0/119 |
| lettered family of ≥2 in one folder | 20/128 (16%) | 0/119 |
| either of the last two | **66/128 (52%)** | **0/119** |
| weapon word (sword\|axe\|blade…) | 35/123 (28%) | 4/124 |
| body-part word (hand\|head\|torso…) | 25/123 (20%) | **8/124** |

Three things that table settles.

**`standalone` catches none of them.** It is Artisan Guild's word for a
*directory class*; what a random draw actually turns up is per-character parts
named `Apprentice_Dagger_R`, `Mephisto_Axe_L`, `Mars_Warlord_Right_Shoulder_Armour`.
Two different populations that barely overlap.

**A body-part word is the trap.** `Body`, `Head` and `Torso` are what DM Stash
and Artisan Guild call the *main sculpt* — `32_Unsupported_Tygrin_Body` is the
character. Eight models would have been cut.

**The lettered family is the signal that is not in the name.** A stem ending in
a bare capital whose own directory holds siblings lettered the same way:
`32mm_OdDracnesThrone_B` beside A, C and D, `32mm_DragonSkullManor_C` beside
seven more. It adds 15 parts the name alone misses, nearly all modular terrain.
It survives the obvious objection — modular *character* kits letter their
variants too (`Dragonpeak_Barbarian_B`) — only because this library gives each
variant its own directory. That is a fact about this collection's layout, not
about naming, and a library that packed variants together would need it
measured again.

0 of 119 is not 0%; at that sample the true rate could be ~2.5%. Both signals
filter the *labelling draw* and deliberately not `src.naming`: a paired weapon
or a spell effect is a real thing someone searches for, and unlike a sprue or a
`75_` twin it has no surviving sibling that carries it. "No defined upright" is
a statement about ground truth, not about the library.

**The filter halves the part rate; it does not remove it.** A draw with both
signals on still came back **38 models to 28 parts — 42%**, against 50%
unfiltered. Forecasting ~2% from "only one survivor is name-detectable" was
wrong, and wrong in the instructive direction: what the filter cannot see is
`Gnomish_Miner_Backpack`, `Rohr_Adapter`, `legs2`, `Experiment Tank (New)_Top`,
`MechGunslinger_Bust_Arm`.

## Two traps worth their own line

**A sprue the sprue rule cannot see.** `Me262_Kit` is a flat aircraft frame
with parts in a runner and "Me-262 Schwalbe" moulded into it. It is packing,
exactly what `sprue` was added to cut, and the rule never sees it because the
file is named `_Kit`. The vocabulary catches a word, not a shape.

**A component letter is not a left/right marker.** `32mm_OdDracnesThrone_B` is
one of A/B/C/D that assemble into one throne; no naming rule can see it, only
the directory can. `Apprentice_Grimoire_L` is the mirror case — it *does* carry
`_L` and the filter does match it; it reached a sheet only because seeds 909
and 5150 were drawn before the filter existed.

## The flag is worth more than the accuracy number

Proposals the labeller marked uncertain were shown with **nothing
pre-selected**, so the human decided them cold.

| pass | pre-selected, kept | flagged, matched the withheld guess |
|---|---|---|
| axes, batches 1–2 (n=85) | 71/72 (99%) | 9/13 |
| part-vs-model, batch 3 (n=66) | 53/54 (98%) | 9/12 |
| axes, batch 3 (n=38) | 30/30 (100%) | 6/8 |

Overall the labeller was right on ~94% of fresh models, against 98% on the
older set — an honest drop. What matters more: **5 of its 6 errors fell inside
the proposals it had flagged**, and the flagged categories are the ones
calibration predicted (winged and no-ground-contact shapes, and pieces where
two orientations both read as standing). A model that knows where it is weak
buys more than one that is slightly more accurate everywhere.

## Three bugs the tooling found by being built

* **`--exclude` never excluded anything.** Manifest paths are anchor-relative
  (`run/media/…`) and were being joined onto `collection_root`, producing
  `/run/media/STLLibrary/run/media/…`, which matched nothing. Batch 2 re-drew 6
  of batch 1's models and batch 3 re-drew 4 more.
* **A stem is not an identity.** `tail1`, `head`, `tile7`, `Naga_Blade_R` and
  `Minoc_Shield_L` each name different files in different directories. Page
  state, proposals and the merge all keyed on stem, so two models shared one
  answer. The page now keys on path; `merge` refuses outright when a stem names
  more than one file rather than guessing which the human meant.
* **Saved answers keyed by URL.** Successive pages are written to the same
  path, so one batch's answers merged into the next batch's page — seen live as
  a 67-model triage reporting "299 of 67 decided". Keyed by batch now.

## The index and the viewer disagree only in coordinates

model-browser's axis pill read `Y` where a contact sheet said `+Z`. Both are
right. `Pikachu_X_Kakashi` measures 63.1 × 62.3 × **100.0** with `+Z` the only
real flat base (0.0123 against ≤0.0055), so the file's up is `+Z`; that app
bakes `rotateX(-π/2)` into every STL on load and its `toSceneSpace` maps
`(x, y, z)` to `(x, z, -y)`, so file `+Z` is scene `y`.

| index (file coords) | model-browser spindle |
|---|---|
| `+Z` | `y` |
| `-Z` | `-y` |
| `+Y` | `-z` |
| `-Y` | `z` |
| `+X` | `x` |
| `-X` | `-x` |

`+Y` and `-Y` are the ones that do not look alike across the two, and `+Y` is
the second most common label here — so that is where a real mismatch would
appear. Filed as model-browser#8, because `OrbitAxis` defaults to `'y'` and
`+Z` resolves to `'y'`: for the commonest orientation the pill cannot
distinguish "resolved from the index" from "nothing stored".

## Where the set stands

206 labels: `orig` 22, `holdout` 20, `hard` 5, `expand` 121, `expand2` 38.
Every one resolves in the walk and survives the vocabulary.

The two new sets are **not interchangeable**, and their notes in the file say
so. `expand` is an unfiltered random draw whose part calls were the human's
from the start; `expand2` was drawn with both part signals on — so it samples
*whole models*, not the collection — and there the labeller proposed both the
part call and the axis, with the human adjudicating both. Pool them only while
saying which is which.
