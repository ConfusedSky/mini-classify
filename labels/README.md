# `labels/` — the hand ground truth

Everything here is **human judgement**. None of it is derivable from the code,
the caches or the collection: each file cost a labelling session, and a rerun
would not reproduce it. That is the whole reason the directory exists — the
part classes lived in gitignored `eval/out/` until 2026-09-11, one `git clean`
away from being lost, while the write-ups that cite them were safely committed.

Read ground truth through `eval/common.py` (`LABELS_FILE`, `PART_CLASSES_FILE`,
`load_labels()`), never by re-deriving a path or a sample — the collection grew
mid-session once and the same seed stopped drawing the same models.

| file | what it is |
|---|---|
| `up_axis_labels.json` | 219 up-axis labels in six sets, plus `collection_root`, per-set `sets` notes and a `history` array of every correction made to the file. The only pose ground truth in the repo. |
| `part_classes.json` | The raw browser export of the part-class pass, **keyed by stem**, exactly as it left the page. Kept unedited as provenance. |
| `part_classes_by_path.json` | The same 148 decisions **keyed by path**, which is the one to read. |
| `triage.json`, `triage3.json` | The raw browser exports of the model-or-part triage (247 and 66 answers), **keyed by stem**, unedited. `keep` means a model; `undefined` means a loose part. |
| `triage_by_path.json` | The same answers **keyed by path**, resolved through the draw manifests (`draws`), plus each draw's seed and whether it was part-filtered. Six stems named two files and are listed under `ambiguous` rather than guessed. Read it through `common.load_triage()`. |

## The sets are not interchangeable

`up_axis_labels.json`'s own `sets` block says what each one is; the short
version, because pooling them wrongly has already produced two bad numbers:

* `orig` (22) — the first labels, and the set the probes were tuned against, so
  it scores optimistically for every method.
* `holdout` (20) — the honest small sample. Its "21/21" was a high-water mark;
  the same models score 80% today.
* `expand` (121) — an unfiltered random draw. A sample of **the collection**.
* `expand2` (38) — drawn with both part signals on, so a sample of the
  collection's **whole models**, not of the collection.
* `reclaim` (13) — the models an earlier part triage *wrongly cut*. An error
  set, deliberately biased toward the shapes that defeated that filter. **A
  sample of nothing**; quote it about flat and near-symmetric geometry only.
* `hard` (5) — hand-picked failures. Never pooled, by convention.

`eval/backbone_sweep.py` derives its rows from the file rather than a hardcoded
list, and excludes `hard` from both pooled rows.

## The triage is not the last word on part-ness

The part-class pass read the triage's parts again and sent 16 back as
`undefined`: those are models, and `common.load_triage()` applies that
correction. It also marked one triage `keep`, `32mm_OdDracnesThrone_C`, as a
`component`. Two human answers disagree there, so the loader leaves it out
rather than pick one.

`label_expand3` was drawn with the name-based part filter on, so its parts are
the ones the filter *missed*. Any rule measured on it looks worse than it is
on the collection; quote the unfiltered draws for a collection-wide rate.

## What the part classes mean

A triage first asked *model or loose part?*; this pass asked, of the parts,
which kind:

* `accessory` (56) — worn or carried: a backpack, a shield, a quiver.
* `component` (82) — one piece of a single assembly: a manor wall, a ramp
  plank, a numbered machine part.
* `undefined` (16) — **neither: not a part at all.** These were triage errors,
  and all 16 became the `reclaim` set above.

Six stems in that draw named two different files each, so 12 files share 6
answers and are **not** represented in `part_classes_by_path.json`; they still
need a class. That collision is why the picking page now keys on path.

The classes carry **no up axis**. Labelling their uprights was attempted and
parked on 2026-09-10 — most parts have none that a person will commit to — see
`OPEN_QUESTIONS.md` and the dated learnings entry. Until that is settled these
are class labels and nothing more.
