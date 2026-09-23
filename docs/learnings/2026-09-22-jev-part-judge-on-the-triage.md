# Jev as a part filter: no better alone, +17 parts beside the regex

*2026-09-22. `eval/part_judge.py`, `eval/common.py` (`load_triage`). Names
only: no cache, no renders, no volume reads. Sibling lists come from
`embed-cache512/walk-e01f3c61998cfeef6b032f39d5910a26ac6b5ba1.json`, the walk
all three triage draws were taken from. Jev is `~typesafe/jev-latest` through
OpenRouter, which answered as `typesafe/jev-1.13-20260917`.*

The draw-time part filter in `pose_label_expand` (`LIKELY_PART` or
`lettered_family`) is two regexes and a sibling count, grown one naming
convention at a time. The question here was whether a judgment model reading
the same names could replace or extend it. Jev returns a calibrated probability
rather than text, so there is no reply to parse.

## Ground truth

`labels/triage_by_path.json` holds 305 paths from the model-or-part triage,
moved out of gitignored `eval/out/` in `db3030b`. `common.load_triage()`
applies the part-class pass on top: its 15 `undefined` answers on triage parts
become models, and `32mm_OdDracnesThrone_C` (a triage `keep` that pass called a
`component`) is dropped. That leaves **304 rows: 128 parts, 176 models.**

`label_expand3` (63 rows, 22 parts) was drawn with the regex filter on, so the
regex catches none of its parts by construction. The harness reports it apart
from the 241 unfiltered rows.

## What each judge is shown

Each judge sees the file's folders under the collection root, its stem, and up
to 80 other stems from its folder. It asks two Nouls and a Choice in one
request:

- `part`: is the file a loose piece of something larger?
- `no_upright`: does it have no upright a person would commit to? This is the
  triage page's own question ("a model — label its up axis" against "a loose
  part — no upright").
- `kind`: model, accessory or component.

The criteria give definitions only. None of them names a naming pattern,
because a judge told that `_L` means a part would just be the regex again,
scored on the data the regex was tuned on.

## The numbers

All 304 rows. "Parts caught" is out of 128 and "models cut" is out of 176.

| rule | parts caught | models cut |
|---|---|---|
| regex (today's filter) | 62 | 1 |
| Jev `part` ≥ 0.5 | 123 | 35 |
| Jev `no_upright` ≥ 0.5 | 97 | 27 |
| Jev `part` alone, ≤ 1 cut (t=0.91) | 47 | 1 |
| Jev `no_upright` alone, ≤ 1 cut (t=0.84) | 50 | 1 |
| **regex OR `no_upright` ≥ 0.85** | **79** | **1** |
| regex OR `part` ≥ 0.91 | 87 | 2 |
| regex OR `part` ≥ 0.88 | 96 | 5 |

**Used alone at a strict threshold, Jev does worse than the regex:** 47–50
against 62 parts caught, for the same one model cut. At 0.5 it catches nearly
every part, but it cuts a fifth of the models. **Combined with the regex, it
adds 17 parts at the regex's own cost of one model cut.** On `label_expand3` it
catches 12 of the 22 parts the regex missed, with one model cut
(`no_upright` ≥ 0.80).

On the triage's parts it also agrees with the part-class labels (accessory or
component) on 115 of 127, with no thresholds involved.

Cost: $0.0106 for 304 files, 16.7 s wall-clock at 8 workers.

## Why "separate piece" is the wrong question

Most of `part`'s wrong cuts at 0.5 are saddles, hats, hoods, helmets, pipe
fittings and modular wall sections. All of them *are* separate pieces, and
the triage kept them because each stands one way up. The labels answer "has an
upright", which is why `no_upright` holds one model cut at a lower threshold.
It is still not clean. Wall sections are labelled both ways across the two
passes: `32mm_Wall4A`/`4B` were kept, while `32mm_Wall5`/`9` are components.
Some of the remaining error is disagreement inside the labels.

The only model the regex cuts is `32mm_OdDracnesThrone_B`, a triage `keep` in
the lettered family whose `_C` the two passes disagree on. That family is the
same unresolved question, not a regex error.

## What these numbers are not

- **The thresholds are fitted to the rows they score.** Every "≤ N cut" row is
  the best threshold chosen on these same 304 rows, so the +17 is an upper
  bound. It becomes an operating point only after a fresh triage draw scored
  at a threshold fixed in advance.
- **Two prompts were tried.** Any further iteration on these labels would be
  fitting the prompt to them.
- **Runs are not identical.** Two runs of the same `part` question gave
  102/106 parts caught with 23/135 models cut, then 101/106 with 26/135 (on the
  unfiltered rows). The second run also asked `no_upright` in the same
  request, so the change may come from adding that question rather than from
  run-to-run noise. The two causes were not separated.
- The recorded 66/128 parts caught with 0/119 models cut (2026-09-09) and
  today's 62/128 with 1/176 are different populations: the corrections above
  changed the set, and `label_expand3` has since been added.
- The chat-model backend (`--backend openrouter`) exists but has not been run.
