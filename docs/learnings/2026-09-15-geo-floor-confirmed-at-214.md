# geo_floor at 214: the v3 attenuation holds, and the table that hid it

*2026-09-15. `eval/geo_floor.py`, `eval/common.py`. embed-cache512,
`siglip2-so400m-patch16-512` resolved from the cache's run-params, four-view
orbit at 384 px, up-candidate tiles at 2048 px, gate margin 0.45.*

`GEO_FLOOR_POWER = 2` shipped with cache version v3 on the strength of 49
labelled models, where it fixed `32mm_Orguss_Head` and changed nothing else.
That is a thin basis for a term in the production ensemble, and
`OPEN_QUESTIONS.md` has carried it as unfinished since. On 214 models it now
has an answer: **the choice was right, and the old set could not have shown
it.**

## The numbers

| scheme | orig 22 | holdout 20 | expand 121 | expand2 38 | reclaim 13 | orig+hold 42 | **all but hard 214** | hard 5 | escalates |
|---|---|---|---|---|---|---|---|---|---|
| unweighted (pre-v3) | 20/22 | 18/20 | 112/121 | 33/38 | 12/13 | 38/42 | **195/214** | 4/5 | 22/214 |
| floor 0.02, p=0.5 | 20/22 | 18/20 | 114/121 | 34/38 | 13/13 | 38/42 | **199/214** | 4/5 | 19/214 |
| floor 0.02, p=1 | 20/22 | 18/20 | 115/121 | 34/38 | 13/13 | 38/42 | **200/214** | 4/5 | 18/214 |
| **floor 0.02, p=2 (production)** | 20/22 | 18/20 | 115/121 | 35/38 | **13/13** | 38/42 | **201/214** | **5/5** | **18/214** |
| hard switch (p=99) | 20/22 | 18/20 | 115/121 | 35/38 | 13/13 | 38/42 | 201/214 | 5/5 | 19/214 |

**+6 models over pre-v3, and it escalates *less*** — 18 gated against 22. That
is the combination this harness exists to look for: the docstring's own test is
that "a scheme that buys a model by escalating twice as often has not bought
anything", and this buys six while asking the arbiter four fewer questions.

The curve is monotone from p=0.5 to p=2 and **p=99 does no better**, which is
the reassuring shape: the effect is the attenuation, not a threshold artifact,
and the shipped comment's claim that "the vote has to be nearly silenced before
it stops overriding SigLIP" is visible in the weights — the fixes land at
w=0.01–0.14.

At p=2 it moves 10 models: **8 fixed, 1 broken, 1 wrong either way.** The
regression is `32mm_ChitinousWatcher` (`+Y` → `+Z`, w=0.02). `32mm_Orguss_Head`,
the case the shipped comment names, is fixed at w=0.14 — the same number the
comment quotes, three months and 165 labels later.

## The old set could not have shown this

**`orig`, `holdout` and `orig+hold` do not move at all.** Every one of the six
gains is in `expand`, `expand2` or `reclaim`. Measured on the 42 models the
question was originally asked against, p=2 and pre-v3 are indistinguishable —
which is exactly why this sat open from 49 labels, and why the answer arrived
only after the set reached 214.

## Where the gains live says why it works

`reclaim` goes 12/13 → 13/13 and `hard` 4/5 → 5/5. `reclaim` is flat slabs,
corrugated debris and near-symmetric solids — by construction the shapes with
no print base to find, so geometry votes confidently on evidence two orders of
magnitude below `ABS_SCORE_FLOOR` and min-max cannot see it. That is the exact
failure `geo_weight` was written for, isolated in a set made of nothing else.

It is also the population where the **VLM arbiter was a no-op** (2026-09-10:
gemini +0, GLM low −1, reading the sheets standalone at 9/13 and 5/13). So the
same shapes that $12/1k of arbiter could not fix are fixed for free by having
geometry vote more quietly. When the ensemble's inputs are wrong, a better
tie-breaker does not help; a quieter wrong input does.

## Two bugs, both "a number quietly changing what it is measured on"

* **`geo_floor.py` scored the wrong tower.** `BACKBONE` was hardcoded to
  `patch14-384`; embed-cache512 is `patch16-512`. Every number this harness
  ever produced described a different vision tower than production — in an
  experiment specifically about low-margin picks, which is what a tower swap
  moves. `pose_baseline.py` already carried a private `_production_model()`
  for this; two harnesses needing it makes it a repo property, so it is now
  `common.production_model()` and both call it.
* **The results table reported four sets and dropped 172 models.** Rows were
  hardcoded to `orig`/`holdout`/`orig+hold`/`hard` — 47 models — while the
  per-model movement list underneath covered all 214, so the first run showed
  "8 fixed, 1 broken" beside a headline that could not move. The escalation
  rate was also read on `orig+hold` while the accuracy beside it was not, which
  makes the one trade this harness exists to judge unjudgeable. Rows are
  derived from the label file now, `hard` stays out of both pooled rows, and
  escalation is read on the same population as the accuracy it sits beside.

This is the third harness in eight days with this disease —
`backbone_sweep.py` (2026-09-10), the arbiter harnesses filtering to a
44-model file (2026-09-10), and now this. The pattern is always a set list
written down when it was complete and never revisited; the fix is always to
derive it. **A hardcoded set list is a silent filter with a date on it.**

## What did *not* change

Nothing in `src/`. `GEO_FLOOR_POWER = 2` was already production and stays
production; this run is confirmation, not a proposal. The one number worth
carrying forward is that pre-v3 would cost **6 models and 4 extra arbiter
calls** on today's set, where on the 2026-06 set it cost one model and nothing
measurable.

Note the figures here are **not** comparable to `pose_baseline.py`'s
194/214: this harness runs the four-view orbit ensemble at 384 px with 2048 px
tiles, and `pose_baseline` runs the production six-tile arrangement at 512 px.
Same ensemble rule, different pixels — name the arrangement beside the number.
