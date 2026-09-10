# The pose benchmark on 206 labels: 94%, and what the small set could not show

*2026-09-10. `eval/pose_baseline.py`, `eval/backbone_sweep.py`,
`eval/gemini_vlm.py`, `eval/glm_vlm.py`. Production shape throughout:
`siglip2-so400m-patch16-512` (the backbone embed-cache512 was built with),
up-candidate tiles at 512 px, contact sheets at 512, margin gate 0.45.*

The labelled set reached 206 on 2026-09-09. This is the first run of the pose
pipeline against it, and the first time the arbiter's *cost* has been visible.

## The numbers

|  | orig 22 | holdout 20 | expand 121 | expand2 38 | orig+hold 42 | **all but hard 201** | hard 5 |
|---|---|---|---|---|---|---|---|
| geometry alone | 77% | 85% | 86% | 87% | 81% | 85% | 40% |
| SigLIP alone | 77% | 65% | 86% | 79% | 71% | 82% | 60% |
| ensemble | 91% | 80% | 93% | 89% | 86% | **91%** (183) | 60% |
| + gemini-3.5-flash | 86% | 95% | 95% | 95% | 90% | **94%** (189) | 60% |
| + GLM-5.3-flash low | 86% | 85% | 96% | 92% | 86% | **93%** (187) | 60% |

The headline moved very little — 86% → 91% for the ensemble, against a sample
five times larger. The old estimate was not wrong, only imprecise.

## The arbiter is +6 net, not +12

The gate fires on **53 of 206 (26%)**. On those, gemini-3.5-flash **rescued 12
and broke 6**. Five further errors sit outside the gate where no arbiter can
reach them, so a *perfect* arbiter tops out at 201/206 on this set.

The 44-model set could not show this. Only 9 of its models were gated, so the
tier reported +4 with no regressions visible at all. Six regressions is the
real cost of the tier, and four of them — `tile9`, `Floor`,
`Leaking_Metal_Drum_`, `32mm_CreditMachine` — are flat slabs and cylinders,
the near-silhouette shapes the 2026-08-30 arbiter comparison already named as
the weakness. The tier is still worth running; it is just worth running with
its price known.

## GLM low holds up at 5x the sample

+4 net against gemini's +6, at roughly a fiftieth of the cost — consistent
with 2026-08-30's "the measured fallback", and now measured on 201 models
rather than 44. One caveat that is not visible in the totals: **GLM left 6 of
the 53 gated models unanswered** at its 120 s deadline and they fall back to
the ensemble, so part of its smaller regression count is chances it never took.

Measured this run, gemini-3.5-flash @512: 1168 in / 872 thinking / 7 out
tokens, 8.6 s per call, **$9.67 per 1k calls — $3.42 for a 354-call
production run**.

## What the wider set says about the sets themselves

**The old holdout was pessimistic, not optimistic.** It scores 80% on the
ensemble, the lowest of the four real sets, where `expand` scores 93%. At
n=20 that is roughly ±9pp and never distinguishable — but the "21/21" that
used to be quoted was a small-sample high-water mark, and those same 20 models
score 80% today.

**The arbiter earns most exactly there**: 80% → 95%, its best set, which is
also where the ensemble is weakest. That is the shape the tier is supposed to
have.

**`orig` gets *worse* under either arbiter**, 91% → 86%. Small n, but it is
the set whose probes were tuned in, so the ensemble is unusually strong there
and the tier has more to lose than to gain.

## Where the ensemble's own errors live

Of the ensemble's 18 errors on the 201: **7 are models geometry had right and
the combination lost**, 3 SigLIP had, 8 neither tier had. Roughly 40% of
ensemble errors are min-max discarding a correct geometry vote — which is
precisely the question `geo_floor.py` was built to ask (weighting geometry by
how much print-base evidence it has; `p=2` fixed `32mm_Orguss_Head` and
changed nothing else on 49 models). On 201 models that experiment now has
enough signal to answer.

## Four bugs, all the same disease

Every one of these is a number quietly changing what it is measured on.

* **`rig.embedder()` defaults to `patch14-384`**, not the `patch16-512` that
  built embed-cache512. Taking that default scored a different backbone than
  production and moved 7 of 206 ensemble picks, all low-margin.
  `pose_baseline.py` now reads the backbone from the cache's own run-params,
  the way every entry point resolves cache identity. Two independent runs then
  agreed exactly, which is what caught it.
* **The arbiter harnesses filter to a 44-model file.** `load_baselines()`
  reads `results-2026-08-12.json`, and both harnesses do
  `if l["stem"] in base` — so a 206-label run would have scored the original 44
  and silently ignored 162. `pose_baseline.py` regenerates that file for any
  label set; `--baselines` points the harnesses at it, and a label with no
  baseline now prints a warning instead of vanishing.
* **`report()` raised `KeyError` after the calls were paid for.** It indexed
  the published file for its comparison columns and died at the reporting step
  with 206 answers already bought. Answers are written before the report, so
  neither run had to be repeated — but the fix is `.get()`.
* **Published columns were scored on a subset**, printing "pipeline 33/206"
  from 44 answers and 162 blanks. A column that does not cover every model
  being scored is now dropped, with a line saying so.

## `pose_baseline.py`

The production tiers over any label set — `up_axis_scores` →
`rank_up_scores` → `upright_scores` → `combine_up` → `needs_arbiter_margin` —
written as `{stem: {geometry, siglip, ensemble, margin, needs_arbiter}}`. It is
not a generalisation of the VLM harnesses: it never touches the network. What
it generalises is the frozen `results-2026-08-12.json`, which holds exactly
these fields for 44 models and which every arbiter harness silently filters to.

The same five-line tier composition also lives in `parser_gate.py::_pose`,
which takes a loader and a live renderer because it is an A/B on the STL
parser. `common.py` deliberately owns pixels and labels, not decisions, which
is why neither put it there.
