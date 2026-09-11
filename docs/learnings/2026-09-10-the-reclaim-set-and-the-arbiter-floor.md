# The reclaim set: where the arbiter has nothing to add

*2026-09-10, second session. `eval/pose_label_expand.py`, `eval/pose_baseline.py`,
`eval/gemini_vlm.py`, `eval/glm_vlm.py`. Production shape throughout:
embed-cache512, `siglip2-so400m-patch16-512`, up-candidate tiles at 512 px,
contact sheets at 512, margin gate 0.45.*

The labelled set reached **219**. The thirteen new labels are not a draw — they
are the models the 2026-09-09 part triage **wrongly cut**, recovered by a
class pass that asked a different question. That makes `reclaim` the error set
of an earlier filter, and it turns out to be the most informative thirteen
models in the file.

## The set that breaks SigLIP and the arbiter alike

| | orig 22 | holdout 20 | expand 121 | expand2 38 | **reclaim 13** | all but hard 214 | hard 5 |
|---|---|---|---|---|---|---|---|
| geometry alone | 77% | 85% | 86% | 87% | **85%** | 85% | 40% |
| SigLIP alone | 77% | 65% | 86% | 79% | **46%** | 79% | 60% |
| ensemble | 91% | 80% | 93% | 89% | **85%** | **91%** (194) | 60% |
| gate fires | 31% | 15% | 24% | 26% | **38%** | 25% | 60% |

The headline does not move: **91% on 214**, against 91% on 201. Thirteen models
were never going to shift it, and at this n they should not.

**SigLIP scores 46% on `reclaim`** — its worst result on any real set, against
79% pooled. Geometry is untroubled at 85%. That is coherent rather than noisy:
this set is *by construction* the shapes that defeated a name-and-shape filter —
flat planks, corrugated debris, near-symmetric columns and junction boxes,
modular terrain panels — so a vision tower has nothing to grip and the print
base is the only evidence left.

## The arbiter is a no-op here, and slightly worse than one

Five of the thirteen are gated. On those:

| | reclaim 13 |
|---|---|
| ensemble | 11/13 |
| + gemini-3.5-flash | **11/13** — rescued 0, broke 0, net **+0** |
| + GLM-5.3-flash low | **10/13** — rescued 0, broke 1, net **−1** |

The standalone column explains it. **gemini reads these sheets at 9/13 and GLM
at 5/13**, against 43/44 and ~41/44 on the original set. The tier meant to
correct the ensemble is *less accurate than the ensemble* on this shape class,
so the gate — which fires more here than anywhere else, 38% against 25% pooled
— is routing the hardest decisions to the worst-qualified decider.

The answer distribution is the tell, and it is printed for exactly this reason:

| | +Z | -Z | +Y | -Y | +X | -X |
|---|---|---|---|---|---|---|
| truth | 7 | 2 | 4 | 0 | 0 | 0 |
| gemini-3.5-flash | 5 | 2 | 3 | 1 | 1 | 1 |
| GLM low | 3 | 2 | 4 | 0 | 1 | **3** |

Ground truth uses `+X`, `-X` and `-Y` **zero times**; both models scatter picks
across all three. That is not a judgement being made badly, it is a guess. It
is the same weakness 2026-08-30 named for near-silhouette shapes and that
2026-09-10 saw in four of the six arbiter regressions (`tile9`, `Floor`,
`Leaking_Metal_Drum_`, `32mm_CreditMachine`) — here it is isolated in a set
made of nothing else.

**The caveat that has to travel with the number:** n=5 gated. "+0 and −1" is
three models from reading differently, and this is consistent with prior runs
rather than independent evidence. What the row supports is a claim about a
*shape class*, not a revision of the arbiter's pooled value.

**Cost this run:** gemini $12.12/1k calls (1168 in / 1144 thinking, 13.2 s per
call — above 2026-09-10's $9.67, on 13 calls); GLM low $0.17/1k.

## A set that is a sample of nothing has to say so

`merge` wrote every set's description as *"random draw from the collection
walk, seed N, excluding every path already labelled"*. For `reclaim` that is
false in the way that matters: it is biased toward precisely the shapes that
fooled the filter that produced it, and `backbone_sweep.py` pools sets into
"all but hard" on the assumption they are samples. A manifest can now carry its
own `note`, and `reclaim`'s says outright what it is. Quote its 46% as a fact
about flat and symmetric geometry; never as a fact about the collection.

## The human is the calibration, again

The thirteen labels came from a pass where the labeller proposed all sixteen
and the reader decided:

| | matched the reader |
|---|---|
| pre-selected (labeller confident) | **6/6** |
| flagged, guess withheld | **3/10** |

**Every one of the labeller's seven errors fell inside a flagged proposal** —
the third consecutive pass where that holds, and the cleanest split yet. The
flagged categories were the ones calibration predicted: near-symmetric solids
(`Minoc_Column_A`), flat slabs (`32mm_Wood`, `Damaged Roofing (2)`), panels
whose two mirror views are indistinguishable (`32mm_WallGateDamaged_Gate`), and
a winged shape with no ground contact (`gargoyle1`, proposed `+Y`, ruled `-Z`).

Three of sixteen the reader ruled to have **no defensible upright** and they
were left unlabelled — the same judgement that parked the parts pass. A set
assumed clean still turned out ~19% unanswerable.

## Three bugs, and one of them is the old one wearing a new coat

* **A stem is still not an identity, and the page was still creating
  collisions.** The 2026-09-09 fix taught `merge` to *refuse* an ambiguous
  stem; it never stopped the page from keying on one. The parts page asked
  about 161 files and collected 154 answers, because six stems — `tail1`,
  `tile7`, `head`, `Naga_Blade_R`, `Minoc_Shield_L_(side_grip)`,
  `Gnomish_Miner_Backpack` — each name two different files. Picks, proposals
  and seeds now key on path; stem keys are read only as a fallback for the two
  exports written before the fix. **Refusing a bad input is not the same as not
  producing it.**
* **`merge` then wrote that path into the `stem` field.** Switching the page's
  key without switching what `merge` stores gave all thirteen reclaim labels a
  full filesystem path as their stem, and `common.build_tiles` used it as a
  *filename* — the benchmark crashed on a missing directory before rendering a
  pixel. It takes `m["stem"]` from the manifest now. A loud crash, for once,
  rather than a quiet wrong number.
* **Every page downloaded `confirmed.json`.** Four passes produced four files
  with one name, renamed by hand, and a class export is indistinguishable from
  an axis export until something opens it. The download is now
  `confirmed-<mode>-<batch>.json` and the file carries its own `run` field.

**Still latent, deliberately not fixed:** `common.build_tiles` caches tiles as
`{stem}_up{i}.png`. All 219 labelled stems are distinct today so nothing is
corrupted, but the parts draw alone held six collisions — the first colliding
stem to be labelled will silently share another model's pixels and geometry.

## What this says about when to call the arbiter

The open question from this session's start was whether parts belong in the
benchmark at all. The reader's answer was that most parts have no defensible
upright (`OPEN_QUESTIONS.md`, parked), and `reclaim` now adds the measurement
that goes with it: **on the shapes where the gate fires hardest, the arbiter
adds nothing and can subtract.** The gate's margin is measuring "the tiers
disagree", not "a VLM can settle this", and those are different questions.

That points at a cheaper tier than a better arbiter — a gate that declines to
call when the shape has no up-axis evidence to read, rather than one that pays
$12/1k to have a model guess `-X`. The parts figures already sized the prize:
the gate fires on 66% of parts against 26% of models, a projected 61–72% of all
arbiter spend.
