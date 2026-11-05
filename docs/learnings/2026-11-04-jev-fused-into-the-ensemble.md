# Jev fused into the ensemble: ties gemini on the labels, loses on the blind split

*2026-11-04. Ran 2026-10-27 to 2026-11-04. `eval/jev_fusion.py`,
`jev_determinism.py`, `arbiter_agreement.py`, `model_traits.py`,
`trait_thresholds.py`, `arbiter_split_page.py`. Volume: STLLibrary. Ensemble
and gate are embed-cache512's backbone (`google/siglip2-so400m-patch16-512`)
over `eval/out/tiles512/`, regenerated into `pose_baseline_scores.json`; its
picks and gate match `pose_baseline.json` on all 219 labels (183/201, 50
gated). Sheets are `eval/out/sheets512/`. Jev is `jev-1.14.0`,
`jev-tiles-clean`, called directly. Gemini is `gemini-3.5-flash`. Its 189 is
the September run; its tile answers on the 1,000-model sample are new.*

The [Jev write-up](2026-10-20-jev-as-pose-arbiter.md) left one variant untested:
feeding Jev's per-tile probabilities into the ensemble instead of letting it
override. Fusion ties gemini on the 201 poolable labels and costs about 90–150×
less. On the only hand labels drawn from outside that set it goes 0 for 8
against gemini's 4. **Gemini stays the arbiter.**

## The fusion

`pose.combine_up`'s score gets a third term, `w · f(p)`, swept over `w`. Saved
probabilities from the two earlier runs were reused, so this cost no API calls.

| variant | w fitted on the same run | w from the other run | fitted on random halves |
|---|---|---|---|
| ensemble alone | 183 | | |
| `add`, `w · p` | 188 / 189 | 187 / 188 | 185.4 / 186.3 |
| `logp`, `w · log(p + 0.01)` | 190 / 191 | 190 / 191 | 187.4 / 188.7 |

- `logp` wins because a confident "no" pulls a candidate down harder than a
  "yes" pushes one up: 8–9 rescues against 1 break.
- Both runs pick `w` = 0.35, and 0.25–0.75 stays at 188–190.
- Every model it moves is under the 0.45 margin gate, and `logp-gated` scores
  the same, so Jev only needs calling on gated models.

## Ten runs of Jev

Eight more `jev-tiles-clean` runs on the 219 labels, ten in all, nothing else
changed. Scores are on the 201 poolable models.

| | min–max | mean |
|---|---|---|
| standalone | 179–186 | 183.1 |
| override, gated | 185–188 | 186.3 |
| fused, logp, `w` = 0.35 | 188–191 | 189.4 |

- Per-model confidence sd: median 0.022, p90 0.044, max 0.067. One tile's
  probability moves by up to 0.24.
- 19 of 219 models change answer. Most are 9–1 or 8–2 splits. `32mm_Gate_L`
  gives three answers, and four more split 6–4.
- AUROC of confidence against correctness is 0.87–0.93 in every run.
- The 187 quoted for override in October was a good draw: its mean is 186.3.
- The eight new runs had no part in choosing `w`, and they score 188–190. That
  tests Jev's run-to-run noise, not new models.

## A thousand unlabelled models

`arbiter_agreement.py` drew 1,000 unlabelled models from embed-cache512's pose
cache (seed 0) and asked both arbiters about the gated ones. Unlike the
labelled draws, this one has no part filter, and the gate fired on 512 (51%,
against 26% of the 219 labels).

Of the 512, gemini agrees with fused Jev on 270 (53%), with Jev alone on 244
(48%) and with the ensemble on 227 (44%). Fused Jev agrees with the ensemble on
309 (60%). All answered. That left 242 disagreements.

## Cost and speed

| on the 512 gated models | median | p90 | max | per 354-call run |
|---|---|---|---|---|
| gemini-3.5-flash | 11.2 s | 29.2 s | 267 s | $10.43 |
| Jev | 1.1 s | 4.1 s | 22 s | $0.07 |

The sample cost $15.08, about 2.5× the estimate made from a 4-call smoke run.
Gemini averaged 3,072 thinking tokens per call, against 872 on the labels on
2026-09-09 ($3.42 per run). `gemini_thinking_drift.py` reran production's prompt
on fresh draws to see whether gemini had started thinking more.

| thinking tokens | n | mean | median | p90 | max |
|---|---|---|---|---|---|
| labels, 2026-09-09 | 206 | 872 | 374 | 1,238 | 62,913 |
| labels, 2026-11-04 | 30 | 500 | 385 | 1,101 | 1,812 |
| gated sample, 2026-10-27 | 512 | 3,072 | 1,287 | 3,941 | 62,914 |
| gated sample, 2026-11-04 | 20 | 1,178 | 950 | 2,922 | 2,923 |

- **No drift.** The labels' median has not moved. Saved usage names no model,
  so this compares draws, not models.
- **The sample is harder:** its median is about three times the labels'.
- **The mean is set by a runaway tail.** 11 of 512 calls (2.1%) ran to about
  62,910 thinking tokens, taking about 200 s each, and cost $6.25 of the $15.08.
  One of the 206 label calls did the same. Without those calls the sample costs
  $0.0176 a call, or $6.24 per 354 calls, and the labels $2.46.
- The 20 and 30 model reruns hit no runaway, so they cannot say how often one
  occurs. Budget a full collection run from $6–10 per 354 calls.
- The 2026-09-01 rebuild's ~$11.5 is an estimate from August's per-call usage,
  not a bill, so it says nothing about what those calls cost.

## Two more questions

`model_traits.py` asked "Is the up axis ambiguous for this model?" and "Is this
a part of a larger model?". Jev answered on all 512 gated models and the 170
labelled models the triage called whole, both on tiles alone and with the
filename. Gemini answered on 429 of those, with sheet and filename: the 242
disagreements, the gated models the triage called parts, and the whole models.
Nothing in `src/` reads these answers. The yes/no wording uses the triage
page's own definitions of a part.

**Part**, against the hand triage. Only 27 parts have sheets, so these rates are
rough, and the whole models come from the labelled set, so they are easier
than gated models.

| | parts caught | whole called part |
|---|---|---|
| gemini | 27/27 | 29/169 |
| Jev + name ≥ 0.35 | 25/27 | 22/170 |
| Jev + name ≥ 0.50 | 23/27 | 14/170 |
| Jev tiles only ≥ 0.50 | 20/27 | 15/170 |
| `LIKELY_PART` | 11/27 | 0/170 |

Jev + name's AUROC is 0.946, and tiles alone 0.934, so most of the signal is in
the pixels. Held out on random halves, the best cutoff catches 86% at 12% false
alarms. At gemini's 29 false alarms Jev catches 25, so gemini is a little
better. Many of gemini's false alarms are mounted riders; many of Jev's are
modular terrain, some of it arguable.

**Ambiguous** has no hand truth. The stand-ins are the whole models, which a
person gave an up axis, and gemini, which is a reference and not ground truth.

| | matches gemini's 213 yes | yes on gemini's 216 no | yes on whole models |
|---|---|---|---|
| Jev + name ≥ 0.50 | 206 | 68 | 47/170 |
| Jev + name ≥ 0.60 | 189 | 39 | 28/170 |
| Jev + name ≥ 0.75 | 134 | 6 | 8/170 |
| gemini | | | 6/169 |

The two questions want different cutoffs, and neither is 0.5. Ambiguity barely
predicts disagreement at any cutoff: 77% of disagreements against 58% of
agreements at 0.60. Gemini says part on 229 of the 242 disagreements and Jev +
name on 184, so most of them have no right answer to find. A gemini call on
these questions costs about $0.0065, with 486 thinking tokens on average.

## The blind split

Taking Jev + name part < 0.35 and ambiguous < 0.75 left 24 disagreements. Masa
labelled them on a blind page (`labels/arbiter_split.json`): 16 no upright, 8
with one.

| right on the 8 | |
|---|---|
| gemini | 4 |
| ensemble alone | 2 |
| Jev alone | 1 |
| fused Jev | 0 |

- Gemini won 4 and fused Jev won 0. Both were wrong on the other 4.
- Fusion broke two models the ensemble had right, flipping each upside down:
  `Hive_Colossus_C` and `32mm_Crater`.
- 16 of the 24 that Jev's cutoffs passed have no upright, so Jev's "not a part,
  not ambiguous" did not mean much here.
- Gemini flags part or ambiguous on 13 of those 16 and on 3 of the 8 with an
  upright.

## What these numbers are not

- 4–0 happens by chance about one time in eight. All 8 are disagreements, so
  0/8 is not fused Jev's accuracy on anything; it says only who wins when the
  two differ on this set.
- The set was selected on Jev's own trait answers, which should if anything
  have favoured Jev. It is a sample of nothing and is never pooled.
- Every cutoff was picked on the models it is scored on, from one Jev run.
- The 189 for gemini is one September run, and the arbiter is not proven
  deterministic (→ #13). No gemini repeat was run here.
- One wording per question and no prompt iteration.
