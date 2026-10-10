# OpenAI's Decisions API as the pose arbiter: rejected

*2026-10-10. `eval/decisions_vlm.py`. No embedding cache. Contact sheets come
from `eval/out/sheets{256,512}/`: 2026-09-09 renders for the original
models, and today's renders from `/run/media/masa/STLLibrary` for the rest of
the 219 labels. The model is `gpt-6-luna` through
`POST https://api.openai.com/v1/decisions`, in public beta since 2026-10-06.
No comparison column was rerun. Gemini, gemma, haiku and sonnet on the
38-model set are the published `results-2026-08-12.json` predictions. On
the full set, gemini and GLM are the 2026-09-09/10 runs
(`gemini_206.json`, `glm_206.json`, `*_reclaim.json`), and the ensemble is
`pose_baseline.json` from 2026-09-10, the same file the Luna runs were scored
against.*

OpenAI's Decisions API answers developer-defined questions with probabilities
rather than text, the same shape as Jev ([2026-09-22](2026-09-22-jev-part-judge-on-the-triage.md)).
The question was whether that shape, with a calibrated confidence to gate on,
makes a cheap model a usable arbiter. It does not.

## Access

The API takes a Platform API key only. A ChatGPT/Codex subscription login
(`~/.codex/auth.json`, `auth_mode: chatgpt`) is an OAuth token for OpenAI's own
clients, and OpenAI's Codex auth docs say general API calls still need a
Platform key. Images must be inline base64 data URLs. Billing is input tokens
only, at $0.10/M.

## The published comparison set is 38, not 44

The harness scores the 44 models the published columns cover, joined on stem.
Five of them were re-anchored as `32_Unsupported_*` when the stale label paths
were fixed (2026-08-31), and `Damaged Metal Sheet Sprue` is no longer
labelled. That leaves **38 models: 21 `orig`, 17 `holdout`**. Every column
below, published ones included, is scored on the same 38. 19 of them fire
`needs_arbiter`.

## Two question shapes

- **`choice`**: one request per model. The sheet is the image, and the question
  is a `choice` over the tile numerals with the arbiter prompt's wording. It
  cannot fail to parse. The API also returns a `confidence` and a probability
  per tile.
- **`predicate`**: six requests per model. Each request sends one tile cropped
  from the sheet and asks "is this upright?". The answer is the tile with the
  highest probability, and that probability is the confidence. This tests
  whether reading the numerals off the sheet was the bottleneck.

## The numbers on the 38

| method | sheet | standalone | as arbiter (19 models) | pipeline |
|---|---|---|---|---|
| ensemble alone | — | 32/38 | — | 32/38 |
| Luna `choice` | 256 | 27/38 | +3 −4 | 31/38 |
| Luna `choice` | 512 | 23/38 | +1 −4 | 29/38 |
| Luna `predicate` | 256 | 24/38 | +0 −2 | 30/38 |
| Luna `predicate` | 512 | 18/38 | +0 −5 | 27/38 |
| haiku (published) | 512 | 20/38 | +0 −4 | 28/38 |
| sonnet (published) | 512 | 31/38 | +4 −1 | 35/38 |
| gemini-3.5-flash (published, default) | 512 | 37/38 | +4 −0 | 36/38 |

**Every Luna configuration makes the pipeline worse than having no arbiter.**
That matches gpt-5.6-luna's net −2 through OpenRouter chat
([2026-08-30](2026-08-30-arbiter-backends-sheets-presentations-effort.md)).
The structured endpoint did not rescue the model family.

`choice` picks ±X 8–9 times, and no model in the set is truly ±X. That is a
positional prior. `predicate` removes the numerals and does worse. It rescued
none of the 19 escalated models, so the numerals were not the bottleneck. A
larger sheet hurts both shapes.

A rerun of `choice` gave identical answers on all 76 calls.

## Confidence on the 38

| method | sheet | median confidence, right / wrong | AUROC |
|---|---|---|---|
| `choice` | 256 | 0.51 / 0.54 | 0.62 |
| `choice` | 512 | 0.63 / 0.32 | 0.89 |
| `predicate` | 256 | 0.62 / 0.88 | 0.47 |
| `predicate` | 512 | 0.59 / 0.38 | 0.54 |

Only `choice` at 512 has a confidence worth gating on. Accepting its answer
only when confidence ≥ 0.5 gives +1 −0 and a pipeline of 33/38: one model
better than the ensemble alone, and three worse than gemini. That threshold was
chosen on the same 38 rows it scores, so the +1 is an upper bound. The
confidence tells you when Luna is wrong, but Luna is seldom right on exactly
the models that escalate. `predicate` confidence carries no signal, and at
256 the wrong answers are more confident than the right ones.

## On the full labelled set

Both shapes at both sheet sizes, on all 219 labels (`--baselines
eval/out/pose_baseline.json`), with every request answered. `hard` and
`reclaim` are reported separately and never pooled.

|  | orig 22 | holdout 20 | expand 121 | expand2 38 | **all but hard 201** | hard 5 | reclaim 13 |
|---|---|---|---|---|---|---|---|
| ensemble | 20 | 16 | 113 | 34 | **183** | 3 | 11 |
| + gemini-3.5-flash | 19 | 19 | 115 | 36 | **189** | 3 | 11 |
| + GLM-5.3-flash low | 19 | 17 | 116 | 35 | **187** | 3 | 10 |
| + Luna `choice`@256 | 19 | 19 | 111 | 32 | **181** | 3 | 10 |
| + Luna `choice`@512 | 18 | 17 | 112 | 34 | **181** | 3 | 8 |
| + Luna `predicate`@256 | 17 | 16 | 100 | 34 | **167** | 3 | 8 |
| + Luna `predicate`@512 | 17 | 16 | 105 | 34 | **172** | 3 | 8 |

The gate fires on 50 of the 201. On those 50:

| arbiter | rescued | broke | net |
|---|---|---|---|
| gemini-3.5-flash | 11 | 5 | +6 |
| GLM-5.3-flash low | 6 | 4 | +2 |
| Luna `choice`@256 | 7 | 9 | −2 |
| Luna `choice`@512 | 6 | 8 | −2 |
| Luna `predicate`@256 | 3 | 19 | −16 |
| Luna `predicate`@512 | 5 | 16 | −11 |

The 38-model verdict holds at five times the sample: **every Luna
configuration makes the pipeline worse than no arbiter.** `predicate` is much
the worst. It rescues almost nothing and breaks a third of what it touches.
The gemini figure reproduces the +6 the 2026-09-10 benchmark reported.

Confidence also holds up, and it is the one encouraging number:

| method | standalone (201) | median confidence, right / wrong | AUROC | best gated pipeline |
|---|---|---|---|---|
| `choice`@256 | 152 | 0.72 / 0.33 | 0.80 | 183 (gate ≥ 0.3) |
| `choice`@512 | 157 | 0.70 / 0.27 | 0.85 | 184 (gate ≥ 0.8) |
| `predicate`@256 | 113 | 0.51 / 0.08 | 0.73 | 183 (gate ≥ 0.9) |
| `predicate`@512 | 105 | 0.13 / 0.06 | 0.63 | 183 (gate ≥ 0.95) |

Gating fixes the regressions and leaves almost nothing behind. The best
configuration, `choice`@512 gated at 0.8, is one model above the ensemble and
five below gemini. Its threshold was chosen on the rows it scores. On the
small set, the `predicate` confidence looked like noise. At 201 it carries
signal, but only because it is very low on the many tiles Luna gets wrong.

## Cost

Per arbitration: `choice` uses 565/1,717 input tokens at 256/512 and
`predicate` uses 1,446/2,598 across six requests. That is $0.02–0.09 for a
354-call collection run. Latency is 1.5–3.8 s per arbitration, not the
~150 ms quoted in press coverage. The test as a whole cost well under $0.05.

## What these numbers are not

- The comparison columns are runs from a month ago, not reruns. The arbiter
  is not proven deterministic (→ #13), so a gemini rerun could move a model or
  two. That is too small to close a five-model gap.
- The `predicate` tiles are crops of the sheet, so each still carries its red
  numeral, and the instruction says to ignore it.
- One prompt wording per shape. No prompt iteration was done.
