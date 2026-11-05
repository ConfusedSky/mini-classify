# Haiku 5.5 as the pose arbiter: better than old Haiku, still behind gemini

*2026-11-04. `eval/arbiter_confidence.py --arms plain`. No embedding cache is
read: the ensemble picks and the gate are embed-cache512's, through
`eval/out/pose_baseline.json` (2026-09-10), the file the
[confidence](2026-10-22-arbiter-confidence.md),
[Jev](2026-10-20-jev-as-pose-arbiter.md) and
[Luna](2026-10-10-openai-decisions-api-as-pose-arbiter.md) tests scored
against. Contact sheets come from `eval/out/sheets512/`. Labels: the 201
poolable models, never `hard` or `reclaim`. The ensemble alone gets 183/201,
and the gate fires on 50.*

`claude-haiku-5-5` was already in the confidence test, but only with a
`confidence` field added to the schema. This run asks it what production asks:
`pose.UP_PROMPT` and `{"tile": n}`, through a new `plain` arm. It answers
whether Haiku could replace `gemini-3.5-flash`.

**It cannot. Keep `gemini-3.5-flash`.** Haiku 5.5 is +2 to +4 as the arbiter,
185–187/201, against gemini's +6 and 189. Pipeline below is 183 plus the net
on the 50 gated models.

| arbiter | labels run | rescued / broke | net | pipeline | thinking tokens | s per call |
|---|---|---|---|---|---|---|
| Haiku, default effort, sheet | 201 | +7 / −5 | +2 | 185 | 276 | 4.5 |
| Haiku, default effort, solo | 201 | +8 / −5 | +3 | 186 | 255 | 5.6 |
| Haiku `low`, sheet | 50 gated | +10 / −6 | +4 | 187 | 246 | 3.6 |
| Haiku `medium`, sheet | 50 gated | +8 / −4 | +4 | 187 | 324 | 3.9 |
| Haiku `high`, sheet | 50 gated | +10 / −6 | +4 | 187 | 622 | 5.7 |
| `gemini-3.5-flash`, production (2026-10-22, not rerun) | 201 | | +6 | 189 | | |

Standalone on all 201, Haiku scores 177 on the sheet and 180 solo, below the
ensemble's 183, so it is only usable behind the gate.

## The gap is about the size of the noise

Default and `@medium` differ on 4 of the 50 gated models, +2 against +4. The
API's default for Haiku 5.5 is `medium`; if the CLI keeps it, those are two
runs of one setting. That was assumed, not checked. The four cells at
`low`, `medium`, `high` and default disagree on 10 of 50. Against that spread,
the 2-model gap from Haiku's best cell to gemini is not a measured difference,
but no Haiku cell reached 189 and gemini's rescues are the larger set.

## Effort does nothing useful, and `max` does not finish

`low`, `medium` and `high` all net +4, by different routes, so `low` is the
setting to use if Haiku is ever wanted. It is the cheapest and the fastest.

`max` was stopped partway. 35 calls hit the harness's 300 s timeout, and
`tile9`, `Floor`, `Bedienkonsole` and `Container_complete` gave up after six
attempts each, 30 minutes per model. It had answered 28 of the 50 when killed.
The partial cell was deleted, so no `max` accuracy is reported. This is the
same failure as [GLM at `max`](2026-08-30-arbiter-backends-sheets-presentations-effort.md):
the model loops on shapes that have no clear answer. Arbiter effort sweeps
stop at `high`; `xhigh` was not run.

## Where it differs from gemini

Comparing `@medium` with gemini's `self` sheet answers, the nearest stored
cell (+7, not production's +6):

- One rescue gemini misses: `FN2205AC06_32mm_GrofMortewielder`.
- Two breaks gemini avoids: `Ork6` and `Leaking_Metal_Drum_`.

`Floor` and `32mm_CreditMachine` break for both. Haiku is a weaker gemini
rather than a complement, as Jev was.

The `+X` tell from 2026-08-12 has faded but not gone. Haiku 5.5 picks `+X` 9
times on the sheet and 6 solo out of 201, where the truth is never `+X`. The
old Haiku picked it 12 times of 44. Scored on the 38 published models that
Haiku was −4 net, so the new one is a real step up, though the label sets
differ.

## What these numbers are not

- One run per cell, at a temperature the CLI does not expose.
- The effort sweep ran on the sheet and the 50 gated models only, so solo at
  each effort is unmeasured.
- Claude ran on the subscription, which production cannot use. Latency is the
  CLI's. The CLI reported $0.0004–0.0012 per call, which is not an API
  measurement: no Anthropic API key was available.
