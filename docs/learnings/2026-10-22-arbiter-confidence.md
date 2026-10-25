# Arbiter confidence: it ranks answers, but gating buys a strong arbiter nothing

*2026-10-22. `eval/arbiter_confidence.py --labels poolable`. No embedding
cache is read: the ensemble picks and the gate are embed-cache512's, through
`eval/out/pose_baseline.json` (below). Contact sheets come from `eval/out/sheets512/`; the solo and per-tile shapes crop the same
sheets with the numerals painted out (`decisions_vlm.tiles(blank=True)`).
Labels: the 201 poolable models (`orig`, `holdout`, `expand`, `expand2`;
never `hard` or `reclaim`). Ensemble picks and the gate come from
`eval/out/pose_baseline.json` (2026-09-10), the file the
[Jev](2026-10-20-jev-as-pose-arbiter.md) and
[Luna](2026-10-10-openai-decisions-api-as-pose-arbiter.md) tests scored
against. The ensemble alone gets 183/201, and the gate fires on 50.*

Every arbiter so far has been trusted whenever it answers. The question was
whether its confidence could say when not to trust it: override the ensemble
only when the arbiter is sure. Two kinds of confidence were tested across
three vendors.

- **`logprob`**: the production `pose.UP_PROMPT` and schema, unchanged. The
  confidence is the probability of the answer token.
- **`self`**: the schema gains a `confidence` field (0–1) that the model fills
  in.

Each was asked three ways:

- **`sheet`**: the production contact sheet, one call, pick a numeral.
- **`solo`**: the six tiles as separate images captioned "Tile i:", numerals
  painted out, one call.
- **`tile`**: one yes/no call per tile; the pick is the tile most likely
  upright. Run only on the 38-model pilot (below).

## Who returns logprobs

Few endpoints do, and none of the current arbiters.

- **Vertex** refuses them for `gemini-3.5-flash` and `gemini-3.6-flash`
  (400, "Logprobs is not supported for this model"). `gemini-2.5-flash` and
  `gemini-2.5-pro` return them.
- **GLM-5.3-Flash on OpenRouter** depends on the provider. DeepInfra,
  production's pin, returns none. Wafer returns a broken top-k list,
  CoreWeave a 400, Together none. Parasail works and was pinned for both GLM
  arms. Its logprobs include the reasoning, so the answer token is found
  after `</think>`.
- **Claude** exposes no logprobs, so it has the `self` arm only. It ran
  through the `claude` CLI on the subscription, which cannot set temperature
  and needs a one-line system prompt to replace its own agentic one.

## The numbers

Pipeline is 183 plus the arbiter's net on the 50 gated models, trusting
every answer. It is computed that way, not read off `--report-only`, so the
two unanswered non-gated rows (see the last section) do not shift it. AUROC
is over the 50 gated models only: the chance that a random right answer is
more confident than a random wrong one. Over all 201 it reads up to 0.15
higher in most cells, because the easy models are confident and right, but
lower in three. The medians are over all 201.

| model | arm | net sheet / solo | pipeline sheet / solo | gated AUROC sheet / solo | median conf right / wrong (solo, all 201) |
|---|---|---|---|---|---|
| gemini-3.6-flash | self | +6 / **+8** | 189 / **191** | 0.89 / 0.78 | 1.00 / 0.95 |
| gemini-3.5-flash | self | +7 / **+8** | 190 / **191** | 0.86 / 0.84 | 0.99 / 0.95 |
| claude-sonnet-5-5 | self | +6 / +7 | 189 / 190 | 0.83 / 0.72 | 0.93 / 0.50 |
| gemini-2.5-pro | self | +2 / +7 | 185 / 190 | 0.66 / 0.81 | 1.00 / 0.95 |
| gemini-2.5-pro | logprob | +4 / +6 | 187 / 189 | 0.76 / 0.86 | 1.00 / 1.00 |
| gemini-2.5-flash | logprob | +1 / +7 | 184 / 190 | 0.74 / 0.82 | 1.00 / 1.00 |
| gemini-2.5-flash | self | +0 / +5 | 183 / 188 | 0.66 / 0.63 | 1.00 / 0.95 |
| claude-opus-5-5 | self | +5 / +5 | 188 / 188 | 0.84 / 0.88 | 0.90 / 0.60 |
| z-ai/glm-5.3-flash | self | +2 / +6 | 185 / 189 | 0.83 / 0.79 | 0.95 / 0.85 |
| z-ai/glm-5.3-flash | logprob | +5 / +4 | 188 / 187 | 0.66 / 0.47 | 1.00 / 1.00 |
| claude-haiku-5-5 | self | +2 / +2 | 185 / 185 | 0.76 / 0.83 | 0.75 / 0.45 |

For reference, production (`gemini-3.5-flash`, sheet, no confidence field)
scored +6 and 189 against the same baseline.

**Logprobs saturate.** Every logprob cell has a median of 1.00 for right and
wrong answers alike. Gemini 2.5 thinks before it answers, so the answer
token is near certain whatever the answer. Its ranking is not quite empty
(gated AUROC 0.74–0.86), but it lives in the fourth to sixth decimal place. GLM's
logprobs rank at chance on solo (0.47).

**Self-reported confidence ranks, and Claude's has the widest spread.**
Sonnet and Opus put a wrong answer near 0.5–0.6 and a right one at 0.9.
Gemini and GLM compress everything into 0.85–1.00, so a wrong Gemini answer
still reads 0.95 and a wrong GLM one 0.85.

## Gating does not pay

The gate compared is "override the ensemble only when confidence ≥ t". The
best threshold for each cell was picked on the same 50 rows it is scored on,
so these are upper bounds:

| model | arm | trust (sheet / solo) | best fitted gate (sheet / solo) |
|---|---|---|---|
| gemini-3.6-flash | self | +6 / +8 | +7 / +8 |
| gemini-3.5-flash | self | +7 / +8 | +8 / +8 |
| claude-sonnet-5-5 | self | +6 / +7 | +7 / +7 |
| gemini-2.5-pro | self | +2 / +7 | +2 / +9 (solo at 0.95) |
| gemini-2.5-pro | logprob | +4 / +6 | +4 / +8 (solo at 0.9998) |
| gemini-2.5-flash | logprob | +1 / +7 | +5 (at 0.999998) / +9 (at 0.999995) |
| gemini-2.5-flash | self | +0 / +5 | +2 / +6 |
| claude-opus-5-5 | self | +5 / +5 | +6 / +6 |
| z-ai/glm-5.3-flash | self | +2 / +6 | +3 / +6 |
| z-ai/glm-5.3-flash | logprob | +5 / +4 | +6 / +4 |
| claude-haiku-5-5 | self | +2 / +2 | +3 / +6 (solo at 0.5) |

Even fitted, the gate adds 0–1 model for the strong arbiters. The larger
gains belong to weaker cells. The Gemini 2.5 logprob gains sit at
thresholds in the fourth to sixth decimal place, which will not survive a
model update. The self-report gains (Haiku solo +4 at 0.5, `gemini-2.5-pro`
solo +2 at 0.95) sit at ordinary thresholds, but they were fitted on 50 rows
with 6–11 wrong answers. Two fitted cells reach 192, one above the ungated
3.5/3.6-flash; a threshold chosen on the rows it scores is not evidence
that it would hold on new ones. There is little for a gate to catch. The arbiter is wrong on
5–13 of the 50 gated models, and about half of those are models the
ensemble also gets wrong, where skipping the override saves nothing. That
leaves 2–7 breaks per cell, 3–4 for the strong arbiters, and the
low-confidence answers that include them include rescues too.

## Showing the tiles separately helps a little

`solo` matches or beats `sheet` on net for every model except GLM's logprob
arm, by 0–6 models; the weak readers gain most (`gemini-2.5-flash` +5 and
+6, `gemini-2.5-pro` self +5). For the two current Gemini flash models it is +1
and +2, inside the noise at 50 gated models. That matches the
[2026-08-30 presentation sweep](2026-08-30-arbiter-backends-sheets-presentations-effort.md),
which found ±1 for gemini and a real gain for GLM on 44 labels. `solo`
changes two things at once, separate images and no numerals, so the Jev
comparison is `jev-tiles-clean` against `jev-sheet`, not the blanking alone.

It costs more on the current Gemini models. Six images bill 6,690 prompt
tokens per arbitration on 3.5- and 3.6-flash against 1,212 for the sheet,
5.5×. Gemini 2.5, GLM and Claude bill about the same either way (within
±15%), so the price is a property of the Gemini 3 image tokenizer, not of
the shape. Latency is about the same.

## Per tile is weak, slow and tie-prone

The 38-model pilot (the published 44 minus `hard` and `reclaim`, 9 gated)
ran all three shapes. One call per tile never beat `solo` standalone and
tied it once (`gemini-3.6-flash`, 35/38). `gemini-2.5-flash` logprob scored
21/38 against 34 for solo, and its self arm tied the top probability on 16
of 38 models, so list order picked the axis. `gemini-3.5-flash` took
122 s per arbitration. Two tile cells are incomplete: `gemini-3.5-flash`
answered 34 of 38 and GLM logprob 36. The full run left per-tile out.

## What these numbers are not

- One prompt wording per arm. Asking for a confidence field changes the
  schema the model answers, so the `self` cells are not the production
  arbiter with a number attached; for `gemini-3.5-flash` sheet they score +7
  against production's +6.
- One run per cell. Claude runs at a temperature the CLI does not expose,
  and the arbiter is not proven deterministic (→ #13).
- Two `gemini-3.5-flash` sheet answers are missing after repeated 429s
  (`Mars_Warlord_Standing_Right_Foot`, `32mm_Warthog`). Neither is gated, so
  every net and gated figure is complete.
- Claude ran on the subscription, which production cannot use; its latency
  (4–10 s per arbitration) is the CLI's, not the API's.
