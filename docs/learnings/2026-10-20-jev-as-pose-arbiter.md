# Jev as the pose arbiter: ties GLM at 187, two short of gemini

*2026-10-20. `eval/decisions_vlm.py` (`--mode jev-*`). No embedding cache.
Contact sheets come from `eval/out/sheets512/` and `sheets256/`, built on
2026-09-09 and 2026-10-10. The model is `jev-1.14.0`, called directly at
`https://api.typesafe.ai/v1/systemone` with `TypeSafe-Danger-Zone-Images:
true`. Every run was scored against `eval/out/pose_baseline.json`
(2026-09-10), the same file as the [Luna test](2026-10-10-openai-decisions-api-as-pose-arbiter.md).
The gemini and GLM columns are the 2026-09-09/10 runs and were not rerun.*

Jev 1.14 (released 2026-10-14) accepts images. TypeSafe labels image
support "Dreamer-ready" rather than production-ready, and says the format
and accuracy may change between releases. The question was whether a
probability-returning decision model can do what Luna could not. Partly: Jev
is the first decision model to help as the arbiter, but it does not beat
gemini.

## OpenRouter drops the images

Through OpenRouter (`/api/v1/systemone` and `/api/alpha/decisions`, as
`typesafe/jev-1.14-20261014`), a request with an image in `assets` answers
normally and bills **exactly 290 input tokens, with or without the image**.
The direct API bills the same request 1,738 tokens, against 290 without the
image. Through OpenRouter, a six-tile question came back near uniform
(0.15–0.21, confidence 0.05). OpenRouter accepts the request and answers
without the image, and nothing in the response says so. `part_judge.py`'s
text-only Jev calls are unaffected. Image work needs a TypeSafe key
(`~/.config/typesafe/key`).

## Four ways to ask

All at sheet 512 unless noted. Jev rejects images under 280 px a side, so
tile crops from the 256 sheet cannot be sent.

- **`jev-sheet`**: the whole sheet in `state`, and a Choice over the tile
  numerals. This is the shape Luna was given.
- **`jev-noul`**: one request holding six Nouls, with one cropped tile in
  each question's instructions. The answer is the tile with the highest
  probability.
- **`jev-tiles`**: one Choice whose six options *are* the cropped tiles, so
  the model never has to read a numeral.
- **`jev-tiles-clean`**: `jev-tiles` with the red numeral painted out of each
  crop (`tiles(blank=True)`).

## The numbers

Pipeline accuracy, with the arbiter overriding the ensemble wherever
`needs_arbiter` fires. `hard` and `reclaim` are kept separate.

|  | orig 22 | holdout 20 | expand 121 | expand2 38 | **all but hard 201** | hard 5 | reclaim 13 |
|---|---|---|---|---|---|---|---|
| ensemble | 20 | 16 | 113 | 34 | **183** | 3 | 11 |
| + gemini-3.5-flash | 19 | 19 | 115 | 36 | **189** | 3 | 11 |
| + GLM-5.3-flash low | 19 | 17 | 116 | 35 | **187** | 3 | 10 |
| + Luna, best (2026-10-10) | 19 | 19 | 111 | 32 | **181** | 3 | 10 |
| + `jev-sheet`@256 | 18 | 18 | 113 | 33 | **182** | 4 | 9 |
| + `jev-sheet`@512 | 19 | 18 | 111 | 32 | **180** | 3 | 9 |
| + `jev-noul` | 19 | 19 | 110 | 36 | **184** | 4 | 9 |
| + `jev-tiles` run 1 / run 2 | 19 / 18 | 19 | 112 / 113 | 34 | **184 / 184** | 4 | 10 |
| + `jev-tiles-clean` run 1 / run 2 | 18 / 19 | 19 | 115 / 114 | 35 | **187 / 187** | 3 / 4 | 10 |

On the 50 gated models among the 201:

| arbiter | rescued | broke | net | standalone (201) | AUROC |
|---|---|---|---|---|---|
| gemini-3.5-flash | 11 | 5 | +6 | 190 | — |
| GLM-5.3-flash low | 6 | 4 | +2 | — | — |
| `jev-sheet`@512 | 6 | 9 | −3 | 163 | 0.90 |
| `jev-noul` | 9 | 8 | +1 | 168 | 0.56 |
| `jev-tiles` run 1 / run 2 | 9 / 8 | 8 / 7 | +1 / +1 | 181 / 180 | 0.92 / 0.94 |
| `jev-tiles-clean` run 1 / run 2 | 8 / 9 | 4 / 5 | **+4 / +4** | 183 / 184 | 0.91 / 0.90 |

**Showing Jev the tiles as images beats asking it to read a numeral off a
sheet**, by 18 models standalone. **Removing the numerals halves what it
breaks** (8 down to 4–5) without costing any rescues. With both changes, Jev
reaches 187 on both runs with no confidence threshold, so that figure is not
fitted to the rows it scores. It ties GLM and is two models short of gemini.

Gating on confidence no longer adds anything for `jev-tiles-clean`. No
threshold beats 0 on either run, even when fitted on the same rows. Gating
helped the numeral-reading shapes only because it suppressed their errors:
`jev-sheet`@512 rises from 180 to 187 at a 0.5 threshold, a fitted figure.
`jev-noul`'s confidence carries no information (median 0.68 when right, 0.69
when wrong), the same failure Luna's per-tile predicate showed.

## It fails where gemini fails

The models `jev-tiles-clean` rescued in both runs are 8 of gemini's 11. Of the
four it broke in both runs, three are gemini's own regressions: `Floor`,
`Leaking_Metal_Drum_` and `tile9`, the flat slabs and cylinders the
[2026-09-10 benchmark](2026-09-10-the-pose-benchmark-on-206.md) named. The
fourth is `32mm_LutbaRobotRider`. Gemini's extra rescues are
`32mm_HunterShark`, `32mm_LampreyWatcher_IndentedIris` and `Bedienkonsole`.
Jev is a weaker reader of the same evidence. It does not cover gemini's
blind spots, so pairing it with gemini as a second opinion has little to
offer.

## Run to run

Jev is not deterministic. Two identical yes/no calls on one tile gave 0.82
and 0.85. Between full runs, 8 of 219 answers changed for `jev-tiles` and 6
for `jev-tiles-clean`, and confidence moved by up to 0.14. Both pairs land on
the same pipeline score, so the 187 is stable, but which models it rescues is
not.

## Cost

About 4,900 input tokens per arbitration for the tile shapes and 1,500 for
the 256 sheet, at $0.042/M input and no output charge. That is $0.07 for a
354-call collection run, against GLM's $0.02 for the same 187. Latency is
1.9–3.5 s per arbitration.

## What these numbers are not

- One instruction wording per shape, and no prompt iteration. Image input is
  pre-production, and TypeSafe says its accuracy will change between
  releases.
- The gemini and GLM columns are a month old. The arbiter is not proven
  deterministic (→ #13), and gemini's 189 against Jev's 187 is a gap of two
  models.
- The blanking replaces any reddish pixel in the tile's top-left eighth. That
  is safe because the renders are grey, but it would not be for coloured
  renders.
- The ensemble-fusion variant was not run: feeding Jev's per-tile
  probabilities into the ensemble instead of letting it override.
  `pose_baseline.json` stores the ensemble's pick and margin but not its
  per-axis scores, so this needs `pose_baseline.py` to save them first.
