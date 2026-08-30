# Arbiter backends, sheet sizes, presentations and effort — 2026-08-30

One question — *could GLM-5.3-Flash replace gemini-3.5-flash as the pose
arbiter?* — and everything it dragged in: sheet sizes past 512, f3d tiles
instead of the production ones, six images instead of one contact sheet,
reasoning effort up to `max`, gpt-5.6-luna, what temperature 0 actually
buys through OpenRouter, and a description-panel experiment that showed
where consensus stops meaning correctness. All arbiter numbers are on the
same 44 hand-labelled models, sheets and `UP_PROMPT` as every published
figure (`eval/gemini_vlm.py`'s tables, via `glm_vlm.py`, `f3d_arbiter.py`,
`o3d_solo.py`); the description work is `eval/description_panel/`.

Reading guide for the tables: *standalone* is the model's own accuracy on
all 44; *tier* is what matters — net rescues minus breaks on the 24 models
where `needs_arbiter` fires, and the pipeline total that results.
"Production tiles" are `rig.pose_sheet_tiles`, the Open3D pixels the
pipeline sends today.

## The answer: keep gemini-3.5-flash; GLM low is the fallback

| arbiter, production tiles | grid 256 | grid 512 | grid 1024 | solo 256 | solo 512 | solo 1024 |
|---|---|---|---|---|---|---|
| **gemini-3.5-flash** | 41 · +4 → 42 | **43 · +4 → 42** | 41 · +3 → 41 | 42 · +4 → 42 | 41 · +4 → 42 | 43 · +4 → 42 |
| GLM low | 39 · +2 → 40 | 38 · +0 → 38 | 37 · +0 → 38 | 37 · +1 → 39 | **40 · +3 → 41** | 40 · +2 → 40 |
| GLM high | 38 · +2 → 40 | 38 · +2 → 40 | 38 · +1 → 39 | — | — | — |
| GLM max (DeepInfra pin, 120 s cap) | 39/42 · +0 → 37 | 37/42 · +0 → 38 | — | — | — | — |
| gpt-5.6-luna low | — | **32 · −2 → 36** | — | — | — | — |

- **gemini-3.5-flash at 512 is unbeaten on every table.** Solo makes it
  robust to sheet size (+4 at all three) without making it better, and
  costs 2× (six images bill 6,646 input tokens; a sheet bills a flat 1,168
  whatever its size).
- **GLM low, solo, 512** is the best non-gemini configuration measured:
  40/44, +3 → 41/44, one model short of production — it rescues all four
  of gemini's rescues and its only break is `Concrete Chunk (6)`, the
  shapeless rock every non-gemini model "gets wrong". At $0.06 per
  354-call run against gemini's $2.7 (measured usage, list prices), 3 s
  median. Grid@256 low is the speed option: 1.3 s median, $0.02/run, +2.
- **1024 sheets are a dead end for both models** — gemini 43 → 41, GLM 39 →
  37, tier −1 each, gemini cost ×3 (1,168 → thinking 686/call, 7.5 s).
  Consistent with the repo's earlier finding that pose tiles are
  near-silhouettes: the information ran out before the pixels did.
- **gpt-5.6-luna is below every model but Haiku** and net *negative* as the
  tier — it would make the pipeline worse than no arbiter (36 vs the
  ensemble's 38). One 44-call pass was enough to close it.

## Effort: `max` loops on shapes that have no answer

GLM at `reasoning.effort: max` (pinned to DeepInfra, 120 s per-call cap) on
production grid sheets: standalone no better than low, tier **+0** at both
sizes (it breaks `Pressurized_Container_2`, which low never did), and **2 of
44 unanswered per pass** — `Floor` exhausted six 120 s attempts. Thinking
went from ~20 tokens (low) to 500–700 mean and 4,028 max among the calls
that answered; one solo call (`Body`) produced **17,584 output tokens** in
142 s and, having been abandoned client-side at 120 s, was completed and
billed server-side anyway. Wall clock per 44-model pass: 768 s and 1,002 s
against ~60 s at low. The tail is not provider noise — the stalls are the
flat slabs and headless torsos the label set already flags as pathological
(`Floor`, `Body`, `Psyflayer_Body`, `WisDevourer_Body`); at low effort the
model shrugs and answers in 2 s, at max it deliberates without a terminus.
The solo-max run was stopped after 14 calls and 16 abandonments.

Rules that fell out, now in the harnesses: **pin the provider** (below),
**enforce a wall-clock deadline from outside the socket read** (a helper
thread whose future is abandoned — `fetch_with_deadline`), and **treat a
deadline as the verdict, not an attempt**: the loop is reproducible, so a
retry re-runs it and re-bills it.

## Renderer and presentation

f3d tiles (AO+AA, `--config=none`, az 0 / el 20 like `pose_sheet_tiles`)
instead of the production Open3D tiles, on the 42 labelled models f3d can
load (`Psyflayer_Body` and `75_Benoit_BodyNoMask` refuse: "No files loaded"):

| f3d tiles | 256 | 512 | 1024 |
|---|---|---|---|
| gemini grid / solo | 36 / 36 | 38 / 37 | 39 / 40 |
| GLM low grid / solo | 31 / 34 | 33 / 32 | 30 / 33 |

**f3d costs every model 3–7 of 44** against production tiles, at every size
and presentation — the renderer that unlocked *description* (LEARNINGS
2026-08-29) hurts *uprightness*. The up-axis task is silhouette and ground
contact; the flat white production tiles are already ideal for it and AO
shadows add distractors. Three tasks (search, description, up-axis), three
different orderings of the same renderers. f3d does show a rising size
trend for gemini (36 → 40) where production peaks at 512 — its looser
framing needs more pixels to reach the same legibility.

Presentation: **no consistent effect for gemini (±1), a real one for GLM**
— solo lifts GLM by up to +3 standalone and turns its tier from +0/+2 to
+3 at 512 on production tiles, and stops it breaking the Wajoski wolves on
f3d. If GLM is the fallback, feed it six captioned images, not a sheet.

## Latency and cost, measured per call

| config | median s | p90 | max | think tok | $/run |
|---|---|---|---|---|---|
| GLM low · grid 256 | 1.3 | ~3 | 4 | 18 | 0.02 |
| GLM low · solo 512 | 3.0 | — | 10 | 25 | 0.06 |
| GLM max · grid 256 (answered only) | 11.2 | 35 | 90 (+120 s × 21 abandoned) | 718 | 0.08 + retries |
| gemini-3.5 · grid 512 (2026-08-12 run) | 24.4 | — | 80 | 638 | 2.68 |
| gemini-3.5 · solo 512 (today) | 4.2 | — | 20 | 496 | 5.13 |
| gemini-3.5 · grid 1024 | 5.9 | — | 34 | 686 | 2.83 |
| gemini-3.5 · solo 1024 | 5.8 | — | 215 | 2,133 | 10.35 |

Two things to carry: GLM is 3–4× faster than gemini per call because it
barely thinks; and **Vertex latency drifts** — the same gemini model on the
same sheet shape ran 24 s mean on 2026-08-12 and 5–8 s today. Latency
decisions get re-measured when they are made, which the per-call `secs`
and `provider` now written into every usage row make free.

## OpenRouter: routing, providers, and what "hung" looked like

- **Auto-routing sent max-effort vision requests to providers that never
  finished them.** The first GLM-max attempt ran 46 minutes with no
  completions after the first ten; `ss -tnio` showed worker sockets that had
  last *sent* 10–46 minutes earlier while *receiving* a few bytes every
  ~150 ms — OpenRouter's keep-alive heartbeat, which defeats urllib's
  socket timeout indefinitely. Masa's generation log for the same window
  showed requests landing on Modal, Parasail and Together, the last
  returning **7 output tokens at `effort: max`** — providers do not agree
  on what effort means.
- **Pins are request-shape specific.** With image + strict JSON schema +
  reasoning, Relace, Z.AI and Novita answer `404 No endpoints found` though
  each advertises structured outputs; DeepInfra ($0.075/M) and Modal
  ($0.15/M) serve it. For plain text or description calls Z.AI works.
- **Provider health flips hour to hour.** BaseTen 429'd all afternoon (the
  `@preset/glm-fixed-provider` preset pinned it and produced nothing);
  DeepInfra 429'd in the morning sweep and was clean by 15:00, then
  throttled intermittently under 4 workers (29 429s in 88 calls, all
  retried successfully with backoff).
- Harness contract now: `OR_MODEL`, `OR_TAG`, `OR_PROVIDER` (pin, no
  fallbacks), `OR_DEADLINE` (default 120 s), 429s printed not swallowed,
  provider and attempt count in every usage row, long runs `tee`'d to
  `eval/out/logs/` — a run piped through `tail` is invisible until it ends,
  which cost an afternoon of "is it broken?".

## Temperature 0 is a distribution, and pinning does not change it

`repeat_control.py`: the description control cell (one call, four views,
temp 0) 20× per renderer × effort, GLM-5.3-Flash, first under default
routing and then pinned to Z.AI.

| cell | unique texts /20 (default → Z.AI) | grade histogram (default → Z.AI) |
|---|---|---|
| f3d-1024 · low | 20 → 20 | ✅19 ✗1 → ✅18 ✗2 |
| f3d-1024 · high | 20 → 20 | ✅18 ◐2 → ✅18 ✗2 |
| f3d 512 · low / high | 7 / 13 → 7 / 14 | ✅20 → ✅20 (both) |
| three_ao · high | 16 → 10 | ✅20 → ✅19 ◐1 |
| three_ao · low | 9 → 10 | ✗15 ✅4 ◐1 → ✗16 ✅3 ◐1 |
| open3d_tight · high | 14 → 12 | ◐11 ✗4 ✅5 → ◐14 ✗4 ✅2 |
| open3d_tight · low | 7 → 12 | ✗13 ◐7 → ✗12 ◐7 ✅1 |

(Rubric from 2026-08-29: ✅ a separate alien erupting from the host's head or
chest; ◐ the host "transforming"; ✗ neither.) Twenty different essays in
twenty temp-0 calls at 1024 px on a single pinned endpoint: the
nondeterminism is in the serving, not the router, and no preset fixes it.
But the *grade distributions* replicate across providers almost exactly —
f3d 512 is ✅ 20/20 on both, `open3d_tight` never reaches a ✅ majority on
either. The per-renderer reliability profile is a property of the model.
Corollary: every single-call cell in any of today's matrices is one draw
from these histograms — the panel matrix's `three_ao` control showed ✅ and
is really ✅ 4/20.

## The description panel: consensus is not correctness

`panel_matrix.py`: 4 renderers (`open3d_tight` 512, `three_ao` 512, f3d 512,
f3d 1024) × (control · temperature panel: 3 describers at 0.8 · viewset
panel: 3 describers on different view packs, one with a head close-up) ×
effort low/high; each panel synthesized text-only by GLM **and**
gemini-3.6-flash. Describers GLM-5.3-Flash, then gpt-5.6-luna.

GLM ($0.019 + ~$0.02 of Gemini synthesis, 139 s): controls ✅ 3/8; panel
syntheses ✅ 10–11/16 per arbiter. When two of three panelists see the
burst, both synthesizers pick it and the third's stray reading lands in
the `disagreements` field — arguably the most useful output for a human,
because it says *when not to trust* the description. `open3d_tight`
produced no ✅ in six cells: three confident wrong panelists synthesize into
a confident wrong consensus. Viewset diversity beat temperature diversity
on good pixels (f3d-1024 view panel ✅ 3/3 at both efforts) and was worst on
bad ones. High effort did not substitute for a second opinion. The two
synthesizers agreed on 14/16 cells and split the other two one each way.

gpt-5.6-luna ($0.098, 766 s): **56 of 56 describer calls read the model as a
"dragonborn / lizardfolk adventurer casting a spell"** — every renderer,
both efforts, including the 1024 px views where GLM and Gemini are ✅
18–20/20. It sees every detail (jumpsuit, utility belt, tendrils, open
jaws) and assembles them into a fantasy race. Both synthesizers dutifully
produce the consensus. The panel measures variance, and a systematic prior
has none: this is the clean counter-example to "three agree, so it is
right", and the reason a describer for a mixed sci-fi/fantasy library
should be checked on a model whose genre is not the library's mode.

## Housekeeping found on the way

- **Six of the 49 up-axis label paths are stale**: Loot's Orconspiracy set
  moved into `Enemies_Part1/2/4_V2`. `f3d_arbiter.resolve` re-finds them by
  filename; `up_axis_labels.json` and `load_labels` are untouched
  (OPEN_QUESTIONS).
- f3d cannot load four STLs across today's sets (`Psyflayer_Body`,
  `75_Benoit_BodyNoMask`, and the pilot's two DM Stash bodies) that Open3D
  reads fine — a reason on its own not to make f3d load-bearing.
- gemini tokenises a contact sheet by tile count (1,168 for six tiles at
  any size) and separate images per image; GLM tokenises by pixels
  (589/2,092/6,836 for grid 256/512/1024). Cost tables must say which.
