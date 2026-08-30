# What a VLM sees in our renders, and the renderer pilot — 2026-08-29

Started as "what is this STL a model of?" (Loot Studios' Juvenile
Proto-Osteotron 1: a crew member in a jumpsuit with an alien bursting out of
where the head was) and turned into a measurement of how much the pixels
limit what a vision model can say about a mesh — first for free-text
description with VLMs, then, since the same pixels feed SigLIP, a pilot for
search quality across renderers. The pilot's numbers are in the last section;
everything before it is settled.

Grading, used in every table below: ✅ = a *separate* alien erupting or
bursting from the host's head or chest, the thing actually being looked for;
◐ = the host "transforming" or "mutating" into an alien, or an alien seen
without a host — the right neighbourhood, a worse read; ✗ = neither.

All VLM calls went through the pose arbiter's path — Vertex `generateContent`
over raw HTTPS with gcloud ADC (`src/pose.py:_ask_gemini`'s endpoint, project
`mini-classify`) — so the numbers are comparable with the arbiter tables in
LEARNINGS 2026-08-12. GLM went through OpenRouter (`~/.config/openrouter/key`).

## The production tiles are not enough for a description

Contact sheet of the model's 16 cached views from **`embed-cache512`** (the
primary cache: 512 px, 8 azimuths × ±20°, the SigLIP-2 recipe), one
question: what is this a model of, 2–4 sentences. Temperature 0, no system
prompt.

| model | renders only | + file path |
|---|---|---|
| gemini-3.6-flash | "tentacle-faced alien or Cthulhuoid entity" in tactical clothing | reads the name; alien in a jumpsuit |
| gemini-3.5-flash | ◐ "infected or mutated humanoid… heavily mutated head" | ◐ "humanoid undergoing a grotesque alien mutation" |
| gemini-3.1-pro-preview | "shark-man" with a dorsal fin | reads the name; alien with a skull-like head |
| gemini-2.5-pro | "sci-fi dwarf… symbiotic creature fused to its head", Chaos Dwarf | ✅ "bursting from or fused with a body" |
| gemini-2.5-flash | "orc, dwarf, or similar warrior" | reads the name |

**0 of 5 recover the concept from the production tiles** — the concept
being a human crew member with an alien erupting from where the head was —
and every model resolves the ambiguous head differently: tentacle face,
mutated head, shark, symbiote, horned crest. With the path all five parrot
"Juvenile Proto-Osteotron / Loot Studios / Spacecraft-Crasher"; 2.5-pro
alone fuses the name with the pixels into "bursting from or fused with a
body", and 3.5-flash gets as far as a mutation. The path is the only thing
the collection's filenames reliably give a VLM, and it is usually less than
this one.

The first pass ran on the **retired `embed-cache2` tiles (384 px)**, with
the same result (0/5 renders-only; crocodile/shark hood, headless figure,
Genestealer psyker, horned helmet) — so 384 → 512 buys a VLM nothing here,
and the resolution that matters is the 1024 of the next section, not the
step the cache already took.

Two gotchas from the runs:

- **"16 headless renders" was read as renders of a headless figure** by
  3.1-pro (384 px run), and in the path condition it argued with "the
  prompt's headless description". Say *offscreen*. The rerun without the
  word moved 3.1-pro to bird-folk — no closer, so the word was not the
  cause, but it was a confound; the 512 px run never used it.
- **Image tokens differ 3× by model family**: the 3.x models billed ~1130
  prompt tokens for the 16-tile sheet, the 2.5 models ~3430 — identical at
  384 (3072×768) and 512 (4096×1024), so the sheet is tokenised by tile count
  and a higher-resolution sheet costs nothing extra. Thinking tokens
  (800–1600 per answer) dominate cost either way; the pro models spent 3× the
  time and 6–7× the money of the flashes and were not more accurate.

## Resolution flips it; camera control only sharpens it

Same question, four **1024 px f3d** views (front/right/back/left, elevation
10, AO + AA, white background, no grid), no path. Condition D adds a
`render(azimuth, elevation, zoom, vertical_focus)` tool via function calling,
max 6 calls.

| model | 512 px cache sheet | 1024 px f3d one-shot | + camera tool |
|---|---|---|---|
| gemini-3.6-flash | Cthulhuoid alien | ✅ "tentacles bursting through the open collar" | ✅ 3 renders — "head and upper chest have erupted into" the alien |
| gemini-3.5-flash | ◐ mutated humanoid | ✅ "serpentine alien… erupting from the neck" | ◐ 1 render — alien "replacing the host's head" |
| gemini-3.1-pro-preview | shark-man | ✅ "entity bursting from the collar" | ✅ 2 renders — "bursting from the neck and chest" |
| gemini-2.5-pro | Chaos Dwarf | ✅ "creature bursting from the figure's head" | ✅ 1 render — "bursting from the neck" |
| gemini-2.5-flash | orc/dwarf warrior | ◐ tentacle-mass head, reads the whole figure as alien | ◐ never called the tool |

**0/5 at 512 → 4/5 at 1024, from resolution alone** (the 3.x models bill ~1130 image tokens for either sheet, so it is not a cost). Every model that used the tool zoomed on
the head first (`az 0–45, focus 0.8–0.85, zoom 2–3`) — the right region — and
the extra frames bought detail, not identity — and once, a worse read:
3.5-flash's close-up answer slid from "erupting from the neck" to
"replacing the host's head". The one model that would have benefited did
not call it. Each tool turn re-sends every prior image, so cost
roughly doubles ($0.023–0.051 vs $0.003–0.022).

Function-calling notes, both measured: gemini-3.6-flash answers `400
"Requests ending with a model turn are not supported"` when the tool image is
a sibling `inlineData` part of the `functionResponse` turn; putting the image
inside `functionResponse.parts` (the 3.x multimodal shape) fixes it, and the
2.5/3.1/3.5 models accept either. Echoing thought parts back in the model
turn made 3.5-flash and 3.1-pro leak their chain of thought into the final
text.

## GLM-5.3-Flash gets it at every effort level, for a tenth of the price

`z-ai/glm-5.3-flash` on OpenRouter ($0.075/$0.25 per M), identical f3d
inputs and tool. `reasoning: {enabled: false}` is refused ("Reasoning is
mandatory for this endpoint"); low / medium / high all ran.

| effort | one-shot | agentic |
|---|---|---|
| low | ✅ 7.4 s, $0.0009 — "head has burst open into a tentacled alien creature"; *Half-Life headcrab* | ◐ 1 render, 35 s — "bursting/transforming into an alien creature" |
| medium | ✅ 23 s, $0.0005 — "head has burst open… beak-like alien entity erupting" | ✅ 1 render, 39 s — "eel-like creature emerging", headcrab |
| high | ✅ 11 s, $0.0005 — "eldritch creature erupting from the skull"; Lovecraftian framing, "trench coat" | ✅ 2 renders, 23 s — "alien head erupts from the collar", *Alien* |

5 ✅ and 1 ◐ of 6 — the same head-first camera instinct, 10–50× cheaper
than any Gemini, slower when agentic. The one ◐ is again an agentic run
drifting from "burst open" toward "transforming into". OpenRouter reported 0 reasoning tokens at low and medium and
only 112–240 at high, so effort is a weak knob on this endpoint; the cost
column is ±1 rounding of a very small number. GLM reaches for franchise
analogies where Gemini stays generic.

## Why none of this transfers to SigLIP directly — and what might

The VLM lever was resolution, and SigLIP cannot use it:
`siglip2-so400m-patch16-512` resizes every input to 512², which is already
`embed-cache512`'s render size. What the cache *does* leave on the table:

- **Framing.** `Renderer.views` orbits at `1.4 × ‖extent‖`
  (`src/renderer.py`), so a figure fills ~55–65% of the frame — ~300 px of
  model after the resize. A tight per-view fit gives ~1.7× the pixels on the
  mesh with no renderer change.
- **Shading.** One sun plus a world-fixed fill held at 10k so it does not
  crush blacks but still swings ~30/255 across azimuths; no AO. f3d's AO+AA
  and model-browser's GTAO look like the grey-resin previews SigLIP saw next
  to captions; ours look flatter.
- **Colour and background.** model-browser's rig has red (−X) and blue (+X)
  rim lights on a dark ground — a constant colour signal SigLIP was never
  told about, and a lever for the *text* side (describe the lighting in the
  template) as much as the image side.

Category classification was already known to be render-size sensitive where
pose is not (OPEN_QUESTIONS, 384 vs 2048: 8/8 poses, 7/8 top-1), and the
SigLIP-2/512 move that Masa found "noticeably better" confounded backbone
with render size. Nothing in the repo could score search quality — the gap
OPEN_QUESTIONS names as the prerequisite for any renderer swap.

## The pilot: six render arms, two prompt arms, a VLM judge

Sample: 301 models from `embed-cache512` — 12 filename-keyword-seeded per
query (144, so each query has candidates whatever the renderer does) + 156
random + the Proto-Osteotron. 16 views each at the cache's angles, 512², up
axis from `pose-cache.json`.

| arm | what |
|---|---|
| `open3d` | the cached production embeddings — control |
| `open3d_tight` | production `Renderer`, same rig, each view fitted to the rotated mesh's projected vertices (5% margin) |
| `f3d` | f3d 3.5 AO+AA, white background, f3d's own fit (loose) |
| `three_ao` | model-browser's thumbnail chain (RenderPass → GTAO → OutputPass, shipped rig, contact shadow) via a dev-only hook, headless Chromium on the iGPU, composited over zinc-900 |
| `three_noao` | same, GTAO pass disabled |
| `three_white` | same as `three_ao`, rim lights set to white |

Prompt arms: production `PROMPT_TEMPLATES`, and a "described" set that names
the render style (grey untextured render / red-and-blue rim lighting on a
dark background). 12 queries scored with `src.query` unchanged (softmax
pool), top-20 per (arm × prompt × query) pooled and judged **once per
(query, model)** by gemini-3.6-flash from two 1024 px f3d views — the model
this write-up just showed reading the concept at that resolution — blind to
which arm retrieved it; 0/1/2 relevance. Metrics P@10 (rel = 2) and nDCG@20.

What building it cost, so the next person does not pay it twice:

- **f3d's system config overrides `--up` for STLs.** `/etc/f3d/config.d/10_native.json`
  matches `*.{3ds,stl}` and sets `up: +Z`, and it wins over the command line.
  Every ±Y/±X model rendered lying down until `--config=none`. The judge
  renders and the f3d arm both carry the flag now.
- **model-browser's `OrbitAxis` is not the STL up vector.** `models.ts` bakes
  `rotateX(−π/2)` into every STL, so file `(x, y, z)` is scene `(x, z, −y)`:
  pose up `(0,−1,0)` is spindle `'z'`, `(0,0,1)` is `'y'`. `pose.ts:axisOf`
  documents exactly this; I found it by rendering the Osteotron and looking.
- **Headless Chromium reaches the iGPU** with `--ignore-gpu-blocklist
  --use-gl=angle --use-angle=gl` (ANGLE on the Radeon 780M). `--use-angle=vulkan`
  gives no WebGL2; no flags gives SwiftShader.
- **Throughput on the 780M**: three.js 3 variants × 16 views for 301 models
  in 483 s (~30 frames/s including base64 readback); Open3D tight ~6 min;
  f3d ~0.5 s/frame, 4 workers, 830 s. Embedding 4816 frames through the
  production `Embedder` on the 4060 took 341 s — PNG decode and compositing
  are the cost, not SigLIP.
- **f3d cannot load two DM Stash STLs** (`Benoit_BodyMask`,
  `Elenil_body`: "No files loaded") that Open3D reads fine. They embed as
  zeros in the f3d arm and the judge falls back to Open3D renders for them.
- **Vertex `429`s need real backoff, not polite retries** — already in
  LEARNINGS 2026-08-12, still true; 6 workers with 4·2ⁿ s waits was fine.

Harness: `sample.py`, `render_{f3d,open3d_tight}.py`, `render_three.mjs`,
`embed_arms.py`, `score.py`, `judge.py`, plus the model-browser hook
(`client/src/dev/pilot.ts`, an `opts` parameter on `renderThumbnail`). To land
under `eval/renderer_pilot/` with the results.

## Results: the renderer is not where search quality is

388 judgments (91 clearly relevant, 128 partial, 169 not), softmax pool,
`eval/renderer_pilot/out/`. nDCG@20 is the number to read — P@10 is capped
by prevalence in a 301-model sample (two dragons, two golems, two riders
exist, so those queries cannot score above 0.2 on any arm).

| arm | prompt | P@10 | nDCG@20 | queries better / worse than control | top-10 overlap with control |
|---|---|---|---|---|---|
| open3d (control) | production | 0.525 | 0.852 | — | 1.00 |
| open3d_tight | production | 0.525 | 0.870 | 7 / 3 | 0.85 |
| f3d AO | production | **0.550** | 0.877 | 6 / 3 | 0.83 |
| three_ao | production | 0.533 | 0.867 | 7 / 3 | 0.83 |
| three_noao | production | 0.533 | 0.872 | 7 / 2 | 0.82 |
| three_white | production | 0.533 | 0.873 | 6 / 4 | 0.83 |
| open3d | described | 0.500 | 0.855 | 6 / 5 | 0.88 |
| f3d AO | described | 0.533 | 0.876 | 9 / 3 | 0.82 |
| three_ao | described | 0.542 | **0.878** | 7 / 2 | 0.78 |
| three_white | described | 0.533 | 0.876 | 5 / 6 | 0.78 |

Read it as three findings, in decreasing confidence:

1. **Every alternative beats the production renders, and by about the same
   small amount.** +0.015 to +0.026 nDCG, 6–9 queries better against 2–4
   worse, whichever arm — tight framing alone, f3d's AO, or model-browser's
   chain with AO on, off, or rims neutralised. That the *direction* is
   uniform across six independent pixel recipes is the strongest signal
   here; the *size* is one or two rank swaps per query, and a paired sign
   test on 7/3 is not significant at n=12. A renderer swap is not the lever
   that would change what a user sees in the top ten.
2. **AO, rim colour, and background do not matter to SigLIP at this scale.**
   three_ao / three_noao / three_white sit within 0.006 of each other; the
   red/blue rims neither help nor hurt on colour-free queries. The concern
   that the rims inject a spurious colour signal is not supported by this
   query set — it did not contain colour words, so it is also not refuted.
3. **Describing the render in the prompt template does nothing
   consistent.** The three.js arms with "red and blue rim lighting on a
   dark background" move −0.007 / +0.011 / +0.003; the grey arms with
   "untextured grey render" move −0.001 / −0.005 / +0.003. The production
   templates are fine.

What the arms *do* change is the order: top-10 overlap with the control is
0.78–0.88, so a renderer swap reshuffles one or two of every ten results
without making them better. The reshuffle is largest on the queries where
the sample has few clear matches and many partials (stone golem 0.62–0.80
across arms, dragon 0.77–0.86), which is where a judge's 1-vs-2 calls
decide the number.

The judge itself read sensibly on a spot check — a cyclops rated 2 for
"troll", a golem's fist rated 1 as a component, a vampire in plate rated 1
for "knight in plate armor" — and the Osteotron was 0 for both queries it
surfaced under, correctly.

**The result survives a change of judge.** The same 388 pairs re-judged by
GLM-5.3-Flash (OpenRouter, effort low, same views and prompt,
`judge_glm.py`): exact agreement with Gemini 0.78, within one level 0.99,
quadratic kappa 0.80, agreement on "clearly relevant" 0.92 — right in the
band two human assessors land in. GLM is the slightly softer grader
(marginals 144/162/82 vs Gemini's 169/128/91), which lifts every nDCG by
~0.02 and drops every P@10 by ~0.03, uniformly. Under GLM the ordering
keeps its shape — every alternative still beats the production renders
(0.876 → 0.890–0.917), the three.js arms edge ahead of f3d, and Spearman
between the judges' twelve arm×prompt means is 0.71. The five two-level
disagreements are genuine judgment calls (is a hunched Dwarven "Mask"
sculpt a troll; is a troll's fist a golem component). Cross-judging cost
$0.18 (1.13 M tokens, 308 s, 0 errors) against the Gemini judge's ~$3.50 —
the next pilot should judge with GLM and spot-check with Gemini, not the
reverse.

What this does **not** settle: the effect of a better renderer on the
*coverage* of a query (recall past 20) and on `robust_z`'s weak-query
verdict, neither of which a top-20 judgement pool can see; and whether the
uniform small gain is real, which a 1000-model sample with ~4× the
judgements would answer for about $15. The framing arm is the cheap one to
take if anything is taken: it is inside `Renderer.views` and changes no
dependency — but it changes every embedding and pose margin, so it is a
cache-rebuild item (`docs/cache-rebuild.md`), not a patch.
