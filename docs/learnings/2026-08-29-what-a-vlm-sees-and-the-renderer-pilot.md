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

## Results

*Pending — the embeddings and judge views were still running when this was
written. To be amended in place.*
