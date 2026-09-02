# renderer_pilot — does the renderer move SigLIP search quality? (2026-08-29)

Six render arms × two prompt arms × 12 queries over a 301-model sample of
`embed-cache512`, scored with the production `src.query`, judged once per
(query, model) by gemini-3.6-flash from two 1024 px f3d views. Write-up and
numbers: LEARNINGS, "What a VLM sees in our renders, and the renderer pilot".

Order of operations (all from this directory, `.venv/bin/python`):

| step | script | writes |
|---|---|---|
| sample | `sample.py`, `add_model.py <path-substring>` | `sample.json`, `arm_open3d.npy` (the control, straight from the cache) |
| render | `render_open3d_tight.py [key]`, `render_f3d.py [workers] [key]`, `node render_three.mjs [key]` | `renders/<arm>/<key>_v<i>.png` |
| embed | `embed_arms.py [arm ...]` | `arm_<arm>.npy` (n, 16, 1152) |
| score | `score.py` | `rankings.json`, `candidates.json` |
| judge | `prerender.py` (optional), `judge.py` | `judgments.json` (resumable) |
| report | `score.py --report` | the table |

`render_three.mjs` needs model-browser's dev server on :5173 with the
dev-only hook installed. The hook was removed from model-browser's tree once
the pilot answered (the three.js arms sit in the same noise band as the rest
— LEARNINGS 2026-08-29); `model-browser-pilot.patch` is the complete diff
(`client/src/dev/pilot.ts`, a DEV-guarded install in `main.tsx`, and an
`opts` parameter on `renderThumbnail` for AO/rim-colour), so a rerun is
`git apply model-browser-pilot.patch` in that repo. It drives the Playwright Chromium under
`~/.cache/ms-playwright` with `--ignore-gpu-blocklist --use-gl=angle
--use-angle=gl`, which is what reaches the iGPU.

Two things that cost an afternoon, both now in the scripts: f3d must run with
`--config=none` or `/etc/f3d/config.d/10_native.json` forces `up: +Z` for
every STL; model-browser's spindle axis is the STL up vector under
`rotateX(-π/2)` — `(x, y, z) → (x, z, -y)` — so pose up `(0,-1,0)` is `'z'`.

Try your own queries against all arms side by side (needs the `arm_*.npy`
files beside the scripts — kept locally, gitignored — and the 4060 for the
text tower):

    .venv/bin/python eval/renderer_pilot/query_arms.py            # REPL
    .venv/bin/python eval/renderer_pilot/query_arms.py "wizard with a staff"

`out/` holds this run's sample, rankings, candidate pool, judgments, and the
Osteotron across the five rendered arms. The renders and `.npy` files are not
kept (≈25k PNGs); the scripts regenerate them in ~30 minutes.

`ev2_check.py` (2026-09-01) re-ranks the pilot's 12 queries before/after the
ev2 rebuild — before from the archived `arm_open3d.npy`, after from the live
rebuilt cache, one GLM judge for both sides — and delta-judges only pairs the
new ranking surfaces. The rebuild's write-up: LEARNINGS, "The ev2 rebuild".

`elev_check.py` (2026-09-01) asks whether the −20° elevation ring earns its
half of the cache: the same 12 queries from the live cache's rows, view
subsets sliced from the `(n, 16, 1152)` matrix (views 0–7 are the +20° ring
— elevation-major), delta-judging into `out/judgments_glm_elev.json`.
Dropping the ring costs −0.005 nDCG@20; write-up: LEARNINGS, "Elevation
rings, finally measured".
