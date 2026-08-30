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
dev-only hook installed (`model-browser-pilot-hook.ts` is a copy of
`client/src/dev/pilot.ts`; `renderThumbnail` grew an `opts` parameter for AO
and rim colour). It drives the Playwright Chromium under
`~/.cache/ms-playwright` with `--ignore-gpu-blocklist --use-gl=angle
--use-angle=gl`, which is what reaches the iGPU.

Two things that cost an afternoon, both now in the scripts: f3d must run with
`--config=none` or `/etc/f3d/config.d/10_native.json` forces `up: +Z` for
every STL; model-browser's spindle axis is the STL up vector under
`rotateX(-π/2)` — `(x, y, z) → (x, z, -y)` — so pose up `(0,-1,0)` is `'z'`.

`out/` holds this run's sample, rankings, candidate pool, judgments, and the
Osteotron across the five rendered arms. The renders and `.npy` files are not
kept (≈25k PNGs); the scripts regenerate them in ~30 minutes.
