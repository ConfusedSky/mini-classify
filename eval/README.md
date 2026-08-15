# eval/

Measurement harnesses for the up-axis pipeline. These are the scripts that
produced the numbers in `LEARNINGS.md`; they are exploratory quality, kept so
the measurements can be reproduced or extended rather than re-derived.

They import `pose` and `classify_stls` directly, so they always measure the
real code path. Ground truth is `../up_axis_labels.json`, loaded through
`common.load_labels()` — never re-derive labels from a random sample index,
because the directory walk grew 509 → 602 files mid-session and the same seed
no longer draws the same models.

Scratch output (renders, contact sheets, prediction dumps) goes to `eval/out/`,
which is gitignored. Set `EVAL_OUT` to keep runs apart.

## The scripts

| script | what it answers |
|---|---|
| `siglip_up.py N` | Sweeps SigLIP probe wordings over N random models. Renders the 6 up-candidate tiles, embeds, scores each probe set, saves contact sheets. This is how you test a new probe phrasing — the spread across wordings was 83% to 4%. |
| `tile_and_vlm.py` | Scores geometry / ensemble / ollama VLM against the labels, and sweeps up-candidate tile resolution (384/512/1024/2048). |
| `claude_vlm.py` | Runs the arbiter prompt through Claude models via the CLI (`--model haiku`/`sonnet`) and compares with gemma. Reads predictions from `out/preds.json`, so run `tile_and_vlm.py` first. |
| `gemini_vlm.py` | Runs the arbiter prompt through Gemini (3.5-flash / 2.5-flash / 2.5-pro) on Vertex AI, at both sheet sizes, and scores them beside the gemma/haiku/sonnet numbers. Also reports measured per-call tokens, latency, and $ per full-collection run. Self-contained: it builds its own sheets and reads the published predictions from `results-2026-08-12.json`. `--report-only` re-prints the tables from the last run's JSON. Auth is gcloud ADC. |
| `backbone_sweep.py` | Crosses SigLIP vision towers with the **source** render size of the up-candidate tiles (384/512/1024/2048), probes and combination frozen, re-embedding identical pixels. `siglip2-so400m-patch16-512` is worth +1 of 44 on accuracy but is identical at every source size, where `patch14-384` flips three models on render size alone — see LEARNINGS. Reports `orig`/`holdout`/`hard` separately and prints the label composition it ran on. |
| `tile_count.py` | Scores the ensemble at `UP_TILE_AZIMUTHS` 4 / 2 / 1 — the tile count `pose-embed` is proportional to (24 / 12 / 6). Checks the azimuth subset against `view_angles` instead of assuming it, replays recorded arbiter answers through the production margin gate, and takes `--source production` to score `render_up_candidate_grid`'s own pixels rather than the cached orbit tiles. On production pixels `n_az=2` changes no pick on 43 labels; `n_az=1` costs 2 of 40 and doubles arbiter firing. `--compare` reports how far the two pixel sources move the answer — and that the `orbit384x4` cache is stale for 39 of 43. |
| `front_first.py` | Tests finding the *front* first and letting it constrain up, against up-first and a 4-azimuth control, over all 24 (front ⊥ up) orientations — which cost the same six geometry uploads as today's six tiles. Front-first loses 9 of 44; the chosen front is perpendicular to the true up on only 38/49, so a fifth of the time it excludes the right answer outright. See LEARNINGS. |
| `geo_floor.py` | Tests scaling geometry's vote by how much print-base evidence it actually has (`w = min(1, best/floor)**p`), against production's unweighted min-max. p=2 fixes `32mm_Orguss_Head` and changes nothing else on 49 models. Distinct from `ensemble.py`'s absolute-scaled scheme, which replaced min-max and lost. |
| `arbiter_gate.py` | Sweeps a gate on the **ensemble's** margin (`top1−top2` of the combined vector) against the current geometry-confidence gate, scoring pipeline accuracy against how often it fires — one firing is one API call. Reuses every recorded VLM run, so it needs no API access. Margin < 0.4 matches the geometry gate's accuracy on 9 calls instead of 24. |
| `gauntlet.py` | Runs one label set (`--set hard`/`orig`/`holdout`/`all`) through every method at once — geometry, both backbones at every render size, and every arbiter (gemma, haiku, sonnet, three Gemini) at both sheet sizes — as a per-model table. Needs `ollama serve` for the gemma row and gcloud ADC for Gemini; it skips what it cannot reach. A *per-model* instrument, not an accuracy measurement. |
| `backbone_memory.py` | VRAM per SigLIP tower: params, image tokens, resident weights, peak allocated at a given batch. Run it with the GPU idle — `ollama` holding gemma4:26b (6.8 GB) makes every figure meaningless. |
| `ensemble.py` | Compares ways of combining the geometry and SigLIP score vectors — min-max, z-score, Borda, softmax, absolute-scaled, and a hard switch. Min-max wins, and `LEARNINGS.md` explains why that is not arbitrary. |
| `one_model.py` | Per-candidate scores for named meshes. Reach for this when one model behaves oddly. |
| `build_report.py` | Builds the standalone HTML failure report — truth tile beside each method's pick, grouped by failure mode. |
| `gold_upright.py` | Renders every label in the orientation it asserts — `rotation_to_z_up(label)`, 3 azimuths — into one self-contained HTML page, so the ground truth itself can be eyeballed. This is how you check a new label before trusting a number measured against it. `--html` rebuilds the page from existing renders. |
| `light_probe2.py`, `light_probe3.py` | Superseded. Compared fill-light strategies before `FILL_INTENSITY` landed; kept because they document how the indirect-light-as-fill decision was measured. |
| `renderer_open3d.py` | What Filament will and will not do with GPU memory: whether a mesh can be staged before `add_geometry` (no), whether a scene can be cleared without evicting it (yes — `show_geometry`, ~11× cheaper than remove+re-add), and which card it runs on (the iGPU, so "resident" is host RAM). Feeds `docs/actor-refactor/renderer_alternatives.md`. |
| `load_path.py` | Where the time before the first pixel goes: parse, weld, upload. Answers "should we weld STL vertices at load" (no — the weld costs more than the upload it saves, and it shades smooth where soup shades flat) and finds the thing that actually dominates: `read_triangle_mesh` is ~15–30× slower than a numpy binary-STL parse, and upload is only 6.6% of the path. Prints a render-noise control row, because the renderer is not bit-exact and every pixel diff needs that floor to be read against. |
| `siglip_bench.py` | The SigLIP side's knobs, measured without moving the embeddings: batch sweep, preprocessing cost, overlap, drift, and `torch.compile` (phase `compile`, `COMPILE_MODES=default,max-autotune,reduce-overhead` — modes are A/B interleaved against eager per the thermal protocol; max-autotune's GEMM autotuner is gated off this 24-SM card). Fed the overlap/thermal LEARNINGS entries and the compile-mode numbers. |
| `score_precision.py` | Does the production scorer's fp16 cast (cached fp32 embeddings → fp16 for the GPU matmul) flip any classification? Scores all 2943 cached models three ways — production fp16-GPU, fp32 numpy, fp64 — under all three pool modes. Production pool: **1 flip in 2943**, on a 1.9e-05 margin between near-synonym categories; fp32 = fp64 everywhere. Max per-view sim delta is 6.2e-05 — 16× below torch.compile's 9.8e-04 embedding drift, and 124 models (4.2%) hold margins below that drift, which is the quantitative case behind compile's disqualification. |
| `compile_flips.py` | What torch.compile actually flips: pushes each model's 16 saved production views through eager and compiled SigLIP as identical preprocessed tensors (compile the *bound method* — wrapping the model silently stays eager, which a 0.0-drift first run exposed), scores both, reports flips by name. On the 241 worst-margin models plus 100 control: **1 flip**, at a 4.3e-06 margin between overlapping categories; drift median 7.3e-04, max 3.1e-03 (a longer tail than the recorded 9.8e-04). Even among the 44 models with margins below the drift, one flipped — direction matters, not just magnitude. Jpeg dilution is real: the flipped model's cached margin was 1.7e-03, its jpg margin 4.3e-06. |
| `compile_pose_flips.py` | The pose-side answer to review R1, bounded per T2: an eager **census** of live margins over all 2799 pose-cached models (reused from `out/pose_live_margins.json`), then compile-vs-eager on the 197 whose *live* margins sit within 0.022 of the 0.45 gate or of 0. **0 gate flips** in the 49 still in-band at compare time (rule-of-three ≈ 6% per in-band model, ~107 in-band collection-wide); **4 up flips**, all at margins ≤ 4.1e-03 — ties. Margin deltas median 1.5e-03, max 1.5e-02; the ~20× pose-vs-category amplification (`combine_up`'s min-max) stands. The census's side findings outweigh the bound: cached margins drifted median **0.119** (p90 0.354) from live under the unbumped `POSE_CACHE_VERSION`; run-to-run noise (median 2.66e-02) is **18× the compile delta** and exceeds the largest compile delta anywhere on 123 of 197 models — the ratio that makes the R1 acceptance categorical; and the in-band population is a range (107/133/385 at band/median-noise/p90-noise), not a point. |
| `render_determinism.py` | Which stage makes pose resolution nondeterministic (review U2), and whether the post-processing toggle fixes it (V1). Isolation: **Filament** — ~43% of pixels move on every repeat, the tower is bit-deterministic on identical tensors, and one tight-margin model flips its up *pick* between identical passes. The fix, run as arms per real model: `set_post_processing(False)` alone gives **7/7 byte-identical renders and 0.00 margin spread** after one throwaway frame (`noaa` unnecessary — even the AA-outlier model stabilises). The one-time cost of adopting it: margins shift median 1.3e-01 / max 3.5e-01, i.e. both cache version bumps plus an accuracy re-read. Needs the census from `compile_pose_flips.py`. |
| `ipc_spike.py` | What the render-child's queue actually costs in transport, isolated from render-wait: blasts the overlap spike's exact payload (24 tiles + 16 views, 17.7 MB/model) through `mp.Queue` vs a SharedMemory block pool with zero render time. Queue is ~13.5–14.3 ms/model (1.3 GB/s — the pipe, not pickle, is the tax); shm in-place is ~2.8–3.5 ms/model. Puts a ceiling on what shared memory can save the real pipeline. |
| `renderer_moderngl.py` | The alternative measured beside it. Selects the GPU by EGL device index — reaching the 4060, which Filament cannot — and checks the one that matters for the actor design: a second context in the same process, which Open3D core-dumps on. Needs its own venv, see the README header in the script. |

## Watch out

- **Contact sheet resolution changes the answer — for some models.**
  `pose.make_contact_sheet` now defaults to `thumb=512` (it was 256, which
  starved sonnet: 27/44 → 37/44 at 512, net −4 → net +3 as an arbiter). It
  barely touches Gemini: +2 for each of the three, with 3.5-flash returning an
  identical answer on 42 of 44 models across both sizes. State the sheet size in
  any VLM comparison, and the model in any sheet-size comparison.
- **Scaled numerals are part of the 512 px result**, and live in
  `pose.sheet_font` — `common.contact_sheet` delegates there rather than
  keeping a second copy. PIL's fixed ~11 px bitmap face is unreadable on a
  1536×1024 sheet, so a `thumb=512` without the scaling measures *worse* than
  256. If you write a sheet by hand, use `pose.make_contact_sheet`.
- **The sheet only scales the cell, never the tile.** `Image.thumbnail` does not
  enlarge, so tiles rendered below 512 px sit padded inside 512 px cells and the
  arbiter sees a smaller sheet than the number suggests. `classify_stls.py`
  warns at startup when `--render-size` is under `pose.SHEET_THUMB`.
- **Don't run the VLM and SigLIP against the same GPU.** On an 8 GB card they
  evict each other; a measured reload costs 10.1 s against 0.49 s of
  inference. Run the SigLIP phase, then the VLM phase — 40 calls took 112 s
  that way against a three-hour stall in the contended run.
- **`orig` labels are tuned, `holdout` labels are not.** The probes and the
  combination scheme were selected against `orig`. Quote pooled or holdout
  numbers; the ensemble scored 91% on `orig` and 81% on the holdout.
- **`hard` labels are neither — they were picked for being failure-prone.**
  Five models added by hand, so `pooled` is no longer a random sample and no
  longer means what it means in LEARNINGS, where every `n=44` predates them.
  `load_labels()` returns all 49; pass `"orig"` / `"holdout"` to reproduce a
  recorded number. Four have near-zero geometry scores — no print base — so
  they measure the arbiter, not the geometry. `PitFiend_Bust` is the exception
  (0.0678, a literal plinth) and is kept as a regression guard against an
  arbiter that overrides strong base evidence.
