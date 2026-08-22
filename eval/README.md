# eval/

Measurement harnesses for the up-axis pipeline. These are the scripts that
produced the numbers in `LEARNINGS.md`; they are exploratory quality, kept so
the measurements can be reproduced or extended rather than re-derived.

## What "measures the real code path" means here

It used to mean less than it sounded like, so it is worth stating exactly.
As of 2026-08-18 (the eval-debt cleanup):

* **The pixels are production's.** Every render below goes through
  `eval/rig.py`, which builds a real `src.messages.RenderConfig` and a real
  `src.renderer.Renderer`. Pose tiles are `Renderer.pose_tiles`, the call the
  render child makes; classification views are `Renderer.views`, the
  rotated-copy path I11 settled. Meshes load through `src.loader.get`, numpy
  STL parser and all.
* **The embeddings are production's.** Every embed goes through
  `src.embedder.Embedder`, the same instance that would run in the pipeline.
  `tests/test_embedder.py`'s GPU test pins `rig.embed` bitwise against
  `Embedder.embed_tiles`/`embed_views`, so this is a checked claim, not a
  comment.
* **The maths is production's.** `src/pose.py` is imported, never
  reimplemented — `up_axis_scores`, `rank_up_scores`, `combine_up`,
  `needs_arbiter_margin`, `upright_scores`, `make_contact_sheet`.
* **The scheduling is not, and a harness result never speaks to it.**
  Production runs the Poser / Renderer / Embedder as actors across a process
  boundary (`src/driver.py`), with a pose cache, an admission window and an
  arbiter that parks files. A harness composes the same calls in one process,
  in one thread, in a loop. So a harness number is a statement about the
  ensemble, the probes, the gate and the pixels — never about ordering,
  queueing or caching. `overlap_spike.py` and `ipc_spike.py` are the
  exceptions that measure scheduling, and they build their own processes to
  do it.

What changed to make that true: the harnesses used to import `classify_stls`'
`make_renderer`, `render_up_candidate_grid`, `render_up_candidate_tiles`,
`render_views`, `resolve_up` and `embed_images` — a single-process
re-arrangement of the same maths kept in the CLI *for them*, which had
measurably drifted from what shipped. `eval/rig.py` replaced it with the
production objects; those six (plus `_shoot`/`_upload`) were deleted from
`classify_stls.py` in phase 2 (2026-08-18) and no longer exist anywhere.

**No harness imports `classify_stls` at all now.** What they used to reach for
through it, they import from the module that owns it:

| was | is |
|---|---|
| `classify_stls.add_cache_args` / `apply_run_params` / `cache_root` / `cache_key` / `load_file_list` / `embeds_dir` / `renders_dir` / `render_index` / `load_run_params` | `src.cachedir` |
| `classify_stls.render_key` / `DEFAULT_ELEVATIONS` / `cache_key_from_identity` | `src.identity` |
| `classify_stls.as_tensor` / `embed_raw` / `embed_texts` | `src.embedder` |
| `classify_stls.pool_sims` | `src.query` (via `src.done` until 2026-08-19) |
| `classify_stls.RENDER_FORMATS` and the camera/light constants | `src.renderer` |
| `classify_stls.view_angles` / `rotation_to_z_up` | `src.pose` (via `src.renderer` until 2026-08-19) |
| `classify_stls.read_binary_stl` / `load_mesh` | `src.loader` |
| `classify_stls.view_config` | `src.cachedir` |

The point of the table is that each of those names has exactly one home, and a
harness now names it. `classify_stls.py` imports from these modules like any
other consumer and exports nothing.

Two harness-only dependencies are worth knowing about, because both point at
something production no longer has:

* `common.ask_gemma` calls `pose._ask_ollama`, a **private** function for a
  backend production retired — `src/arbiter.py` ships gemini and claude. It
  exists so `gauntlet.py`'s recorded gemma column can be re-run. Never run it
  beside SigLIP (see "Watch out").
* `rig.embed_probe_texts` calls `Embedder._embed_raw`, also private, because
  the Embedder carries production's four prompt banks and no way to ask for a
  fifth. `pose_probe_sweep.py` needs arbitrary wordings; reaching for one
  private beats keeping a second text forward for it to measure.

## Rules of the road

* `eval/rig.py` holds every `OffscreenRenderer` for the process lifetime and
  **never destroys one** — Filament aborts from that destructor (CLAUDE.md).
  A script that renders must end with `rig.exit_without_teardown()`, not by
  falling off the end of `main()`. `views_camera_rotation.py` is the worked
  example.
* Ground truth is `../up_axis_labels.json`, loaded through
  `common.load_labels()` — never re-derived from a random sample index,
  because the directory walk grew 509 → 602 files mid-session and the same
  seed no longer draws the same models. The one script that samples a walk
  file, `pose_label_sheets.py`, scores nothing: it exists to reach models the
  labels do not cover.
* `src/pose.py` is `from src import pose`; a bare `import pose` will not
  resolve.
* Scratch output (renders, contact sheets, prediction dumps) goes to
  `eval/out/`, which is gitignored. Set `EVAL_OUT` to keep runs apart.

## The plumbing

| file | what it is |
|---|---|
| `rig.py` | The adapter. `rig(size, views, elevations)` → a cached, never-destroyed `Renderer`; `load`/`as_loaded` → `LoadedMesh`; `pose_tiles`/`pose_sheet_tiles`/`views` → the production render calls; `embedder`/`embed`/`embed_probe_texts` → the production `Embedder`; `exit_without_teardown` → the mandatory exit. Every function is a handful of lines of plumbing over a production call — if a harness wants behaviour production lacks, that is a finding, not a branch in here. Note the type change from the old CLI helpers: renders come back as `np.ndarray`, so anything that saves one needs `Image.fromarray`. |
| `common.py` | Labels (`load_labels`, `collection_root`), scoring helpers, the VLM callers (`ask_claude`, `ask_gemma`), and the three cached render sets every scorer reads: `build_tiles` (6 up-candidate tiles + geometry per label), `build_sheets` (contact sheets at a given thumb size), `build_orbit_tiles` (24 tiles, 6 ups × 4 azimuths — the rotate-the-mesh pixels every published azimuth number was measured on; **not** the pose tiles, see `tile_count.py --compare`). |

A harness that wants **cached** embeddings rather than fresh renders should
not open the load preamble by hand — `src/collection.py` is that preamble,
and `src/query.py` is the scoring:

| want | reach for |
|---|---|
| render or embed something now | `rig.py` (builds the production `Renderer`/`Embedder`) |
| the whole collection's cached vectors, its files, poses and scope filtering | `src.collection.Collection.load(args)` |
| pooling, robust z, ranking | `src.query` — `score`, `pool_sims`, `robust_z`, `rank` |
| just the `.npy` matrix | `src.embed_store.load_embedding_matrix` |

`Collection` costs no torch and no open3d, so a harness that only reads
cached vectors stays as cheap as `cluster_models.py` is. Both it and
`src/query.py` exist because the REPL and the query API needed the same
formulas; a third copy in a harness is the thing they were extracted to
prevent.

## The scripts

| script | what it answers |
|---|---|
| `pose_probe_sweep.py` | Which upright/toppled probe **wording** picks the right up-candidate tile, on the labelled set, with pixels and combination frozen — SigLIP alone and ensembled. `production` is a row, so the shipping answer is always the baseline. Takes `--models a,b`, which is how to attack OPEN_QUESTIONS' "would a much smaller tower do for pose?": `UPRIGHT_PROMPTS`/`TOPPLED_PROMPTS` were tuned for so400m, so a smaller tower must be read against *its own* best wording. The labelled half of the retired `siglip_up.py`. |
| `pose_label_sheets.py` | Contact sheets for models nobody has labelled yet — the input to widening the labelled set (OPEN_QUESTIONS' root bottleneck: 49 models, 5 hand-picked as hard, an honest holdout of 20). Samples a cache's walk file, renders the six tiles through `Renderer.pose_tiles`, writes one sheet per model plus an `index.json` with geometry's pick and the cached pose to speed hand-labelling. Walk file, cache dir, count and seed are all arguments — the hardcoded walk path is why the original stopped running. Deliberately does **not** call `load_labels()`. |
| `gold_upright.py` | Renders every label in the orientation it asserts — `Renderer.views` at the labelled up, 3 azimuths — into one self-contained HTML page, so the ground truth itself can be eyeballed. This is how you check a new label before trusting a number measured against it. `--html` rebuilds the page from existing renders. |
| `parser_gate.py` | Does the numpy STL parser change pose decisions on the labelled set? Runs the production pose path with only the loader changed, composing the three tiers in the harness out of the same `src/pose.py` functions the Poser calls, over `Renderer.pose_tiles` pixels and `Embedder` embeddings. Reports an **A/A control** beside the A/B: one shared renderer carries scene state, so zero variables changed still moves margins, and no margin-level claim here is attributable without that floor printed beside it. The same renderer nondeterminism `render_determinism.py` later isolated to Filament's post-processing. The only label-level gate on the STL loader. |
| `tile_count.py` | Scores the ensemble at `UP_TILE_AZIMUTHS` 4 / 2 / 1 — the tile count `pose-embed` is proportional to (24 / 12 / 6). Drives `Renderer.pose_tiles(..., n_az=...)`; that parameter exists for this harness, so the sweep sweeps the production call rather than a copy. Checks the azimuth subset against `view_angles` instead of assuming it, replays recorded arbiter answers through the production margin gate, and takes `--source production` to score the pose tiles' own pixels rather than `common.build_orbit_tiles`' cached orbit tiles. On production pixels `n_az=2` changes no pick on 43 labels; `n_az=1` costs 2 of 40 and doubles arbiter firing. `--compare` reports how far the two pixel sources move the answer — and that the `orbit384x4` cache is stale for 39 of 43. |
| `backbone_sweep.py` | Crosses SigLIP vision towers with the **source** render size of the up-candidate tiles (384/512/1024/2048), probes and combination frozen, re-embedding identical pixels. One `Embedder` per tower, so "probes frozen" is structural — the banks come off the tower with it. `siglip2-so400m-patch16-512` is worth +1 of 44 on accuracy but is identical at every source size, where `patch14-384` flips three models on render size alone — see LEARNINGS. Reports `orig`/`holdout`/`hard` separately and prints the label composition it ran on. |
| `geo_floor.py` | Tests scaling geometry's vote by how much print-base evidence it actually has (`w = min(1, best/floor)**p`), against production's unweighted min-max. p=2 fixes `32mm_Orguss_Head` and changes nothing else on 49 models. Distinct from the absolute-scaled scheme in the retired `ensemble.py`, which replaced min-max and lost. |
| `gauntlet.py` | Runs one label set (`--set hard`/`orig`/`holdout`/`all`) through every method at once — geometry, both backbones at every render size, and every arbiter (gemma, haiku, sonnet, three Gemini) at both sheet sizes — as a per-model table. Needs `ollama serve` for the gemma row and gcloud ADC for Gemini; it skips what it cannot reach. A *per-model* instrument, not an accuracy measurement. |
| `gemini_vlm.py` | Runs the arbiter prompt through Gemini (3.5-flash / 2.5-flash / 2.5-pro) on Vertex AI, at both sheet sizes, and scores them beside the gemma/haiku/sonnet numbers. Also reports measured per-call tokens, latency, and $ per full-collection run. Self-contained: it builds its own sheets through `common.build_sheets` and reads the published predictions from `results-2026-08-12.json`. `--report-only` re-prints the tables from the last run's JSON. Auth is gcloud ADC. |
| `gemini_sheet_fill.py` | Does the arbiter care that a sub-512 tile sits *padded* inside a 512 px sheet cell? Three presentations of the same six tiles — `padded-384`, `filled-384` (upscaled to the cell, adding no detail), `native-512` — through one Gemini model. The measured half of the "the sheet only scales the cell, never the tile" note below. |
| `capture_vertex_verdicts.py` | The tri-state pass-2 pre-ship check (docs/tri-state-pass-2.md §C2): captures gemini-3.5-flash's real refusal envelopes — a `maxOutputTokens: 1` MAX_TOKENS body and a strict-safety SAFETY block — and reports what `_ask_gemini`'s 200-body split classifies each as. Two paid calls; raw bodies to `eval/out/vertex-verdicts/`, with the shapes pinned durably in `tests/test_pose.py::test_the_captured_vertex_envelopes_classify_as_designed`. Found MAX_TOKENS arrives with `parts: [{"text": ""}]` (the unparseable lane), not the documented no-parts husk. Auth is gcloud ADC. |
| `backbone_memory.py` | VRAM per SigLIP tower: params, image tokens, resident weights, peak allocated at a given batch, measured on the production `Embedder` load (fp16 — the `--dtype` sweep went with the copy). The weights row carries the four prompt banks on top of the parameters, identically in both rows. Run it with the GPU idle — `ollama` holding gemma4:26b (6.8 GB) makes every figure meaningless. |
| `views_camera_rotation.py` | I11: must `renderer.views` carry the up-rotation in the camera or in the mesh? Three paths per model × candidate up (the camera trick, production's rotated copy, and a `mesh.rotate` reference), under three arms — post-processing off (byte-stable renders, the real test), indirect light off (attribution), and the production config (scale). The camera trick **fails**: max 75/255 on 53.7% of pixels against a 2/255 repeat floor, because the ambient fill is a world-fixed environment map this Open3D build cannot rotate. The reworked `views` is byte-identical to the reference on 16/18 cases and matches its own repeat noise on the other 2. See LEARNINGS, "camera rotation and the world-fixed fill". |
| `render_determinism.py` | Which stage makes pose resolution nondeterministic (review U2), and whether the post-processing toggle fixes it (V1). Isolation: **Filament** — ~43% of pixels move on every repeat, the tower is bit-deterministic on identical tensors, and one tight-margin model flips its up *pick* between identical passes. The fix, run as arms per real model: `set_post_processing(False)` alone gives **7/7 byte-identical renders and 0.00 margin spread** after one throwaway frame (`noaa` unnecessary). The one-time cost: margins shift median 1.3e-01 / max 3.5e-01, and the toggle removes tone mapping wholesale, so the stable render is a much darker image. Both cache version bumps plus an accuracy re-read on the new-look pixels. Needs the census from `compile_pose_flips.py`. |
| `compile_pose_flips.py` | The pose-side answer to review R1, bounded per T2: an eager **census** of live margins over all 2799 pose-cached models (reused from `out/pose_live_margins.json`), then compile-vs-eager on the 197 whose *live* margins sit within 0.022 of the 0.45 gate or of 0. **0 gate flips** in the 49 still in-band at compare time (rule-of-three ≈ 6% per in-band model, ~107 in-band collection-wide); **4 up flips**, all at margins ≤ 4.1e-03 — ties. Margin deltas median 1.5e-03, max 1.5e-02; the ~20× pose-vs-category amplification (`combine_up`'s min-max) stands. The census's side findings outweigh the bound: cached margins drifted median **0.119** (p90 0.354) from live under the unbumped `POSE_CACHE_VERSION`; run-to-run noise (median 2.66e-02) is **18× the compile delta** and exceeds the largest compile delta anywhere on 123 of 197 models; and the in-band population is a range (107/133/385), not a point. |
| `compile_flips.py` | What torch.compile actually flips on the *category* side: pushes each model's 16 saved production views through eager and compiled SigLIP as identical preprocessed tensors (compile the *bound method* — wrapping the model silently stays eager), scores both, reports flips by name. On the 241 worst-margin models plus 100 control: **1 flip**, at a 4.3e-06 margin between overlapping categories; drift median 7.3e-04, max 3.1e-03. Even among the 44 models with margins below the drift, one flipped — direction matters, not just magnitude. Jpeg dilution is real: the flipped model's cached margin was 1.7e-03, its jpg margin 4.3e-06. |
| `score_precision.py` | Does the production scorer's fp16 cast (cached fp32 embeddings → fp16 for the GPU matmul) flip any classification? Scores all 2943 cached models three ways — production fp16-GPU, fp32 numpy, fp64 — under all three pool modes. Production pool: **1 flip in 2943**, on a 1.9e-05 margin between near-synonym categories; fp32 = fp64 everywhere. Max per-view sim delta is 6.2e-05 — 16× below torch.compile's 9.8e-04 embedding drift, and 124 models (4.2%) hold margins below that drift, which is the quantitative case behind compile's disqualification. |
| `siglip_bench.py` | The SigLIP side's knobs, measured on `Embedder.embed_images` itself without moving the embeddings: batch sweep, preprocessing cost, overlap, drift, and `torch.compile` (phase `compile`, `COMPILE_MODES=default,max-autotune,reduce-overhead` — modes are A/B interleaved against eager per the thermal protocol; max-autotune's GEMM autotuner is gated off this 24-SM card). Fed the overlap/thermal LEARNINGS entries and the compile-mode numbers. |
| `overlap_spike.py` | The most-cited spike in the refactor set: does a renderer child process feeding SigLIP through a bounded queue saturate the 4060? Three modes — `baseline` (sequential), `overlap` (the ceiling: cached poses, no cycle), `roundtrip` (the real pose→embed dependency graph, child holds up to `--inflight` meshes, now as `Renderer` residency slots rather than a hand-rolled dict). Produced the 1.17–1.21× cold-run overlap at ~94% busy, the roundtrip's 1.11× with a three-mesh resident dict, and two structural facts the interfaces note inherits: the parent→child queue is unbounded (the deadlock rule) and the roundtrip *rotated held meshes* — which is why its 1.11× survived I11 intact. |
| `ipc_spike.py` | What the render-child's queue actually costs in transport, isolated from render-wait: blasts the overlap spike's exact payload (24 tiles + 16 views, 17.7 MB/model) through `mp.Queue` vs a SharedMemory block pool with zero render time. Queue is ~13.5–14.3 ms/model (1.3 GB/s — the pipe, not pickle, is the tax); shm in-place is ~2.8–3.5 ms/model. Puts a ceiling on what shared memory can save the real pipeline. |
| `renderer_gil.py` | Splits the hot render line to attribute GIL hold: `render_to_image` holds the GIL ~85–92% of its 36–61 ms, `np.asarray` is free, `Image.fromarray` releases. The Spike-4 result that made the renderer a process rather than a thread; the ModernGL comparison shows ~3–5× less hold, not zero. |
| `renderer_open3d.py` | What Filament will and will not do with GPU memory: whether a mesh can be staged before `add_geometry` (no), whether a scene can be cleared without evicting it (yes — `show_geometry`, ~11× cheaper than remove+re-add), and which card it runs on (the iGPU, so "resident" is host RAM). Feeds `docs/actor-refactor/renderer_alternatives.md`. |
| `renderer_moderngl.py` | The alternative measured beside it. Selects the GPU by EGL device index — reaching the 4060, which Filament cannot — and checks the one that matters for the actor design: a second context in the same process, which Open3D core-dumps on. Needs its own venv, see the README header in the script. |

## Retired

Deleted 2026-08-18; git history keeps them, and every finding they produced is
already in `LEARNINGS.md` or `docs/learnings/`. They are listed because a
LEARNINGS entry may still name one.

| script | why |
|---|---|
| `arbiter_gate.py` | Scored a margin gate against geometry's ratio gate; the margin won and shipped as `needs_arbiter_margin` / `--up-margin`. Kept alive only the losing side — `pose.needs_arbiter`, deleted with it. Its `load_arbiters` survives in `common.py`, so recorded arbiter answers can still be replayed against a future gate without an API key. |
| `siglip_up.py` | Split into `pose_probe_sweep.py` (the labelled probe sweep, now through `load_labels()`) and `pose_label_sheets.py` (the unlabelled sheet builder, now with arguments). It drew its models from a seeded sample over a hardcoded walk path, which is both the convention CLAUDE.md warns about and the reason it had stopped running. |
| `tile_and_vlm.py` | Dead input: superseded by `tile_count.py` for the resolution sweep and `gauntlet.py` for the VLM comparison, and its ollama tier needs a backend production retired. |
| `claude_vlm.py` | Read `out/preds.json`, which only `tile_and_vlm.py` wrote. |
| `build_report.py` | Built its HTML from the same dead prediction dump. |
| `ensemble.py` | Compared ways of combining the geometry and SigLIP vectors — min-max, z-score, Borda, softmax, absolute-scaled, a hard switch. Min-max won and shipped; `LEARNINGS.md` explains why that is not arbitrary, and `geo_floor.py` carries the live question forward. |
| `one_model.py` | Per-candidate scores for named meshes, built on `siglip_up.py`'s copied geometry scorer. |
| `front_first.py` | A recorded negative result: front-first loses 9 of 44, and the chosen front is perpendicular to the true up on only 38/49, so a fifth of the time it excludes the right answer outright. Its 24-tile orbit builder moved to `common.build_orbit_tiles` — three surviving scorers read those pixels. |
| `load_path.py` | Its finding shipped as `src/loader.py`: `read_triangle_mesh` is ~15–30× slower than a numpy binary-STL parse, welding costs more than the upload it saves, and upload is only 6.6% of the path. |
| `light_probe2.py`, `light_probe3.py` | Shipped as `FILL_INTENSITY`; already marked superseded here before they were deleted. |

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
  warns at startup when `--render-size` is under `pose.SHEET_THUMB`, and
  `pose_label_sheets.py` warns for the same reason. `gemini_sheet_fill.py`
  measures how much it matters.
- **The port moved the pixels slightly, and the tile caches predate it.**
  `Renderer.pose_tiles` is not byte-identical to the `render_up_candidate_grid`
  that filled `out/tiles*`, `out/orbit384x4` and `out/upgrid384x4`. Measured on
  `test-stls` with post-processing off, one path per process (the two cannot
  share a process — the CLI helper calls `scene.clear_geometry()` and the
  `Renderer` keeps an LRU, so each pulls geometry out from under the other):
  **max 1–30 of 255 on 0.05–1.1% of pixels**, deterministic (the same path
  repeats byte-identically across processes, 18/18), and zero on a
  24-triangle mesh while both a 1.2k- and a 69k-triangle mesh differ. The
  meshes, cameras and material parameters are provably identical between the
  paths — the residue is the add/show/hide sequence Filament sees. It is
  *production's* sequence, which is the direction of travel: the harnesses now
  render what the render child renders. It moved no answer — `parser_gate.py`
  re-ran at 44/49 ensemble and 37/49 geometry with 0/49 picks moved, inside
  the A/A floor it prints — but it is another reason a cached tile and a fresh
  one are not interchangeable, alongside the orbit-cache staleness
  `tile_count.py --compare` reports.
- **Renders are `np.ndarray` now, not `PIL.Image`.** `src/renderer` hands back
  arrays because arrays are what cross the process boundary. Anything that
  saves a render needs one `Image.fromarray`; anything that embeds one needs
  nothing, because the SigLIP processor takes either.
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
