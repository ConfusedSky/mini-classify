# Session Learnings — STL miniature classification (2026-08-10)

Journey: set up [Find3D](https://github.com/ziqi-ma/Find3D) → realized it was the
wrong tool for whole-model classification → built a local render + SigLIP
zero-shot classifier (`classify_stls.py`) for a ~1000-model printable
miniature collection.

This file is the index. Each session's write-up lives in `docs/learnings/`,
one file per dated section, split from the original single file with content
unchanged — a citation like "LEARNINGS, Overlap and the thermal ceiling"
resolves through the list below. New sessions get new files; the evergreen
notes at the bottom are amended in place. Open work is tracked separately in
`OPEN_QUESTIONS.md`.

## Session write-ups

- [2026-08-10 — the first classifier](docs/learnings/2026-08-10-first-classifier.md)
  — why Find3D was the wrong tool; rendering STLs so CLIP-family models can
  read them; the embedding cache; the three-script tool family; view pooling.
- [2026-08-11 — canonical pose](docs/learnings/2026-08-11-canonical-pose.md)
  — up axis from flat-base geometry with a confidence ratio, the VLM arbiter,
  and the `front_view` column.
- [2026-08-11 — elevation rings and the run manifest](docs/learnings/2026-08-11-elevations-and-run-manifest.md)
  — multi-elevation turntables, and `run-params.json` so the reader tools stop
  needing the classifier's flags retyped.
- [2026-08-12 — overlapping the arbiter](docs/learnings/2026-08-12-overlapping-the-arbiter.md)
  — deferring network arbiter calls, measured at 28% end to end, and the bug
  where the first version ran slower than what it replaced.
- [2026-08-12 — where a 7-hour run actually went](docs/learnings/2026-08-12-where-a-7-hour-run-went.md)
  — the accuracy campaign: arbiter comparisons across models and sheet sizes,
  the margin gate, the backbone sweep, geo_floor, and the hand-labelled ground
  truth. The largest write-up in the set.
- [2026-08-13 — instrumented](docs/learnings/2026-08-13-instrumented.md)
  — where the run goes stage by stage; rendering happens on the iGPU, not the
  4060; the GIL findings; what the numpy STL parser removed.
- [2026-08-13 — overlap and the thermal ceiling](docs/learnings/2026-08-13-overlap-and-thermal-ceiling.md)
  — a renderer child process takes the 4060 from ~57% to ~94% busy for a true
  1.2×; the 80 W card trades clocks for duty cycle; better cooling moves the
  ceiling; the ascending-sweep benchmark trap.
- [2026-08-13 — the round trip, half the tiles, and the full label set](docs/learnings/2026-08-13-roundtrip-tiles-and-the-full-label-set.md)
  — the pose→embed cycle keeps 1.11× of the overlap's 1.21× in the all-cold
  worst case, clearing the modularization; UP_TILE_AZIMUTHS=2 flips nothing on
  49/49; parser_gate gets its A/A control and passes; the reorg's six orphaned
  labels recovered, hard 5/5 at its true n=5.
- [2026-08-14 — hardening the unpacker](docs/learnings/2026-08-14-unpacker-hardening.md)
  — eight review passes, twenty-eight findings: Deflate64/cp437 archives,
  a swap with no zero-copy moment, CRC-verified redundancy with an override,
  flag-independent destinations, collision diversion to a fixed point — and
  four corrections to this write-up's own first draft.
- [2026-08-14 — data structures, and the queue's transport tax](docs/learnings/2026-08-14-ipc-transport.md)
  — the actor refactor's shapes fixed in `docs/actor-refactor/data_structures.md`;
  the render-child queue's cost isolated from render-wait: the pipe is the tax,
  not pickle, shm is 4–5× — and still only ~0.5% of a cold run, so v1 ships on
  `mp.Queue` behind a transport interface.
- [2026-08-14 — precision, and what it actually flips](docs/learnings/2026-08-14-precision-and-compile.md)
  — the fp16 scoring cast flips 1 of 2943, compile flips 1 of 341, both at
  coin-toss margins; jpeg dilution and the compiled-tower-that-wasn't canary;
  max-autotune is GEMM-gated off this card; `--compile` ships with the
  numeric regime keyed into the embedding cache.
- [2026-08-17 — the cache learned its schema, and the design was reviewed into shape](docs/learnings/2026-08-17-cache-schema-and-design-by-review.md)
  — cache-meta.json and the up_str token (a rename, not a re-embed; the 144
  the walk left behind); map-on-load beats version bumps; and what fourteen
  adversarial passes taught: walk protocols, remove carve-outs, make
  invariants mechanical, guard the clean path; convergence is severity
  falling, not counts.
- [2026-08-17 — camera rotation and the world-fixed fill](docs/learnings/2026-08-17-camera-rotation-and-the-world-fixed-fill.md)
  — the refactor's "rotate into the camera, never the mesh" rule measured and
  reversed (I11): the ambient fill is a world-fixed environment map, so a
  rotated rig lights the model differently — 75/255 on half the pixels under
  the production config. `views` rotates a copy instead; the caches stay valid,
  residency keeps the parse and pays the ~275 ms upload, and the roundtrip
  spike's 1.11× was always measured on this design.

- [2026-08-18 — old against new: the refactor's parity run](docs/learnings/2026-08-18-old-against-new-parity.md)
  — the branch measured against `main` on real runs: every cache key, pose,
  and top-3 ordering matches, and cold single-model runs are byte-identical on
  two of three models. The differences that remain are the renderer's own —
  the same old binary differs from itself by as much (7.0e-03) when a model is
  rendered in company rather than alone, which generalises the pose-cache-state
  entry to any change in the preceding draw sequence.
- [2026-08-18 — the refactored pipeline at scale](docs/learnings/2026-08-18-the-refactored-pipeline-at-scale.md)
  — first real run of the actor architecture: 133 models cold in 6:34, then a
  warm re-run in 12.4 s with a byte-identical CSV and no re-embedding. A
  baseline, not a speedup — but the 32× warm path, the exact 133×16 render
  count and the 1:1:1 rows/embeds/poses are `route()` and invariant 1 holding
  at scale. Tag filtering takes 292 files to 133, so per-model estimates off a
  raw file count run ~2× high.
- [2026-08-18 — the eval rig's tiles are not the old tiles](docs/learnings/2026-08-18-pose-tiles-and-the-draw-sequence-again.md)
  — porting the harnesses onto the render child's own `pose_tiles` changes
  pixels by up to 30/255 on ~1% of them, with geometry, cameras, material and
  framing all identical. The residue is the add/show/hide sequence Filament
  sees, making this the third independent measurement that its output depends
  on the scene's history. It moved no answer: `parser_gate` gives 0/49 picks
  moved and the difference sits inside its own A/A floor.
- [2026-08-18 — the untimed join, and why a missing EGL line is not evidence](docs/learnings/2026-08-18-the-untimed-join-and-the-missing-egl-line.md)
  — a 1-in-17 stall traced by elimination to `child.join()` being untimed on
  the one path that reaches quiescence by *killing* the child rather than
  waiting for it to idle. Two pieces of evidence read as damning and are
  worthless: the progress bar hits 100% at admission, and Open3D's EGL line is
  block-buffered, so its *absence* is a fingerprint of SIGKILL rather than of a
  stall before renderer startup.

- [2026-08-18 — a deferred import charges the caller](docs/learnings/2026-08-18-deferred-imports-charge-the-caller.md)
  — two modules whose docstrings asserted they were light were made heavy by
  the same move: `add_cache_args` charged 836 modules and ~0.9 s of torch to
  every tool that built a parser, and `embed_store` pulled a rendering library
  (+2602) to read `.npy` files. Deferring an import relocates its cost onto
  callers; it removes cost only when the signature already implies the
  dependency, which is why `up_axis_scores(mesh)` may defer open3d and a
  string-returning helper may not. The obvious fix was also the wrong one:
  open3d was 2596 of the 2602 and served exactly one function.

- [The query API, live](docs/learnings/2026-08-19-the-query-api-live.md) —
  2801 models, ready in 16 s, `/query` median **49 ms**. Two SigLIP instances
  fit the 4060 (**4740 of 8188 MiB**), so the server and a classify run
  coexist — unlike ollama, which is a reload thrash rather than a capacity
  problem. The API's top-10 is identical to `test_categories.py`'s, which is
  what `src/query.py` was extracted to guarantee. (~~The API's top-10~~ —
  amended 2026-08-27: the API has no ten-row default any more, and the REPL's
  is a display default it sends away under a floor. The shared *ranking* is
  still one copy, which is what the extraction guaranteed; the two defaults
  were never part of it. See below.)

- [Tri-state pass 2, and the new primary cache](docs/learnings/2026-08-21-tri-state-pass-2-and-embed-cache512.md)
  — the retry split's default flips after three passes each found transient
  failures on the permanent fallthrough; `arbitrated` becomes four-state with
  absence reading as `false` (`bool("rejected") is True` nearly shipped a
  schema that lied on disk); the warm/reload race took three tries because a
  generation counter is one decision per field it guards. embed-cache512
  (SigLIP-2 @512 px, 16 views) becomes the primary cache, and its backfill
  census answers "is the arbiter worth it" for the first time: **1048
  arbitrations, 651 moved the pose — a 62% move rate** on the gated
  population, zero rejections, zero transient failures, zero render errors.

- [Loading SigLIP without the hub](docs/learnings/2026-08-26-loading-siglip-without-the-hub.md)
  — `from_pretrained` on a repo id revalidates etags, so a fully cached model
  still needed the network; the failure was a **per-file 5-retry ladder** over
  files the repo does not even contain, indistinguishable from a wedged
  server, not an error. `src/embedder.py:load_siglip` is now the one load path
  for all five call sites: `local_files_only=True` first, retrying online on
  `OSError` so an unpulled model still downloads. It loads the **processor
  before the model** — a doomed offline attempt otherwise leaves an fp16 copy
  on the 4060 for the whole retry, since the traceback keeps the raising
  frame's locals alive, peaking at two.

- [A default that only became load-bearing later](docs/learnings/2026-08-27-a-default-that-became-load-bearing.md)
  — making `min_score` and `top` compose turned `QueryRequest.top`'s harmless
  ten-row default into a silent cut: on embed-cache512, `fantasy character` at
  floor 0.1 goes **875 rows to 10**, the cut landing in the top 1.4% of a set
  the caller asked to be exhaustive. The default lived in **four** places and
  only one was the schema — `rank()`'s own signature, `show_query`'s, and the
  contract table too. The composition *order* is invisible in the rows (the
  floor set is always a prefix of the sorted order, so the two orders commute
  on the result set — 0 divergences in 20000 tie-forced cases) and observable
  only in `matched`, which is what a contract test has to assert. Both guards
  written for this change were caught by mutation rather than by reading: a
  parity oracle that copies the call it guards tests the copy.

- [fp16 on a CPU](docs/learnings/2026-08-28-fp16-on-a-cpu.md)
  — `load_siglip` asked for fp16 on every device; on a CPU that is **3.7×
  slower** than fp32 (1.14 s vs 0.31 s per so400m text query) and **peaks at
  7.7 GB** converting the fp32 checkpoint, against fp32's 2.8 — enough to OOM
  an 8 GB host before its first query, which model-browser would show as
  `wedged`. fp32 returns identical rankings. The dtype now follows the device.
  Two CPU optimisations measured and declined: int8 dynamic quantisation is
  3× faster again but changes top-1 on 3 of 16 queries and moves `best_z` by
  ±1; loading only the text tower saves nothing, because `from_pretrained`
  mmaps the checkpoint and the vision pages are never touched. embed-cache-test
  (2165 models), 7940HS, `eval/cpu_dtype.py`.
- [What a VLM sees in our renders, and the renderer pilot](docs/learnings/2026-08-29-what-a-vlm-sees-and-the-renderer-pilot.md)
  — asked five Gemini models what one STL is from the production 512 px
  tiles (and the retired 384 px ones — same result): **0/5** got the concept (a crew member with an alien bursting from
  the head); from four 1024 px f3d views, **4/5**, and a camera tool the
  models could call added detail but not identity. GLM-5.3-Flash via
  OpenRouter got it 6/6 across effort levels at a tenth of the price.
  Resolution cannot help SigLIP (512 in, 512 rendered), so a pilot for
  *search* quality was built instead: six render arms (production, tight
  framing, f3d AO, model-browser's three.js chain with AO on/off and white
  rims), two prompt arms, 301 models, a VLM judge. Gotchas: f3d's system
  config forces `up: +Z` for `*.stl` over the command line; model-browser's
  spindle axis is the STL up rotated by `rotateX(−π/2)`. **Result: every
  alternative renderer beats production by +0.015–0.026 nDCG@20 (0.852 →
  0.867–0.878), uniformly and insignificantly** — 6–9 of 12 queries better,
  2–4 worse, one or two rank swaps each; AO on/off, rim colour and
  render-describing prompt templates are all within 0.01. The renderer is
  not where search quality is — and the verdict survives swapping the
  judge for GLM-5.3-Flash (kappa 0.80 with Gemini at 1/20th the cost;
  every arm still beats production, Spearman 0.71). `eval/renderer_pilot/`.

- [Arbiter backends, sheet sizes, presentations and effort](docs/learnings/2026-08-30-arbiter-backends-sheets-presentations-effort.md)
  — could GLM-5.3-Flash replace gemini-3.5-flash as the pose arbiter? On the
  44 labels: gemini at 512 stays unbeaten (43/44, **+4 → 42/44**); GLM low
  is +2 on the sheet and **+3 → 41/44 when shown six captioned images**
  instead of one sheet, at $0.06/run and 3 s median against gemini's $2.7 —
  the measured fallback. Ruled out with numbers: 1024 sheets (both models
  lose), f3d tiles (every model loses 3–7), GLM `max` (loops without
  terminus on flat slabs — 17k-token generations, 2/44 unanswered, tier
  +0), gpt-5.6-luna (net −2, and 56/56 descriptions call the Osteotron a
  dragonborn). Also: temperature 0 through OpenRouter is a distribution and
  pinning the provider does not change it; auto-routed max-effort requests
  hung for 46 min behind OpenRouter's keep-alive heartbeats, so the
  harnesses now pin, cap wall-clock per call, and log provider per call; a
  3-describer panel cannot rescue a model with a systematic prior.

- [The ev2 rebuild](docs/learnings/2026-09-01-the-ev2-rebuild.md)
  — embed-cache512 regenerated from empty under tight framing / ev2 / the
  provenance stack (2026-08-31, ~4 h 52 m): 3,380 models, 1,526 arbiter
  escalations **all answered by gemini-3.5-flash** (944 moved, 582
  confirmed, zero rejected — §8 closed by construction, every judgment
  stamped with its judge), ~$11.5 of Vertex. Tight framing then measured
  live against the archived pre-ev2 embeddings on the pilot's 12 queries,
  one GLM judge for both sides: **0.876 → 0.887 nDCG@20 (+0.012)**, 8/4
  queries better, matching the pilot's +0.015–0.018 prediction in
  direction and order. `eval/renderer_pilot/ev2_check.py`.

- [Elevation rings, finally measured](docs/learnings/2026-09-01-elevation-rings.md)
  — the −20° ring costs **−0.005 nDCG@20** to drop (0.886 → 0.881, GLM
  judge, pilot sample, live embed-cache512) — inside pilot noise, and half
  the tight-framing gain; halving by ring beats halving by azimuth
  (−0.011). A `--elevations 20` rebuild would halve render/embed cost and
  cache size; not acted on. `eval/renderer_pilot/elev_check.py`.


- [Parts, duplicates, and whether SigLIP can see part-ness](docs/learnings/2026-09-03-parts-duplicates-and-the-probe.md)
  — the duplicate census (98 md5-identical groups; 57 DM Stash `75_` scale
  twins the `75mm` tag never met; scale twins invisible to byte-dedupe by
  construction) and `eval/part_probe.py`: zero-shot part-ness over
  embed-cache512's cached embeddings reads ~41% of the corpus as parts at
  ~90% recall, **0%** on handless bodies, 0.4% true character FPs — not too
  wide. Shipped: `"standalone"` tag + anchored `^75_` rule, 3,380 → 3,092
  files. The two-tier flag is the repo issue "Part/accessory handling".


- [The query text budget: characters are not tokens](docs/learnings/2026-09-08-the-query-text-budget.md)
  — `/query`'s 500 on a long text was SigLIP2's 64-position tower reached
  without `truncation`, not a character limit: 60 emoji are 61 tokens, 100 CJK
  characters 101, 500 ASCII characters 97. Fixed by clipping in `embed_raw`, a
  422 naming both numbers, and `text_budget` (**54** = 64 − the templates' 9 −
  1 for re-segmentation at the join) published on `/status`. The counter shares
  the forward's lock: a fast tokenizer is mutable, and unlocked it raised
  `Already borrowed` 117,725 times in 8 s. Issue #5, measured against
  embed-cache512.

## Evergreen notes

- [Queries and filters](docs/learnings/queries-and-filters.md) — open-set
  "not in the collection" detection; filter gotchas.
- [Workflow](docs/learnings/workflow.md) — REPL affordances, repo hygiene,
  environment and tooling.
