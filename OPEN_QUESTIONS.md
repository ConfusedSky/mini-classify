# Open questions and loose ends

Threads left unpulled as of 2026-08-12. This file holds open work only — the
open arguments and the reasoning kept around them, amended in place as they
evolve. When a question settles it moves to `RESOLVED_QUESTIONS.md` (a one-line
answer, linked to its `LEARNINGS.md` write-up) and leaves here. A discrete,
closable task instead becomes a GitHub
issue ([open issues](https://github.com/ConfusedSky/mini-classify/issues)) and
keeps a `→ #N` pointer in its entry. Ground truth for anything pose-related is
`up_axis_labels.json` — 49 hand-labelled models in three sets: `orig` (23,
tuned on), `holdout` (21, frozen), `hard` (5, hand-picked failures, not a
sample of anything). Read LEARNINGS before quoting a number off `orig`, and
never pool `hard` into an accuracy.

## Ready to do — measured, decided, not yet written

- **Render up-candidate tiles at a fixed 384 px** (→ #21), independent of
  `--render-size` — but note it collides with the 512 px sheet, and
  `classify_stls.py` now says so at startup whenever `--render-size` is below
  `pose.SHEET_THUMB`. `thumb=512` only upsamples the *cell*: `Image.thumbnail`
  never enlarges, so 384 px tiles sit padded inside 512 px cells and the arbiter
  sees a 384 px sheet with more whitespace. Either render the pose tiles at
  ≥512 — which costs nothing, since the sweep found 384–2048 equivalent for the
  ensemble and the cost is the geometry upload either way — or drop the sheet
  back to the tile size and accept the weaker arbiter. Now measured rather than assumed: neither tower gains
  anything from a source above 384 px, and 384 is `patch14-384`'s *best*
  column (39/44 against 38 at 2048). Embedding from it is ~3× faster (0.23 s vs
  0.66 s per model) before render savings. This also closes the dependency of
  pose on an unrelated output setting — `patch14-384` currently flips
  `Propane_Tank`, `Floor` and `32mm_Gate_L` on render size, and the pose cache
  (`path + mtime + size`) cannot tell the difference. Either fix the tile size
  or adopt `patch16-512`, which is invariant across 384–2048; the fixed size is
  much the cheaper of the two.
- **Pose resolution is nondeterministic at gate-crossing scale, and the
  mechanism is Filament** (→ #12) (review U2, `eval/render_determinism.py`,
  `eval/compile_pose_flips.py` census). Two identical eager passes move a
  model's ensemble margin by a median of 2.7e-02 — 6% of the 0.45 gate — and
  up to 2.7e-01; one tight-margin model (`32mm_Pipe5`) picked a *different up
  axis* on a re-run. Isolated: re-rendering the same loaded mesh changes ~43%
  of pixels by 2–28/255 every time, while SigLIP on byte-identical tensors is
  bit-deterministic (0.0 delta) — the renderer is the sole source, fp16
  kernel variance excluded. This bounds what any margin-level claim in this
  repo can mean (it is the `parser_gate` A/A conclusion with a magnitude),
  and geometry's arm was seeded precisely to prevent this class of
  irreproducibility. ~~Accept the floor, average over k renders, or file it
  under renderer-alternatives~~ — superseded by review V1, verified on 7
  real models: **`scene.view.set_post_processing(False)` makes every render
  byte-identical** after one throwaway frame (margin spread 0.00 on all 7,
  including `32mm_Pipe5`, whose pick instability reproduces with it on;
  antialiasing did not need to come off). The noise is Filament's temporal
  dithering and the fix is one line on the current renderer. **But the line
  is a blunt one** (seen, not just measured — `eval/out/render_arms.html`):
  `set_post_processing(False)` also removes tone mapping, so the stable
  render is near-black against the production look (max 187/255 over 100%
  of pixels), a different image, not the same image de-noised. And there is
  no finer knob: every `ColorGrading` quality × tone-mapping combination in
  Open3D's binding still dithers (all unstable, all 41–54/255 from
  production) — Filament's dither switch is not exposed separately. What
  remains is the adoption decision: the toggle moves margins by median
  1.3e-01 / max 3.5e-01 on these models — every pixel, embedding and margin
  changes — so
  adopting it is a `POSE_CACHE_VERSION` **and** `EMBED_CACHE_VERSION` bump
  (full re-resolve + re-embed, hours; the machinery exists — M3 built it for
  exactly this), and SigLIP's accuracy must be re-read with post-processing
  off (`gold_upright`, `tile_count`) since every recorded number was
  measured with it on. Until adopted, margin-adjacent analyses must quote
  the 2.7e-02 floor.
- **Upload the mesh once in `render_up_candidate_tiles`.** It builds six
  rotated copies and re-uploads each; the cost is geometry upload, not pixels
  (3.74 s at 2048 px vs 3.33 s at 384 px on 1.8M triangles). Move the camera
  instead.

## Open questions — genuinely unknown

- **Does the ensemble actually help?** (→ #11) Pooled over 44 models geometry and the
  ensemble differ on only **6**, where the ensemble is right 4 and geometry 1.
  Sign test p≈0.375. It survived the holdout as "probably helps, unproven".
  Resolving it needs more labels, not more analysis — the bottleneck is that
  labelling is manual and ~45% of any random sample has no defined upright.
- **A correct-but-uncertain answer can still be broken by escalation.** (→ #15)
  Attenuating geometry fixed `32mm_Orguss_Head` in the ensemble (`+Y`), but its
  margin is 0.19, so it still escalates and the arbiter answers `-Z` — the
  pipeline returns the wrong answer for a model the ensemble now gets right.
  The gate is not misfiring: 0.19 genuinely is unsure. The open question is
  whether escalation should be allowed to *lower* expected accuracy when the
  arbiter's own reliability on that model is poor, and whether anything
  observable at call time predicts that.
- **Is `up-only, 4 views` real?** (→ #11) Averaging the upright score over four
  azimuths instead of one is +2 on the holdout (17/21 → 19/21) and +1 pooled,
  at no extra geometry upload — but it turns on three disagreements, 2–1, which
  is p=0.5. Paired with the margin gate it is the best configuration measured
  (43/44 pooled, 21/21 holdout, 9 calls) — see LEARNINGS — so it is now part of
  the same decision as the gate rather than an independent lead. Still blocked
  on the same thing everything else is: more labels.
- **Is the arbiter deterministic?** (→ #13) Two runs of the same code on the same input
  disagreed on the up axis for one model, and a third arm on two — all
  `vlm`-sourced, all at `temperature: 0` against byte-identical sheets. The
  likely answer is that Gemini simply is not deterministic at temperature 0, but
  that has not been tested: run one arm twice and see whether it disagrees with
  *itself*. It matters beyond tidiness — every VLM number in this file is a
  single sample, and if the arbiter has a per-call variance then "43/44" carries
  an error bar nobody has measured. The 3-sample gemma test earlier (21/23
  unanimous) is the only evidence in the other direction, and it was a different
  model on a different backend.
- **Should an attempt counter bound cross-run re-asks?** (→ #14) Filed 2026-08-21 with
  the four-state `arbitrated` (`docs/archive/tri-state-pass-2.md`, out of scope
  there). `false` means "ask again on a later run", with no memory of how many
  runs have already asked — so a failure that is *deterministic for one model*
  and never returns a judged verdict is re-asked, and re-billed, once per
  arbiter-on run forever. The run-level breaker cannot catch it: it trips on
  **consecutive** `VLMUnavailable` folds, and one model failing among hundreds
  that succeed never produces a streak. This bites both backends, and the
  claude one hardest — `_ask_claude` raises only `VLMUnavailable`, so it can
  never write `"rejected"` at all, and a deterministic claude parse failure
  has no path to settling. The shape of a fix is a counted state in the same
  field (`arbitrated: 0|1|2|true`, absent still reading as no-claim), which
  keeps the schema's mixed-type readers but needs a rule for what happens at
  the cap — settle it as `"rejected"`, or keep re-asking and just report the
  count. What is unknown is whether the population is real: nobody has
  measured how many models fail the same way on two consecutive arbiter-on
  runs, and until the first backfill run (§8 of `docs/cache-rebuild.md`) there
  is no marked population to measure it over.
- **What does `--save-renders` cost on a *warm* embedding cache?** Measured free
  on a cold one (456 s against 459 s, inside noise), but that run rendered every
  model anyway. Warm, the flag is the difference between rendering nothing and
  rendering everything to refresh the debug images — which is exactly the case
  the day-to-day reruns hit.
- **Confidently wrong models are a class no gate can catch.** (→ #15)
  `32mm_Orguss_Head` (ensemble `-Z`, truth `+Y`, margin 0.41) and
  `Concrete Chunk (2)` (margin 1.04) are wrong *and* confident, so no usable
  margin threshold escalates them. Every local tier inverts Orguss_Head, and
  the three VLMs that get it each fail it at the other sheet size. Is this a
  label-ambiguity problem, a render problem, or a genuine capability ceiling?
  `eval/gold_upright.py` renders the label's own claim, which is where to start.
- **Can a "reason" field diagnose the VLM's failures?** (→ #18) Ask for
  `{"tile": n, "reason": "..."}` instead of the tile alone, and read the
  reasons on models where a model is stably wrong (`tile9`, `Bunker_MiniV2_Roof_`,
  `Floor`, `Concrete Chunk (6)`). Distinguishes hypotheses that accuracy alone
  cannot: is it misreading the numerals, misjudging which way is down, applying
  a positional prior, or is the *task* genuinely ambiguous for terrain? haiku
  picks `+X` ten times of 44 when the truth is never `+X`, and
  gemini-2.5-flash five to six times — a reason field would say so directly.
  Cheap: one prompt change, re-run over the 44. It may also *improve* accuracy
  by forcing deliberation, which would confound it as pure diagnostics — worth
  measuring both with and without.
- **Is 512 px the peak sheet size, or does 768/1024 keep helping?** (→ #20) Only 256 vs
  512 measured. Now known to be model-dependent, which changes who the question
  is for: gemini-3.5-flash is nearly flat across 256/512, so a sweep matters
  mainly for the weaker arbiters, where the gain was large (sonnet +10). Cost
  is nil at inference — one resize of already-rendered tiles.
- **Is the arbiter tier out of headroom, or is the gate the limit?** Three
  Gemini generations across two tiers rescue an *identical* four models
  (`Bedienkonsole`, `Mortimer_BodyNoMask`, `WisDevourer_Body`,
  `Container_complete`) and differ only in which terrain they break. That looks
  like a saturated rescue set rather than a model-quality ceiling — a better VLM
  is not the lever. The testable version: under the margin gate (which escalates
  ~20% instead of 55%) does the *set* of escalated models change enough that a
  different arbiter starts to matter? Every arbiter number in this file was
  measured under the geometry gate.
- **Should `--pose-vlm-model` default to `gemini-3.6-flash` for throughput?** (→ #11)
  Same net +4 and same pipeline 42/44 as the incumbent, one model lower
  standalone (42/44 vs 43/44), but 8.3 s per call against 24.1 s — ~12 min of
  arbiter wall-clock per full run instead of ~35, for $0.58 more. Trading one
  model of 44 for 3× the throughput is a judgement call nobody has made; the
  one model is inside the noise this file keeps warning about, and the 23
  minutes is not. Blocked on the same thing as everything else: at n=44 the
  accuracy difference is one model, so more labels would decide it.
- **Is `Floor` the new `tile9`?** `gemini-3.6-flash` and
  `gemini-3.1-pro-preview` both fail `Floor` (truth `-Z`) at both sheet sizes
  while gemini-3.5-flash gets it. Every remaining error at this tier is
  terrain, and the models now fail on *different* terrain. Worth pointing
  `eval/gold_upright.py` at `Floor` and `tile9` together: if a flat slab's "up"
  is genuinely ambiguous from six silhouettes, this is a label/task problem and
  no arbiter fixes it. Note `tile9` turned out **not** to be the universal
  failure this file claimed — geometry, the ensemble and gemini-2.5-pro all get
  it, and it never escalates. `Floor` may be the same kind of overstatement;
  check the whole column before calling any model "the hard one".
- **Is `Bedienkonsole` reachable without a VLM?** A console with a large flat
  rear panel, upright `+Z`. Geometry, the ensemble, and every SigLIP probe put
  it on its back; every Gemini model rescues it, as gemma and sonnet sometimes
  do. It remains the single model no geometric or embedding method in this
  project has solved.
- **Would a much *smaller* tower do for the pose ensemble?** (→ #19) The backbone work
  above asks whether a bigger tower helps. Pose is the place to ask the
  opposite, and it is now the most expensive thing in the run. Instrumented
  full-collection pass, 602 models at 384px: `pose-embed` is **29.3% of wall**
  (1176 ms/model), and after the STL parser landed it is **41.7%** on a subset —
  the single largest line item, costing 1.6× what embedding the actual
  classification views costs. All of it is SigLIP over 24 up-candidate tiles
  (6 candidates × `UP_TILE_AZIMUTHS`), on `so400m-patch14-384`: 1.1B params, the
  same tower used for fine-grained category probes.

  The tiles are near-silhouettes with no detail to resolve — LEARNINGS says so
  explicitly when explaining why a *stronger* tower buys resolution invariance
  rather than accuracy there. If that is right, a far smaller tower should cost
  little or nothing on pose. `google/siglip-base-patch16-224` is already in the
  HF cache, and `eval/backbone_sweep.py` crosses towers against the 44 labels
  with probes and combination frozen, re-embedding identical pixels — so this is
  a `--models` argument, not new code.

  Two things to watch. The probes (`UPRIGHT_PROMPTS` / `TOPPLED_PROMPTS`) were
  selected against `so400m`, so a weaker tower is being scored on prompts tuned
  for a different text encoder — a loss may be the probes, not the tower.
  And `MARGIN_THRESHOLD` gates arbiter escalation off the ensemble margin; a
  different tower rescales margins, so the gate would need re-reading against
  the labels rather than carried across.

  The cheaper variant of the same question needs no backbone at all, and it is
  now **measured** (`eval/tile_count.py`, LEARNINGS 2026-08-13): halving
  `UP_TILE_AZIMUTHS` to 2 flips zero ensemble picks on all 49 labels and costs
  one extra escalation (8 → 9 of 49), for half of `pose-embed` — see the
  ready-to-do entry. n_az=1 breaks three models and doubles the calls. The
  backbone half of this question stays open.
- **Does a better backbone help *category* classification?** (→ #23) The whole backbone
  comparison above is up-axis only, where the tiles are near-silhouettes with
  no detail to resolve. Categories are fine-grained text probes over detailed
  renders — the place extra resolution would plausibly pay. Untested and
  currently untestable: `up_axis_labels.json` is the only ground truth in the
  repo. Hand-labelling a category set is the prerequisite, not another backbone
  run.
- **Is the category ranking any good at all?** (→ #22) The prior entry asks whether a
  *better* backbone helps. This asks the question underneath it, and it is the
  one that gates the rest: every accuracy number in this project — 43/44, 39/44,
  every arbiter comparison — is up-axis. The top-3 category ranking that the CSV
  actually delivers has never been scored against anything. Until it is, render
  size, backbone and pipeline work are all optimising the cost of producing an
  output whose quality is unmeasured. **Cheaper than it looks:** the recorded
  prerequisite is hand-labelling a category set (602 models), but if the real
  use is search, precision@k over a dozen queries needs only the top ~20 results
  per query judged — ~200 judgements, scoring the thing that is actually used.

  The real use **is** search (2026-08-13): the collection is queried
  interactively through `test_categories.py`, and the classifier's CSV is an
  afterthought. So the precision@k variant is not the cheap alternative — it
  is the question, and hand-labelling 602 models to validate CSV top-3
  columns is not worth doing at all. The harness for it mostly exists:
  `test_categories.py` already produces those ranked lists from cached
  embeddings.

  Measured (2026-08-29), the precision@k variant with gemini-3.6-flash on
  1024 px views standing in for ~400 human judgements: production ranking
  scores **nDCG@20 0.852 / P@10 0.525** over 12 queries on a 301-model
  sample, and no renderer, framing, AO, rim-colour or prompt-template change
  moves it by more than +0.026 — so the ranking is decent and the pixels are
  not its ceiling. The harness is `eval/renderer_pilot/`, reusable for any
  other change to the embedding side; the recall/coverage half and a larger
  sample remain open. LEARNINGS, "What a VLM sees in our renders, and the
  renderer pilot".
- **Category classification is render-size sensitive; pose is not.** (→ #24) First data
  on the asymmetry the entry above predicts. Same 8 models, cold, `--views 8
  --elevations 20,-20`, 2048px against 384px:

  | | identical |
  |---|---|
  | up axis | 8/8 |
  | top1 category | 7/8 (`Remorhaz_A`: "demon or monster" → "dragon") |
  | `front_view` | 4/8 |

  Poses match exactly, as `patch16-512` invariance in LEARNINGS predicts for a
  384px source. But half the front-view indices moved and one top1 flipped, so
  the two paths cannot be assumed to behave alike, and the free win of rendering
  pose tiles at a fixed 384 does **not** transfer to the classification views
  without measuring it. With no ground truth this says "sensitive", not "worse" —
  the Remorhaz flip is arguably an improvement. A no-label sensitivity run
  bounds the risk without resolving correctness: at 2.7 s/model a full 602-model
  pass at 384 is ~27 minutes, which counts how many models change answer.
  ~~Note this decoupling is currently blocked anyway~~ — it is not: the
  one-renderer-per-process limit was refuted (four coexisted in one process;
  the abort is teardown-only, `docs/archive/reviews/2026-08-13.md` §3.1), so the
  decoupling needs only a second renderer kept alive for the process lifetime.
- **Which duplicate wins a tied query is now decided, but it was never
  chosen.** (→ #25) `rank`'s sort is stable as of `0f524e7`, so exact ties break by
  ascending collection index. That fixed a real non-determinism — the previous
  quicksort ordered equal scores arbitrarily, so two consumers of the same
  cache could list different models — but it settles *which* copy is shown by
  accident: "lowest index" is whatever order `load_file_list` happened to walk,
  and nothing about it says the winner is the better copy. The effect is not
  confined to ordering: when a tie straddles the top-N cut, a different model
  is **listed at all** (2124 of 16000 fuzzed cases, always at an equal score;
  `tests/test_query.py` pins the winner). The other copies vanish from the
  listing with no indication they existed.

  Ties are real here rather than theoretical because duplicated kits render
  byte-identically, so the same mesh under two paths earns the same embedding
  and therefore the same cosine score to every query. What is unknown is how
  often the *live* cache produces them: the 2124 figure is fuzzed data rounded
  to force ties, not a measurement of any real cache. Run it against
  `embed-cache2` (2945 pose entries), the primary one — a duplicate-heavy library is
  exactly where duplicated kits accumulate, and the 133-model `embed-cache4` is
  too small and too unrepresentative to answer it. First thing to run, and
  it is cheap — score the real matrix against a few dozen queries and count
  how many top-10 listings contain an exact score tie, then check whether the
  tied rows of `matrix` are bit-identical, which is the thing that causes the
  tie. Note `pose.file_identity` cannot answer this: it keys on relative path,
  mtime and size, so two copies of one mesh are deliberately *different*
  identities — file-level duplication needs size plus a content hash, and the
  embeddings answer the query question without touching the STLs at all.

  Only then is the design question worth arguing: leave it arbitrary-but-
  deterministic (one copy is probably what a browser wants), collapse exact
  duplicates and report a count, or return all of them and let the caller
  collapse. It bites the API sketch harder than the REPL — `docs/api/surface.md`
  caps hits at `top`/`cap`, so a duplicate-heavy query spends its budget
  listing the same mesh repeatedly, and model-browser has no way to tell.

## Performance work not done

- **Raise the thermal ceiling — hardware, not code.** (→ #27) The 4060 is an 80 W
  laptop part that enters SW thermal slowdown within seconds of sustained load
  (2250 → ~1400 MHz, ambient 66–71 °C at idle), which is what capped the
  overlap win at ~1.2×. A cooling or power-limit change is worth up to ~1.5×
  of embed throughput — more than every remaining software option combined.
  Check what the machine's power profile allows before optimising further.
- **Thread the render writes.** Largely obsolete: the default is `jpg` now, at
  0.13 s/model against PNG's 23 s. Still ~4 s of single-threaded CPU per model
  if someone runs `--render-format png`, and PIL releases the GIL during encode.
- **Should the contact sheet fill its cells?** (→ #20) `make_contact_sheet` uses
  `Image.thumbnail`, which never enlarges, so tiles rendered under
  `SHEET_THUMB` sit padded in the top-left of their cells. Measured on
  gemini-3.5-flash: padded-384 scores **41/44**, filling the cells 42/44, native
  512 43/44 — i.e. rendering pose tiles at 384 costs the entire documented
  256→512 gain, and upscaling recovers half of it for free
  (`eval/gemini_sheet_fill.py`, LEARNINGS). The one-line change is not made:
  +1 of 44 is p=0.5 alone, the upscale moves *more* answers than a real 512
  render does (6/49 against 3/49), and on the frozen holdout padded actually
  beat filled 21/21 to 20/21. Worth a second measurement before adopting, and
  worth deciding against `--render-size` rather than in isolation.
- **Can pose-tile render size be decoupled from `--render-size` at all?** (→ #21) Three
  separate wanted changes want three different sizes: pose tiles at a fixed 384
  for speed (LEARNINGS says neither tower gains above a 384 source), tiles at
  ≥512 so the Gemini arbiter is not starved, and classification views at 2048
  for detail. ~~Any two of those need two live renderers, which aborts the
  interpreter~~ — refuted by the review (`docs/archive/reviews/2026-08-13.md` §3.1:
  four renderers coexisted; only teardown aborts), so decoupling needs only a
  second renderer kept alive for the process lifetime. `--render-size` is
  still one knob serving three consumers with different optima.
- **Does `MARGIN_THRESHOLD` want re-reading after the parser swap?** (→ #17) The numpy
  parser shifts ensemble margins by a mean of 0.024, which walks models across
  the 0.45 gate. `Concrete Chunk (2)` stops escalating at both render sizes and
  `Bedienkonsole` at 384, and the ensemble has both wrong — two arbiter rescues
  silently removed, invisible in any accuracy column. Whether that costs
  anything depends on whether Gemini would have got them right, which
  `eval/parser_gate.py` does not test. More generally: the gate was tuned
  against margins from a different loader.
- **`common.build_sheets`' docstring is wrong** and should be corrected. It says
  "only the sheet size matters to a VLM; render_px ... made no difference", but
  the sweeps it rests on measured the *ensemble* — `tile_and_vlm.py` hands the
  VLM only its 2048 tiles. Measured directly, render_px matters to the arbiter a
  great deal (entry above). Left in place for now because it is a code change.

## Structural questions

- **`MpQueueTransport.recv(timeout)` is not bounded once a partial message is
  in the pipe** (→ #26) (2026-08-18, found while diagnosing the untimed-join hang —
  see the dated learnings entry). `_poll` succeeding only proves the *first*
  bytes arrived; the read of the remainder has no deadline, so a writer that
  dies mid-message leaves the parent blocked in `Connection._recv_bytes`, and
  that wait sits in the drain, *upstream* of the join the hang fix bounded. It
  needs a completed render to reach, so it cannot explain the observed stall,
  but it is reachable on a long run. A proper fix is a length-prefixed read
  with its own deadline — not a wrapper timeout, which would leave the
  half-message in the pipe and desynchronise the stream.
- **A wedged child still ends the run.** (→ #28) `fail_outstanding` sets
  `child_failed`, which stops the walk, so one wedge at model 500 of 1000
  retires the outstanding files and quits — now in ~250 s with durable caches
  and honest `Failure` rows (so a warm rerun resumes) rather than hanging
  forever with nothing flushed. Respawning the child instead would let the run
  continue; deliberately out of scope for the hang fix.
- **Faster failure detection: a child `Ready` handshake.** (→ #29) Considered and
  rejected as part of the hang fix (reasoning in the learnings entry): it moves
  detection from `STALL_S` to ~60 s but arrives at the same place, and adds a
  false-positive mode where a slow-but-healthy cold start is killed by a tight
  deadline. Worth doing on its own if 240 s to first diagnosis is too slow.

- **Renders are not reproducible across pose-cache states** (→ #12) — **and the cause
  is more general than that** (amended 2026-08-18, the refactor's parity run:
  LEARNINGS, "old against new"). The original observation: a cold pose cache
  renders the six up-candidate tiles through the same `OffscreenRenderer`
  first, and the view renders that follow differ from a warm-cache run by up to
  0.0098 per embedding component — a fifth of the ~0.03 gap between competing
  categories. Measured on all three test STLs, identical on the old code, so it
  is long-standing rather than new. Every embedding in the live cache was
  therefore computed in whichever state that file happened to hit.

  What the parity run adds: the pose cache is only one way to change the
  **sequence of draws preceding a view**, and *any* such change moves the
  pixels. Rendering a model in a three-model batch instead of alone moves its
  embedding by up to 7.0e-03 per component and its pose margin from 0.0724 to
  0.0791 with `front_view` 2 → 3 — measured on **unmodified old code**, so this
  is the renderer, not the refactor. Within one code path *and* one
  arrangement, both pipelines are bit-reproducible: two identical runs agree on
  every byte, so the noise floor is exactly zero and the variable is the
  arrangement alone.

  ~~Cheapest honest fix is to warm the renderer the same way on both paths~~ —
  there are not two paths to equalise. A fix has to make the renderer's output
  independent of its history, which is `set_post_processing(False)` and the
  cache-version bumps that come with it (the nondeterminism entry above), or a
  renderer we control. Widening the cache key still only makes the
  irreproducibility explicit rather than removing it. Until then: no
  margin-level claim in this repo survives a change of arrangement, and that
  now includes batch composition.
- **Widen the labelled set.** (→ #11) 44 sampled models, ~45% exclusion rate, and the
  decisive comparisons come down to 6 disagreements. Most conclusions in
  LEARNINGS are one or two models from flipping. The `hard` set added since
  sharpens specific failures but cannot substitute — it is chosen, not sampled,
  so it can only ever answer "which method survives this", never "how often".
  Everything else here is downstream of this.

- **Arbiter backend, decided for now (2026-08-30):** (→ #16) primary stays
  gemini-3.5-flash on production tiles, grid, 512 (43/44, +4 → 42/44,
  ~$2.7/run). The measured fallback is GLM-5.3-Flash **low** effort, production
  tiles, **solo** presentation, 512, pinned to DeepInfra with a hard per-call
  deadline (+3 → 41/44, $0.06/run, 3 s median) — better than the gemma/ollama
  fallback (+0/+1, evicts SigLIP) and no GPU. Not yet wired into
  `src/pose.py`; LEARNINGS, "Arbiter backends, sheet sizes, presentations and
  effort" has every number, including the ones that rule out GLM max, 1024
  sheets, f3d tiles and gpt-5.6-luna.
- **Can a part carry pose ground truth at all? Parked 2026-09-10.** (→ #30) The 206-model
  benchmark excludes parts from ground truth but production poses them anyway,
  so 94% describes a population the collection does not have. The plan was to
  label the up axis of the 160 triaged parts — those with a defensible upright
  joining the benchmark, those without becoming an explicit skip list. The class
  pass finished (56 accessory, 82 component, 16 that turned out not to be parts);
  the axis pass stopped after 11 proposals because **the reader's judgement was
  that most parts are genuinely ambiguous**, which is itself the finding: if a
  human cannot name the upright of a shield, a wing panel or a lettered manor
  wall, there is no ground truth to score against and the honest move may be to
  keep excluding them — and instead measure what the *pipeline costs* on them
  (the gate fires on 66% of parts against 26% of models, a projected 61-72% of
  arbiter spend), rather than what it gets right. Unsettled and blocking: for a
  worn or carried piece, is "up" as-worn/as-assembled or as it sits on a print
  bed? The proposals so far assume as-worn. State: `eval/out/parts/` holds the
  sheets, `manifest.json`, `classes.resolved.json` (path-keyed) and 11 axis
  proposals; the page is `eval/out/parts/index.html`.
  **Amended 2026-09-10:** the `reclaim` set gives the first measurement that
  bears on this. On 13 models built from exactly the shapes that defeat a
  name filter, the arbiter is **+0 (gemini) and −1 (GLM low)** while reading
  the sheets standalone at 9/13 and 5/13 — so on the shapes where the gate
  fires hardest it adds nothing. That argues the cheaper fix is a gate that
  declines to call when there is no up-axis evidence to read, rather than a
  better arbiter. See the dated learnings entry.
