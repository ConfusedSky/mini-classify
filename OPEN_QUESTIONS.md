# Open questions and loose ends

Threads left unpulled as of 2026-08-12. `LEARNINGS.md` records what was
settled; this records what was not. Ground truth for anything pose-related is
`up_axis_labels.json` — 49 hand-labelled models in three sets: `orig` (23,
tuned on), `holdout` (21, frozen), `hard` (5, hand-picked failures, not a
sample of anything). Read LEARNINGS before quoting a number off `orig`, and
never pool `hard` into an accuracy.

## Settled since the last revision

Moved out of this file; the measurements are in `LEARNINGS.md`.

- **Set `UP_TILE_AZIMUTHS = 2`** — adopted. Measured on all 49 labels at
  production pixels first: zero ensemble pick changes, per-set accuracy
  identical (21/23, 18/21, 5/5), one extra escalation — for half of the run's
  largest GPU item. The gate re-read this entry required happened in the same
  harness: replaying ten recorded arbiters through `MARGIN_THRESHOLD=0.45` at
  n_az=2, every one scores ≥ its n_az=4 self (gemini-3.5-flash@512 gains,
  37/40 → 38/40), so 0.45 stands; re-read again if production escalation
  rates drift, since crossing counts carry ±2–3 models of pixel-source noise.
  `POSE_CACHE_VERSION` deliberately not bumped: the measurement is that picks
  do not move, VLM-sourced entries are kept whatever gated them, and
  re-resolving 2284 poses would cost hours to reproduce answers shown
  equivalent — bump to 5 if that judgement ever looks wrong.
  `eval/tile_count.py`, LEARNINGS 2026-08-13.
- **Does overlapping render and embed saturate the 4060?** — measured, yes and
  no. A renderer child process feeding SigLIP through a bounded queue takes the
  card from ~57% to ~94% busy, but wall-clock gains only **1.17–1.21×** against
  the Amdahl prediction of 1.45×: saturating the card drops it into thermal
  slowdown and the same embed work runs 1.4–1.6× slower per image. The
  boundary is worth adopting (it is also the actor proposal's one load-bearing
  process split); more threading beyond it buys nothing the fan doesn't take
  back. `eval/overlap_spike.py`, LEARNINGS "Overlap and the thermal ceiling".
- **Is there headroom inside SigLIP itself?** — measured, effectively no.
  Throughput is flat across batch sizes 1–128 (power/thermal-clamped, never
  VRAM-limited), `torch.compile`'s 1.10× is disqualified because its drift is
  the size of the closest top-1 margin and would seed the permanent `.npy`
  cache inconsistently, and threaded preprocessing is a measured 1.044× —
  available, minor. Cutting `pose-embed`'s tile count (below) is the remaining
  software lever. `eval/siglip_bench.py`.
  gemini-3.5-flash scores 43/44 standalone (21/21 on the holdout), beating the
  ensemble outright; net +4 as an arbiter tier, pipeline 42/44.
  `eval/gemini_vlm.py`.
- **Do newer Gemini models beat the arbiter default?** — measured, no.
  `gemini-3.6-flash` ties the incumbent as an arbiter (net +4, pipeline 42/44)
  and `gemini-3.1-pro-preview` is worse (net +3) at 2.3× the price; standalone,
  both land one model below gemini-3.5-flash's 43/44. All four rescue the same
  four models. `gemini-3-pro-image` answers correctly but is rate-limited to
  uselessness. Default unchanged. `eval/gemini_vlm.py --out gemini_vlm-new3.json`.
- **Would a 512 px SigLIP help?** — `siglip2-so400m-patch16-512` is worth +1 of
  44 (p=0.5) but is *resolution-invariant* where `patch14-384` flips three
  models on render size alone. Memory is a wash: both 4.3 GB on disk, 2.19 GB
  resident, differing by 1 MiB of weights; the +40% image tokens cost ~1.02×
  memory and ~24% time. `eval/backbone_sweep.py`, `eval/backbone_memory.py`.
- **Gate the VLM on the ensemble's margin** — measured, then ~~still to be
  *written*~~ **written** (2026-08-17, the actor refactor): the escalation gate
  is `pose.needs_arbiter_margin(margin, threshold)`, the threshold is
  `MARGIN_THRESHOLD = 0.45` and `--up-margin` exposes it; ~~`needs_arbiter(ratio,
  best)` survives only for the geometry-only arm no production path takes~~ —
  **the geometry gate itself was deleted 2026-08-18**, along with
  `eval/arbiter_gate.py`, the harness that scored the two against each other.
  The measurement stands: `margin < 0.4` matched the geometry gate's accuracy
  on 9 calls instead of 24 and stopped a weak arbiter going net negative
  (haiku@256: 30/44 → 39/44). Reproducing that *comparison* now means
  recovering both from git history (`git show 4a34009:src/pose.py` for
  `needs_arbiter`, `git show 4a34009:eval/arbiter_gate.py` for the sweep) —
  a deliberate trade: the losing side of a settled comparison was costing a
  live function and a harness to keep. `load_arbiters` survived the deletion
  in `eval/common.py`, so replaying recorded arbiter answers against any
  *future* gate needs no API access. The *pairing* the entry below insists
  on — adopt it with the four-view ensemble — did **not** ship: `--views` for
  the up ensemble is still `UP_TILE_AZIMUTHS = 2`, so that half stays open
  below.

- **Raise the contact sheet to `thumb=512`** — done, with the scaled numerals
  it depends on, now in `pose.make_contact_sheet` rather than only in the eval
  harness. sonnet gains 10 of 44 across that step and gemma 3; gemini-3.5-flash
  barely notices, so the size still belongs in any report of a VLM number.
- **Wire a Gemini backend into the arbiter** — done. `--pose-vlm gemini`,
  default `gemini-3.5-flash`, ADC auth, project from `--gemini-project` /
  `$GOOGLE_CLOUD_PROJECT` / `gcloud config`. **`auto` now prefers it**, ~~falling
  back to ollama and then to no arbiter~~ — **gemini or nothing** since
  2026-08-17 (C-R1-4): the Arbiter is a thread pool with no inline arm, and a
  pooled ollama call would put gemma on the 4060 beside SigLIP (10.1 s of
  reload against 0.49 s of inference, CLAUDE.md's hard constraint), so
  `--pose-vlm ollama` is retired from the CLI and `VlmConfig` refuses the name
  at construction. `auto` resolves gemini or falls straight to no arbiter;
  measured cost of dropping the local tier, ~1 model in 44. It bills per call
  (~$0.30 per full-collection run at ~120 escalations), so the selection is
  always announced. With the 512 px sheet the local path was net positive too,
  so the "run with `--pose-vlm off`" advice no longer applies either way — and
  a serialized-inline ollama mode can return as its own design item.
- **Should saved renders keep feeding SigLIP?** — no, and they no longer do.
  Embeddings now come from the `.npy` cache or a fresh in-memory render; saved
  renders are debug output living under the config that produced them. That
  also removed a worse bug than the format question it was filed under: a
  render filename carries only stem and view index while `cache_key` covers
  size/views/elevations, so changing `--render-size` embedded the old size's
  pixels under the new key — 0.976 cosine, permanently wrong.
- **`compress_level=1` on saved renders** — superseded. Decoupling made lossy
  safe, so `--render-format` defaults to jpg: 23.3 s → 0.13 s per model and
  ~27 GB → ~2 GB for the collection. `png` keeps `compress_level=1`.

## Ready to do — measured, decided, not yet written

- ~~**Gate the arbiter on the ensemble's margin**~~ — **shipped 2026-08-17**
  as `--up-margin` / `needs_arbiter_margin` (see the settled entry above); at
  `MARGIN_THRESHOLD = 0.45`, not the 0.4 this entry proposed — the 0.45 re-read
  is the `UP_TILE_AZIMUTHS = 2` entry's. The pose-cache migration it says the
  change needs was **not** run: `POSE_CACHE_VERSION` stayed at 4, so every
  cached `source: "vlm"` pose still carries an answer decided under the
  *geometry* gate, and only files re-resolved since escalate under the margin.
  The rest of this entry is the reasoning that bought it, and the four-view
  pairing below is still unadopted, which is the part that matters:
  (`top1 − top2` of `_unit(geo) + _unit(siglip)`) instead of
  `needs_arbiter(ratio, best)` (deleted 2026-08-18; see the settled entry).
  Threshold 0.4 picked on `orig`; holdout 19/21 against the geometry gate's
  20/21, pooled tie at 42/44, on ~20% of the collection instead of ~55%. That
  is 354 arbiter calls down to ~120 per run. It also removes the tier's ability
  to be net-negative with a weak arbiter, which is what the "is this tier worth
  keeping" argument was always about. Needs a pose-cache migration: every
  cached `source: "vlm"` pose was decided under the old gate.

  **Adopt it together with the four-view ensemble, or the four-view change will
  look worthless.** Under the geometry gate four views measures *worse* (42/44
  against 43/44) because its rescues duplicate the arbiter's; under the margin
  gate the two stack to 43/44 on 9 calls and 21/21 on the holdout. A tier that
  overrides another hides that tier's progress — measure component changes
  against the component, gate changes against the pipeline.
- **Render up-candidate tiles at a fixed 384 px**, independent of
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
- ~~**Rename `source: "heuristic"` to `"confirmed"`.**~~ Resolved 2026-08-14,
  differently on both counts (`docs/reviews/2026-08-14-data-structures.md`
  §P2.3-A): the vocabulary is now `geometry`/`siglip` — one axis, *whose
  answer prevailed*, where "confirmed" mixes a state with two mechanisms and
  is actively false for a `--no-up-ensemble` run. And no pose-cache migration
  after all: `load_pose_cache` maps the old spellings on load, and the
  `up_str` embed token (§P2.3-B) took `source` out of the cache key entirely.
- **Pose resolution is nondeterministic at gate-crossing scale, and the
  mechanism is Filament** (review U2, `eval/render_determinism.py`,
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

- **Does the ensemble actually help?** Pooled over 44 models geometry and the
  ensemble differ on only **6**, where the ensemble is right 4 and geometry 1.
  Sign test p≈0.375. It survived the holdout as "probably helps, unproven".
  Resolving it needs more labels, not more analysis — the bottleneck is that
  labelling is manual and ~45% of any random sample has no defined upright.
- **A correct-but-uncertain answer can still be broken by escalation.**
  Attenuating geometry fixed `32mm_Orguss_Head` in the ensemble (`+Y`), but its
  margin is 0.19, so it still escalates and the arbiter answers `-Z` — the
  pipeline returns the wrong answer for a model the ensemble now gets right.
  The gate is not misfiring: 0.19 genuinely is unsure. The open question is
  whether escalation should be allowed to *lower* expected accuracy when the
  arbiter's own reliability on that model is poor, and whether anything
  observable at call time predicts that.
- **Is `up-only, 4 views` real?** Averaging the upright score over four
  azimuths instead of one is +2 on the holdout (17/21 → 19/21) and +1 pooled,
  at no extra geometry upload — but it turns on three disagreements, 2–1, which
  is p=0.5. Paired with the margin gate it is the best configuration measured
  (43/44 pooled, 21/21 holdout, 9 calls) — see LEARNINGS — so it is now part of
  the same decision as the gate rather than an independent lead. Still blocked
  on the same thing everything else is: more labels.
- **Is the arbiter deterministic?** Two runs of the same code on the same input
  disagreed on the up axis for one model, and a third arm on two — all
  `vlm`-sourced, all at `temperature: 0` against byte-identical sheets. The
  likely answer is that Gemini simply is not deterministic at temperature 0, but
  that has not been tested: run one arm twice and see whether it disagrees with
  *itself*. It matters beyond tidiness — every VLM number in this file is a
  single sample, and if the arbiter has a per-call variance then "43/44" carries
  an error bar nobody has measured. The 3-sample gemma test earlier (21/23
  unanimous) is the only evidence in the other direction, and it was a different
  model on a different backend.
- **What does `--save-renders` cost on a *warm* embedding cache?** Measured free
  on a cold one (456 s against 459 s, inside noise), but that run rendered every
  model anyway. Warm, the flag is the difference between rendering nothing and
  rendering everything to refresh the debug images — which is exactly the case
  the day-to-day reruns hit.
- **Confidently wrong models are a class no gate can catch.**
  `32mm_Orguss_Head` (ensemble `-Z`, truth `+Y`, margin 0.41) and
  `Concrete Chunk (2)` (margin 1.04) are wrong *and* confident, so no usable
  margin threshold escalates them. Every local tier inverts Orguss_Head, and
  the three VLMs that get it each fail it at the other sheet size. Is this a
  label-ambiguity problem, a render problem, or a genuine capability ceiling?
  `eval/gold_upright.py` renders the label's own claim, which is where to start.
- **Can a "reason" field diagnose the VLM's failures?** Ask for
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
- **Is 512 px the peak sheet size, or does 768/1024 keep helping?** Only 256 vs
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
- **Should `--pose-vlm-model` default to `gemini-3.6-flash` for throughput?**
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
- **Would a much *smaller* tower do for the pose ensemble?** The backbone work
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
- **Does a better backbone help *category* classification?** The whole backbone
  comparison above is up-axis only, where the tiles are near-silhouettes with
  no detail to resolve. Categories are fine-grained text probes over detailed
  renders — the place extra resolution would plausibly pay. Untested and
  currently untestable: `up_axis_labels.json` is the only ground truth in the
  repo. Hand-labelling a category set is the prerequisite, not another backbone
  run.
- **Is the category ranking any good at all?** The prior entry asks whether a
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
- **Category classification is render-size sensitive; pose is not.** First data
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
  the abort is teardown-only, `docs/reviews/2026-08-13.md` §3.1), so the
  decoupling needs only a second renderer kept alive for the process lifetime.
- **Which duplicate wins a tied query is now decided, but it was never
  chosen.** `rank`'s sort is stable as of `0f524e7`, so exact ties break by
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
  to force ties, not a measurement of `embed-cache4`. First thing to run, and
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
- **The cache key records the pose but not the recipe that drew it.**
  `cache_key` covers views, elevations, render size, model, `--compile` and the
  up *vector* — not `rotation_to_z_up`, which decides *which* rotation realised
  that vector and therefore which side of the model each azimuth sees. Change
  that function and 27 of `embed-cache4`'s 133 non-`+Z` models are re-posed
  under unchanged keys: the embeddings on disk answer a different question than
  the ones a fresh run would produce, and nothing anywhere fails.
  `tests/test_renderer.py::test_rotation_to_z_up_is_pinned_for_all_six_candidates`
  (2026-08-19) stops the accidental version of this. It cannot stop a
  deliberate one, which is the actual question.

  `rotation_to_z_up` is only the instance that surfaced. The whole render
  recipe is outside cache identity — `orbit_camera`'s framing, the 1.4 radius
  factor, the sun direction, the material. model-browser hit this class and
  answered it with `RIG_VERSION`: a hand-set integer in the cache key that
  anything altering thumbnail pixels must bump, on the reasoning that a cache
  keyed on inputs which do not capture the recipe is silently *wrong* rather
  than merely stale. That is the shape of a fix here too, and it is not free —
  a bump invalidates every cached embedding, and a full pass is hours.

  ~~The line worth drawing is probably between a semantic re-pose and
  numerical drift, and nobody has written that distinction down.~~ **That
  framing was wrong, and the fix came from model-browser (2026-08-19): phrase
  the trigger over the recipe *surface*, not over the output.** `RIG_VERSION`
  reads "bumped whenever rendered output changes for the same input", and then
  enumerates what that means — rig contents, materials, tone mapping — with
  the shadow-fit constants pinned by a separate test whose comment says
  changing one needs a bump. So the trigger is *did an author touch a named
  part of the recipe*: a judgement about intent at the call site, not a
  measurement of pixels.

  That dissolves the drift problem instead of solving it. Numerical drift does
  not touch the enumerated surface, so it cannot trip a rule phrased over that
  surface, and no tolerance has to be defined. Filament's draw-history
  dependence stops mattering too, because nothing ever compares two renders to
  decide. What is needed is a *list* — `rotation_to_z_up`, `orbit_camera`, the
  1.4 radius factor, the sun direction, the material — and a rule that touching
  any of them is a bump however the pixels happen to land.

  What remains genuinely open, then, is narrower than this entry first claimed:
  **what belongs on that list**, and whether the cost is worth paying at all —
  a bump invalidates every cached embedding and a full pass is hours, which is
  a real reason to keep pinning behaviour by test and bump nothing. Two things
  to carry if it is ever built: `CACHE_VERSION` versions the *key scheme* by
  design, so this would be a second integer with different rules and the two
  will be confused unless the difference is written down; and it wants a
  changelog rather than a bare integer, in the shape `RIG_VERSION` uses
  (`1 = pre-rim rig, 2 = rim accents, … 6 = STL normals from winding`). An
  integer only says the cache is invalid. The log says which models to look at
  and why, which is the difference between a deliberate re-pose being
  reviewable and merely being loud.

## Performance work not done

- **Raise the thermal ceiling — hardware, not code.** The 4060 is an 80 W
  laptop part that enters SW thermal slowdown within seconds of sustained load
  (2250 → ~1400 MHz, ambient 66–71 °C at idle), which is what capped the
  overlap win at ~1.2×. A cooling or power-limit change is worth up to ~1.5×
  of embed throughput — more than every remaining software option combined.
  Check what the machine's power profile allows before optimising further.
- **Thread the render writes.** Largely obsolete: the default is `jpg` now, at
  0.13 s/model against PNG's 23 s. Still ~4 s of single-threaded CPU per model
  if someone runs `--render-format png`, and PIL releases the GIL during encode.
- ~~**Prefetch mesh loads**~~ — **settled, and the old guidance here was
  wrong.** This entry said one loader thread is enough whenever the load/GPU
  ratio is under 1, from a 0.37 ratio measured on a subset. Collection-wide
  instrumentation says `mesh-wait` — the main thread actually blocked — was
  **18.6% of a 2121 s run**, the third largest line item. Subsets understate this
  badly, because mesh size varies more across the collection than anything else.
  The fix was not more threads: the numpy STL parser took it to 0.4%. See
  LEARNINGS.
- **Overlap the VLM arbiter with everything else** — done and measured at
  **28%** end to end (631 s → 456 s on 74 models). See LEARNINGS for the A/B and
  for the bug the A/B caught, where the first version ran *slower* than what it
  replaced.

  ~~Still open: the GPU is idle ~73% of the main pass and ~2 s per model is
  unattributed.~~ **Attributed** — `--instrument` plus py-spy, see LEARNINGS.
  Two corrections to what this entry assumed. The idle was not one GPU going
  unused: rendering runs on the **amdgpu iGPU**, so the 4060's idle time was
  never render/embed contention. And `py-spy` does *not* need root here —
  `ptrace_scope=1` permits tracing descendants, so `py-spy record -- <cmd>`
  works; only attaching to an unrelated pid needs sudo.
- ~~**Split the VLM pass from the render pass**~~ in `classify_stls.py` —
  **closed 2026-08-17 by removal rather than by scheduling.** Measured from the
  other side: gemma4:26b sits at 6818 MiB resident on a 7834 MiB card, so it and
  SigLIP (2.2 GB) genuinely cannot coexist. Every harness in `eval/` phases
  render → SigLIP → VLM by construction (`common.build_tiles` caches the pixels
  so the towers never overlap), and ~~the production path still interleaves
  them~~ — the production path no longer has a local VLM to interleave: the
  Arbiter is a thread pool of *network* calls, and `--pose-vlm ollama` is
  retired (C-R1-4, entry above). Nothing in the pipeline can put a second model
  on the 4060 any more, so there is no pass to split. The question returns
  intact the day a serialized-inline ollama mode is wanted.

- **Should the contact sheet fill its cells?** `make_contact_sheet` uses
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
- **Can pose-tile render size be decoupled from `--render-size` at all?** Three
  separate wanted changes want three different sizes: pose tiles at a fixed 384
  for speed (LEARNINGS says neither tower gains above a 384 source), tiles at
  ≥512 so the Gemini arbiter is not starved, and classification views at 2048
  for detail. ~~Any two of those need two live renderers, which aborts the
  interpreter~~ — refuted by the review (`docs/reviews/2026-08-13.md` §3.1:
  four renderers coexisted; only teardown aborts), so decoupling needs only a
  second renderer kept alive for the process lifetime. `--render-size` is
  still one knob serving three consumers with different optima.
- **Does `MARGIN_THRESHOLD` want re-reading after the parser swap?** The numpy
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
  in the pipe** (2026-08-18, found while diagnosing the untimed-join hang —
  see the dated learnings entry). `_poll` succeeding only proves the *first*
  bytes arrived; the read of the remainder has no deadline, so a writer that
  dies mid-message leaves the parent blocked in `Connection._recv_bytes`, and
  that wait sits in the drain, *upstream* of the join the hang fix bounded. It
  needs a completed render to reach, so it cannot explain the observed stall,
  but it is reachable on a long run. A proper fix is a length-prefixed read
  with its own deadline — not a wrapper timeout, which would leave the
  half-message in the pipe and desynchronise the stream.
- **A wedged child still ends the run.** `fail_outstanding` sets
  `child_failed`, which stops the walk, so one wedge at model 500 of 1000
  retires the outstanding files and quits — now in ~250 s with durable caches
  and honest `Failure` rows (so a warm rerun resumes) rather than hanging
  forever with nothing flushed. Respawning the child instead would let the run
  continue; deliberately out of scope for the hang fix.
- **Faster failure detection: a child `Ready` handshake.** Considered and
  rejected as part of the hang fix (reasoning in the learnings entry): it moves
  detection from `STALL_S` to ~60 s but arrives at the same place, and adds a
  false-positive mode where a slow-but-healthy cold start is killed by a tight
  deadline. Worth doing on its own if 240 s to first diagnosis is too slow.

- **Dedup pass over the wave-1 `src/` modules** (2026-08-17, after wave 2
  lands and before the whole-branch review). AST-level scan found one true
  src-internal duplicate: `render_key` byte-identical in
  `cache_checker.py:65` and `renderer.py:126` (parent and child each grew a
  copy because neither may import the other). Consolidation: keying helpers
  (`render_key`, `cache_key_from_identity`, `EMBED_CACHE_VERSION`) move to
  `src/identity.py` — the stdlib-only leaf both sides import (E-R1-5's
  recorded direction; the leaf's stdlib-only import rule must survive the
  move). Everything else duplicated is `src/`-vs-`classify_stls.py` and
  dissolves when the wave-2 CLI rewrite delegates to `src/` — functions
  `pool_sims`, `view_config`, `orbit_camera`, `view_angles`, the key
  builders; constants `STL_RECORD`, `RENDER_FORMATS`, `DEFAULT_ELEVATIONS`,
  `EMBED_CACHE_VERSION`, `FILL_INTENSITY`, `SUN_INTENSITY`,
  `UP_TILE_AZIMUTHS`, `UP_TILE_ELEVATION`, the prompt constants (full
  constant sweep 2026-08-17; each already has exactly one `src/` home).
  Retire each pair's parity-pin test together with its copy, not before.
  Exit check: an AST sweep for same-named module-level constants or
  functions across `src/` + `classify_stls.py` comes back empty (bar
  `__init__`).
  ~~**Done 2026-08-18.**~~ The sweep found 17 duplicated names (the four the
  entry did not list: `rotation_to_z_up`, `read_binary_stl`, `load_mesh`,
  `STL_RECORD`) and comes back empty now. Keys moved to `src/identity.py`
  (`render_key`, `cache_key_from_identity`, `EMBED_CACHE_VERSION`, and
  `DEFAULT_ELEVATIONS`, which the key elides and so had to travel with it);
  `done` imports them from there rather than from `cache_checker` (E-R1-5
  closed). Everything else is a `classify_stls` re-export of the `src/` home:
  `loader` for the STL parse, `renderer` for the camera/light/tile names —
  and `pool_sims`/`view_config`, whose home `src/done.py` owns torch, are
  forwarded by a module `__getattr__` so the CLI stays torch-free at module
  scope for the spawned child. Keys verified byte-identical to the pre-dedup
  source across 36 flag combinations, so no cache is invalidated. Three
  parity pins retired with their copies (`test_key_composition_matches_
  classify_stls`, `test_pool_sims_parity_all_modes`, `test_view_config_
  parity`): 398 → 395 passing. Pylint's duplicate-code checker over `src/` +
  `classify_stls.py` reported 7 clone pairs before and none after; the
  camera-carried rotation, which was a real clone the entry did not name,
  became `renderer.rotated_cams` (one definition for the child's pose tiles
  and the evals' contact-sheet grid), and `make_renderer` now delegates to
  `renderer.make_offscreen`.
  **Amended 2026-08-18 (eval-debt cleanup, phase 2).** The re-exports the
  entry describes are gone, not just deduplicated: every consumer imports the
  owning `src/` module, so the forwarding block and the `__getattr__` shim
  were deleted along with `make_renderer`/`_shoot`/`_upload`/`render_views`/
  `render_up_candidate_grid`/`render_up_candidate_tiles`/`resolve_up`/
  `embed_images`. The shared plumbing that was never CLI code in the first
  place became `src/cachedir.py` (cache layout, keys, the version stamp,
  run-params, `add_cache_args`) and `src/embed_store.py`
  (`load_embedding_matrix`, which `cluster_models.py` had been importing out
  of `test_categories.py`); `view_config` moved from `done` to `cachedir` so
  naming a view config no longer costs a torch import, and
  `as_tensor`/`embed_raw`/`embed_texts` moved into `src/embedder.py` with the
  Embedder's methods delegating to them. `classify_stls.py` went 875 → 394
  lines and now exports nothing; `import classify_stls` leaves `sys.modules` at
  186 instead of 2653, with no numpy, open3d, PIL or torch among them. (That is
  138 modules *added* over a bare interpreter's 48 — the number phase 2's
  commit message quotes. Same measurement, counted from a different zero;
  `len(sys.modules)` after the import is the one these docs use.)

- **Renders are not reproducible across pose-cache states** — **and the cause
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
- **Widen the labelled set.** 44 sampled models, ~45% exclusion rate, and the
  decisive comparisons come down to 6 disagreements. Most conclusions in
  LEARNINGS are one or two models from flipping. The `hard` set added since
  sharpens specific failures but cannot substitute — it is chosen, not sampled,
  so it can only ever answer "which method survives this", never "how often".
  Everything else here is downstream of this.
