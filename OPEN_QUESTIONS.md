# Open questions and loose ends

Threads left unpulled as of 2026-08-12. `LEARNINGS.md` records what was
settled; this records what was not. Ground truth for anything pose-related is
`up_axis_labels.json` — 49 hand-labelled models in three sets: `orig` (23,
tuned on), `holdout` (21, frozen), `hard` (5, hand-picked failures, not a
sample of anything). Read LEARNINGS before quoting a number off `orig`, and
never pool `hard` into an accuracy.

## Settled since the last revision

Moved out of this file; the measurements are in `LEARNINGS.md`.

- **Gemini as a third arbiter** — done via Vertex ADC, no API key needed.
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
- **Gate the VLM on the ensemble's margin** — measured. `margin < 0.4` matches
  the geometry gate's accuracy on 9 calls instead of 24, and stops a weak
  arbiter from going net negative (haiku@256: 30/44 → 39/44). Still to be
  *written* — see below. `eval/arbiter_gate.py`.

- **Raise the contact sheet to `thumb=512`** — done, with the scaled numerals
  it depends on, now in `pose.make_contact_sheet` rather than only in the eval
  harness. sonnet gains 10 of 44 across that step and gemma 3; gemini-3.5-flash
  barely notices, so the size still belongs in any report of a VLM number.
- **Wire a Gemini backend into the arbiter** — done. `--pose-vlm gemini`,
  default `gemini-3.5-flash`, ADC auth, project from `--gemini-project` /
  `$GOOGLE_CLOUD_PROJECT` / `gcloud config`. **`auto` now prefers it**, falling
  back to ollama and then to no arbiter, because it is the only one that beats
  the ensemble. It bills per call (~$0.30 per full-collection run at ~120
  escalations), so the selection is always announced and `--pose-vlm ollama`
  opts out at 41/44. With the 512 px sheet that local path is net positive too,
  so the "run with `--pose-vlm off`" advice no longer applies either way.
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

- **Gate the arbiter on the ensemble's margin** (`top1 − top2` of
  `_unit(geo) + _unit(siglip)`) instead of `needs_arbiter(ratio, best)`.
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
- **Rename `source: "heuristic"` to `"confirmed"`.** It currently means "the
  ensemble ran and agreed", and reads as "the ensemble was skipped" — it
  actively misled during a previous session. Needs a pose-cache migration.
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
  `test_categories.py` already produces those ranked lists from cached
  embeddings, so the harness mostly exists.
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
  Note this decoupling is currently blocked anyway — one `OffscreenRenderer` per
  process, fixed size at construction, so pose tiles at 384 and views at 2048
  cannot coexist in one process (see `docs/masa/actors_proposal.md`).

## Performance work not done

- **Thread the render writes.** Largely obsolete: the default is `jpg` now, at
  0.13 s/model against PNG's 23 s. Still ~4 s of single-threaded CPU per model
  if someone runs `--render-format png`, and PIL releases the GIL during encode.
- **Prefetch mesh loads** — done (`MeshPrefetcher`, `--prefetch`, default 2).
  Not currently the pacer: measured load/GPU ratio 0.37 on the Loot Studios
  subset (0.67 s load against 1.81 s GPU per model). It would only become one on
  the heavy tail, where a p99 mesh loads in 15.4 s. **Do not multi-thread it
  without measuring that ratio on the input in question** — one loader thread
  ahead of the GPU is enough whenever the ratio is under 1.
- **Overlap the VLM arbiter with everything else** — done and measured at
  **28%** end to end (631 s → 456 s on 74 models). See LEARNINGS for the A/B and
  for the bug the A/B caught, where the first version ran *slower* than what it
  replaced.

  Still open: **the GPU is idle ~73% of the main pass** and about 2 s per model
  is unattributed. GPU work is ~2.7 s of a 5.4 s model; mesh load (0.67 s) is
  prefetched, JPEG is 0.13 s, geometry is 0.01 s. Where the rest goes needs a
  profiler on a live run — `py-spy` requires root here (`ptrace_scope=1`), so it
  wants a terminal, not a harness. This is where any further speedup lives, and
  it should be attributed before anything is built.
- **Split the VLM pass from the render pass** in `classify_stls.py`. Measured
  again this session from the other side: gemma4:26b sits at 6818 MiB resident
  on a 7834 MiB card, so it and SigLIP (2.2 GB) genuinely cannot coexist. Every
  harness in `eval/` now phases render → SigLIP → VLM by construction
  (`common.build_tiles` caches the pixels so the towers never overlap); the
  production path still interleaves them.

## Structural questions

- **Renders are not reproducible across pose-cache states.** A cold pose cache
  renders the six up-candidate tiles through the same `OffscreenRenderer` first,
  and the view renders that follow differ from a warm-cache run by up to 0.0098
  per embedding component — a fifth of the ~0.03 gap between competing
  categories. Measured on all three test STLs, identical on the old code, so it
  is long-standing rather than new. Every embedding in the live cache was
  therefore computed in whichever state that file happened to hit. Cheapest
  honest fix is to warm the renderer the same way on both paths; widening the
  cache key would only make the irreproducibility explicit, not remove it.
- **Widen the labelled set.** 44 sampled models, ~45% exclusion rate, and the
  decisive comparisons come down to 6 disagreements. Most conclusions in
  LEARNINGS are one or two models from flipping. The `hard` set added since
  sharpens specific failures but cannot substitute — it is chosen, not sampled,
  so it can only ever answer "which method survives this", never "how often".
  Everything else here is downstream of this.
