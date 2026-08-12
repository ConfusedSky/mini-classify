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
- **Would a 512 px SigLIP help?** — `siglip2-so400m-patch16-512` is worth +1 of
  44 (p=0.5) but is *resolution-invariant* where `patch14-384` flips three
  models on render size alone. Memory is a wash: both 4.3 GB on disk, 2.19 GB
  resident, differing by 1 MiB of weights; the +40% image tokens cost ~1.02×
  memory and ~24% time. `eval/backbone_sweep.py`, `eval/backbone_memory.py`.
- **Gate the VLM on the ensemble's margin** — measured. `margin < 0.4` matches
  the geometry gate's accuracy on 9 calls instead of 24, and stops a weak
  arbiter from going net negative (haiku@256: 30/44 → 39/44). Still to be
  *written* — see below. `eval/arbiter_gate.py`.

## Ready to do — measured, decided, not yet written

- **`compress_level=1` on saved renders.** PNG encoding is ~22 s/model of a
  ~44 s/model run; `compress_level=1` is 6.1× faster, losslessly identical,
  22% more disk. About 3 hours off a 600-model run for one keyword argument.
  Nothing blocks this.
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
- **Raise the contact sheet to `thumb=512`** — with the caveat that the size
  matters far less than first measured. sonnet gains 10 of 44 going 256 → 512;
  every Gemini model gains 2, and gemini-3.5-flash returns an identical answer
  on 42 of 44 models across both sizes. 512 is the better default, but it is
  not the pipeline-wide starvation the first measurement implied — it was
  starving *that model*. Acting on it still means re-resolving cached
  `source: "vlm"` poses.

  **The numerals must scale with the tile, or this makes things worse.**
  Implemented in `eval/common.contact_sheet`; `pose.make_contact_sheet` still
  uses PIL's fixed ~11 px bitmap face, which is illegible on a 1536×1024 sheet
  — a naive `thumb=512` in production would measure *worse* than 256. Port the
  scaling when adopting:

  ```python
  size = max(11, thumb * 44 // 512)      # 44px at 512, 22px at 256, 88px at 1024
  try:
      FONT = ImageFont.load_default(size=size)          # Pillow >= 10.1
  except TypeError:
      FONT = ImageFont.load_default()                   # bitmap, fixed ~11px
  draw.text((x + thumb // 36, y + thumb // 64), str(i + 1), fill="red", font=FONT)
  ```

- **Render up-candidate tiles at a fixed 384 px**, independent of
  `--render-size`. Now measured rather than assumed: neither tower gains
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

## Performance work not done

- **Thread the PNG writes.** Even at `compress_level=1` it is ~4 s of pure CPU
  per model with 15 cores idle. PIL releases the GIL during encode.
- **Prefetch mesh loads.** ~2.5 s/model, IO-bound, single-threaded; Open3D's
  reader is C++ and releases the GIL, so a small thread pool suffices.
- **Split the VLM pass from the render pass** in `classify_stls.py`. Measured
  again this session from the other side: gemma4:26b sits at 6818 MiB resident
  on a 7834 MiB card, so it and SigLIP (2.2 GB) genuinely cannot coexist. Every
  harness in `eval/` now phases render → SigLIP → VLM by construction
  (`common.build_tiles` caches the pixels so the towers never overlap); the
  production path still interleaves them.

## Structural questions

- **Should saved renders keep feeding SigLIP?** `classify_stls.py:505`
  re-embeds from disk when render files exist but the embedding cache misses.
  That puts the PNG encoder on the classifier's input path and is why lossy
  formats are unsafe (JPEG q92 moves per-view embeddings up to 0.028 cosine,
  the same order as the gap between competing categories). Decoupling costs a
  re-render on those cache misses but makes renders a true debug artifact.
- **Widen the labelled set.** 44 sampled models, ~45% exclusion rate, and the
  decisive comparisons come down to 6 disagreements. Most conclusions in
  LEARNINGS are one or two models from flipping. The `hard` set added since
  sharpens specific failures but cannot substitute — it is chosen, not sampled,
  so it can only ever answer "which method survives this", never "how often".
  Everything else here is downstream of this.
