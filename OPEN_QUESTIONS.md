# Open questions and loose ends

Threads left unpulled as of 2026-08-12. `LEARNINGS.md` records what was
settled; this records what was not. Ground truth for anything pose-related is
`up_axis_labels.json` (44 hand-labelled models, `orig` tuned / `holdout`
frozen — see LEARNINGS before quoting a number off the `orig` set).

## Ready to do — measured, decided, not yet written

- **`compress_level=1` on saved renders.** PNG encoding is ~22 s/model of a
  ~44 s/model run; `compress_level=1` is 6.1× faster, losslessly identical,
  22% more disk. About 3 hours off a 600-model run for one keyword argument.
  Nothing blocks this.
- **Raise the contact sheet to `thumb=512`.** Every VLM tested improves;
  sonnet goes from net −4 to net +3 as an arbiter. The current 256 default has
  been starving the tier for the whole project. Open sub-question: is 512 the
  peak, or does 768/1024 keep helping? Only 256 vs 512 was measured.
- **Render up-candidate tiles at a fixed size**, independent of
  `--render-size`, so pose resolution stops depending on an output setting.
  `Damaged Roofing (4).stl` resolves `+Y` at 2048 px tiles and `-Y` at 384 px,
  and the pose cache (`path + mtime + size`) cannot tell the difference. Needs
  a one-time re-resolution and probably a tile-size term in the pose cache key.
- **Rename `source: "heuristic"` to `"confirmed"`.** It currently means "the
  ensemble ran and agreed", and reads as "the ensemble was skipped" — it
  actively misled during this session. Needs a pose-cache migration.
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
- **Can a "reason" field diagnose the VLM's failures?** Ask for
  `{"tile": n, "reason": "..."}` instead of the tile alone, and read the
  reasons on models where it is stably wrong (`tile9`, `Bunker_MiniV2_Roof_`,
  `Floor`, `Concrete Chunk (6)`). Distinguishes hypotheses that accuracy alone
  cannot: is it misreading the numerals, misjudging which way is down,
  applying a positional prior, or is the *task* genuinely ambiguous for
  terrain? haiku picks `+X` ten times out of 44 when the truth is never `+X`,
  which looks like a positional prior — a reason field would say so directly.
  Cheap: one prompt change, re-run over the 44. Note it may also *improve*
  accuracy by forcing deliberation, which would confound it as pure
  diagnostics — worth measuring both with and without.
- **Gemini as a third arbiter**, once an API key is available. Same 44 models,
  same 512 px sheets, same prompt; compare against gemma4:26b 37/44,
  sonnet 37/44, haiku 26/44. Worth including the reason field from the start.
  The `_ask_claude`/`_ask_ollama` split in `pose.py` is the seam to extend.
- **Gate the VLM on the ensemble's margin, not geometry's confidence.** The
  trigger is currently `ratio > 0.6 or best < 0.02` — purely geometric — but
  the VLM then overrides the *ensemble*. Every model the VLM broke was one the
  ensemble already had right. Gating on ensemble uncertainty should keep the
  rescues and drop the damage; untested.
- **Is `Bedienkonsole` reachable at all?** A console with a large flat rear
  panel, upright `+Z`. Geometry, the ensemble, and every SigLIP probe put it
  on its back. Only the VLM has ever got it. It is the single model no
  geometric or embedding method in this project has solved.
- **Would a 512 px SigLIP help?** Renders are made at 2048 and downsampled to
  384 for `siglip2-so400m-patch14-384`. Rendering at 512 vs 2048 shifts
  embeddings by 0.976 cosine, so resolution demonstrably matters.
  `siglip2-so400m-patch16-512` is the same capacity at 1024 patches (+40%
  compute on a stage that is only 2% of runtime). Changing `--model`
  invalidates the embedding cache, so it is a full re-embed to test.

## Performance work not done

- **Thread the PNG writes.** Even at `compress_level=1` it is ~4 s of pure CPU
  per model with 15 cores idle. PIL releases the GIL during encode.
- **Prefetch mesh loads.** ~2.5 s/model, IO-bound, single-threaded; Open3D's
  reader is C++ and releases the GIL, so a small thread pool suffices.
- **Split the VLM pass from the render pass.** Demonstrated during this
  session: 40 VLM calls took 112 s (2.8 s each) with SigLIP unloaded, against
  a measured 10.1 s *model reload* per call when both compete for the 8 GB
  card. One window of the 7-hour run ran at 187 s/model.

## Structural questions

- **Should saved renders keep feeding SigLIP?** `classify_stls.py:505`
  re-embeds from disk when render files exist but the embedding cache misses.
  That puts the PNG encoder on the classifier's input path and is why lossy
  formats are unsafe (JPEG q92 moves per-view embeddings up to 0.028 cosine,
  the same order as the gap between competing categories). Decoupling costs a
  re-render on those cache misses but makes renders a true debug artifact.
- **Widen the labelled set.** 44 models, ~45% exclusion rate, and the
  decisive comparisons come down to 6 disagreements. Most conclusions in
  LEARNINGS are one or two models from flipping. Everything else here is
  downstream of this.
