## Canonical pose: confidence, VLM arbiter, front view (2026-08-11)

- **`sample_points_uniformly` interpolates vertex normals by default**, which
  blends normals across creases (a cone's base rim inherits mantle tilt) and
  destroys the flat-base up-detection signal on vertex-sharing meshes. Pass
  `use_triangle_normal=True`: measures actual face orientation and drops the
  hidden compute-vertex-normals-first precondition. Real STLs mostly dodge
  this (the reader duplicates vertices per facet) — procedural test meshes
  don't, which is how the tests caught it.
- **Up-detection confidence = runner-up/best flat-base score ratio.** Cone
  (one flat face) is decisive; torus measured 0.98, cylinder ≈1.0 (±Z caps
  identical). Ratio > 0.6 or best score < 0.02 escalates to a VLM arbiter;
  everything else never pays for one. A box is *maximally* ambiguous under
  this scorer (six flat faces) — don't use one as a "decisive" test fixture.
- **Warm caches can survive a pose-pipeline redesign.** The heuristic's
  answer is a deterministic function of the file, so heuristic-resolved
  poses keep the legacy cache-key token ("auto"); only VLM *overrides* —
  where the render genuinely differs — re-key as `vlm:x,y,z`. Rebuilding the
  pose logic invalidated zero existing embeddings.
- **Front view is metadata, not a render decision.** Score cached per-view
  embeddings against front/back text prompts (mean front-sim − mean
  back-sim, argmax) and record the winning index; REPL links and contact
  sheets reorder views at display time. No re-render, caches untouched, and
  the choice retunes for free when prompts change (after deleting
  pose-cache.json — its identity doesn't cover prompts).
- Don't hardcode ollama model tags: the plan assumed `gemma3`, the actual
  pull landed as `gemma4:26b` — hence `--pose-vlm-model`.

Verification results (2026-08-11, all end-to-end on real renders):

- **Poses/fronts confirmed by eye**: bunny detected Y-up (conf 0.16) and
  rendered upright; its `front_view=3` is the face-toward-camera view (view 1
  is the rear). Torus (conf 0.97) escalated to the arbiter as designed.
- **Disable thinking on ollama VLM calls.** gemma4:26b with thinking on
  spent every token in the `thinking` field and returned *empty* content
  under `format` constraints — 3859 tokens / 281 s vs 7 tokens / 25 s with
  `"think": false`. The empty answer degrades identically to a timeout, so
  the symptom (heuristic fallback) hides the cause. `_ask_ollama` now sends
  `think: false` and retries without the field if the server 400s on it;
  timeout bumped 120→300 s for CPU-offloaded models.
- **VLM + SigLIP contend for VRAM (8 GB GPU, 17 GB model): load order
  decides who wins.** Pipeline order (SigLIP first, ollama call later) works
  — ollama CPU-offloads what doesn't fit. Warming ollama *first* leaves
  SigLIP nothing to load into → hard CUDA OOM.
- **A fresh VLM override must invalidate saved renders.** `--save-renders`
  reused on-disk PNGs on an embedding-cache miss, but those were rendered
  under the old pose — silent wrong-pose embeddings. Guarded: a newly
  resolved override forces the re-render. Since renders became debug output
  the correctness half is gone (the override re-keys the embedding by
  itself), but the guard stays so the files on disk show the pose that was
  actually used; it now also covers `source=ensemble`.
- Claude CLI backend works in `-p` mode as written (no file-read permission
  issue; 22 s). On the torus it picked "lying flat" where gemma picked
  "standing on edge" — both defensible for a symmetric shape; arbiter answers
  on degenerate geometry are a coin flip between valid conventions.
- Live-transport test doubles as a degrade-path check: with ollama up but
  the default `gemma3` tag unpulled, `ask_vlm_up` returns None (heuristic
  kept), never raises.

