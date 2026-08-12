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
| `front_first.py` | Tests finding the *front* first and letting it constrain up, against up-first and a 4-azimuth control, over all 24 (front ⊥ up) orientations — which cost the same six geometry uploads as today's six tiles. Front-first loses 9 of 44; the chosen front is perpendicular to the true up on only 38/49, so a fifth of the time it excludes the right answer outright. See LEARNINGS. |
| `arbiter_gate.py` | Sweeps a gate on the **ensemble's** margin (`top1−top2` of the combined vector) against the current geometry-confidence gate, scoring pipeline accuracy against how often it fires — one firing is one API call. Reuses every recorded VLM run, so it needs no API access. Margin < 0.4 matches the geometry gate's accuracy on 9 calls instead of 24. |
| `gauntlet.py` | Runs one label set (`--set hard`/`orig`/`holdout`/`all`) through every method at once — geometry, both backbones at every render size, and every arbiter (gemma, haiku, sonnet, three Gemini) at both sheet sizes — as a per-model table. Needs `ollama serve` for the gemma row and gcloud ADC for Gemini; it skips what it cannot reach. A *per-model* instrument, not an accuracy measurement. |
| `backbone_memory.py` | VRAM per SigLIP tower: params, image tokens, resident weights, peak allocated at a given batch. Run it with the GPU idle — `ollama` holding gemma4:26b (6.8 GB) makes every figure meaningless. |
| `ensemble.py` | Compares ways of combining the geometry and SigLIP score vectors — min-max, z-score, Borda, softmax, absolute-scaled, and a hard switch. Min-max wins, and `LEARNINGS.md` explains why that is not arbitrary. |
| `one_model.py` | Per-candidate scores for named meshes. Reach for this when one model behaves oddly. |
| `build_report.py` | Builds the standalone HTML failure report — truth tile beside each method's pick, grouped by failure mode. |
| `gold_upright.py` | Renders every label in the orientation it asserts — `rotation_to_z_up(label)`, 3 azimuths — into one self-contained HTML page, so the ground truth itself can be eyeballed. This is how you check a new label before trusting a number measured against it. `--html` rebuilds the page from existing renders. |
| `light_probe2.py`, `light_probe3.py` | Superseded. Compared fill-light strategies before `FILL_INTENSITY` landed; kept because they document how the indirect-light-as-fill decision was measured. |

## Watch out

- **Contact sheet resolution changes the answer — for some models.**
  `pose.make_contact_sheet` defaults to `thumb=256`, which starves sonnet
  (27/44 → 37/44 at 512, net −4 → net +3 as an arbiter). It barely touches
  Gemini: +2 for each of the three, with 3.5-flash returning an identical
  answer on 42 of 44 models across both sizes. State the sheet size in any VLM
  comparison, and the model in any sheet-size comparison.
- **Scaled numerals are part of the 512 px result.** `common.contact_sheet`
  scales the tile numbers with the tile; `pose.make_contact_sheet` uses PIL's
  fixed ~11 px bitmap face, which is unreadable on a 1536×1024 sheet. A naive
  `thumb=512` measures worse than 256 — see OPEN_QUESTIONS.
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
