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
| `ensemble.py` | Compares ways of combining the geometry and SigLIP score vectors — min-max, z-score, Borda, softmax, absolute-scaled, and a hard switch. Min-max wins, and `LEARNINGS.md` explains why that is not arbitrary. |
| `one_model.py` | Per-candidate scores for named meshes. Reach for this when one model behaves oddly. |
| `build_report.py` | Builds the standalone HTML failure report — truth tile beside each method's pick, grouped by failure mode. |
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
