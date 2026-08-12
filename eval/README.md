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
| `ensemble.py` | Compares ways of combining the geometry and SigLIP score vectors — min-max, z-score, Borda, softmax, absolute-scaled, and a hard switch. Min-max wins, and `LEARNINGS.md` explains why that is not arbitrary. |
| `one_model.py` | Per-candidate scores for named meshes. Reach for this when one model behaves oddly. |
| `build_report.py` | Builds the standalone HTML failure report — truth tile beside each method's pick, grouped by failure mode. |
| `light_probe2.py`, `light_probe3.py` | Superseded. Compared fill-light strategies before `FILL_INTENSITY` landed; kept because they document how the indirect-light-as-fill decision was measured. |

## Watch out

- **Contact sheet resolution changes the answer.** `pose.make_contact_sheet`
  defaults to `thumb=256`, which starves the VLM: sonnet went 27/44 → 37/44 at
  512, and from net −4 to net +3 as an arbiter tier. Any VLM comparison must
  state its sheet size.
- **Don't run the VLM and SigLIP against the same GPU.** On an 8 GB card they
  evict each other; a measured reload costs 10.1 s against 0.49 s of
  inference. Run the SigLIP phase, then the VLM phase — 40 calls took 112 s
  that way against a three-hour stall in the contended run.
- **`orig` labels are tuned, `holdout` labels are not.** The probes and the
  combination scheme were selected against `orig`. Quote pooled or holdout
  numbers; the ensemble scored 91% on `orig` and 81% on the holdout.
