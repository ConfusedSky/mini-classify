# mini-classify

Zero-shot classification of a ~1000-model printable-miniature STL collection:
render each mesh headless (Open3D/Filament, AMD iGPU), resolve its up-axis
(geometry + SigLIP ensemble, VLM arbiter for the hard ~20%), embed the views
with SigLIP (RTX 4060), score against category text embeddings.

## Read before working

- `LEARNINGS.md` — index of dated session write-ups in `docs/learnings/`.
  Every measured number in this repo has its story there; a citation like
  "LEARNINGS, Overlap and the thermal ceiling" resolves through the index.
- `OPEN_QUESTIONS.md` — open work. Entries are amended in place with the
  answer (often ~~struck through~~), not deleted.
- `docs/actor-refactor/` — the pipeline refactor: `actors_proposal.md`
  (stage boundaries + what the spikes measured), `data_structures.md` (the
  settled shapes), `interfaces.md` (the calling conventions between the
  modules that hold them), `renderer_alternatives.md` (renderer research).
- `docs/reviews/` — dated review notes.
- `eval/README.md` — the measurement harnesses; they import the production
  code path, never a copy.

## What the project actually is

The pipeline's product is the **caches** (`embed-cache*/`: pose-cache.json,
per-view `.npy` embeddings, renders). Real querying happens interactively in
`test_categories.py`; the `results.csv` top-3 scoring in `classify_stls.py`
is a thin consumer, whatever the code's framing suggests. Weigh work
accordingly: cache-building throughput and REPL quality first.

## Conventions

- A measured finding gets a dated write-up in `docs/learnings/` plus an index
  line in `LEARNINGS.md`. Spike scripts live in `eval/` with a row in
  `eval/README.md`; raw output goes to `eval/out/` (gitignored) — the
  write-up is the record.
- Make file changes with the Edit/Write tools, not sed/heredoc scripts: the
  diff is the review surface. Bulk mechanical rewrites of generated data are
  the exception, and show a diff summary after.
- Ground-truth labels load through `common.load_labels()` — never re-derive
  them from a sample index; the collection grew mid-session once and the same
  seed stopped drawing the same models.

## Hard-won constraints (measured; don't relitigate without new numbers)

- **`OffscreenRenderer` teardown aborts; creation does not.** Multiple
  renderers coexist and render correctly in one process (four measured,
  `docs/reviews/2026-08-13.md` §3.1) — the abort is Filament throwing from a
  destructor. Keep renderers alive for the process lifetime; never destroy
  one.
- **Rendering runs on the AMD iGPU, SigLIP on the 4060.** They do not
  contend; the split is a measured win — keep it.
- **`render_to_image` holds the GIL** (~85–92%). Threads cannot overlap
  rendering; a child process can (measured 1.17–1.21×, thermally capped).
- **`pose.py` never imports `classify_stls`** — no rendering or model code
  in it.
- **`read_binary_stl`: the file outvotes a lying header** (commit `bd6be81`,
  for Materialise Magics files). The remaining guards — ASCII detection,
  finite/magnitude bound, whole-record remainder — are deliberate; don't
  loosen them.
- **ollama and SigLIP cannot share the 4060** (10.1 s reload vs 0.49 s of
  inference); never overlap them.
