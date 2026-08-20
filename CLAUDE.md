# mini-classify

Zero-shot classification of a ~1000-model printable-miniature STL collection:
render each mesh headless (Open3D/Filament, AMD iGPU), resolve its up-axis
(geometry + SigLIP ensemble, VLM arbiter for the hard ~20%), embed the views
with SigLIP (RTX 4060), score against category text embeddings.

**The shape of the code** (the actor refactor, merged 2026-08-18): a
self-contained `src/` package holds the pipeline — a sequential driver in the
parent that owns admission and drains results, one spawned render child doing
all Open3D work, and SigLIP in the parent. `classify_stls.py` is the CLI entry
and **nothing imports it**; the other top-level files are tools
(`test_categories.py` is the REPL, plus `cluster_models.py`,
`migrate_cache_keys.py`, `unpack_models.py`). Start from
`docs/actor-refactor/interfaces.md`: it is the spec of what exists, not a
proposal.

## Read before working

- `LEARNINGS.md` — index of dated session write-ups in `docs/learnings/`.
  Every measured number in this repo has its story there; a citation like
  "LEARNINGS, Overlap and the thermal ceiling" resolves through the index.
- `OPEN_QUESTIONS.md` — open work. Entries are amended in place with the
  answer (often ~~struck through~~), not deleted.
- `docs/actor-refactor/` — the refactor, **shipped**: `interfaces.md` is the
  live spec (calling conventions, the wiring, the import-rule table — read it
  before adding a module or an import), `data_structures.md` the message and
  driver shapes, `actors_proposal.md` the argument and what the spikes
  measured, `renderer_alternatives.md` renderer research. Findings cited as
  `I3`/`K6`/`C-R1-4`/`F-7` resolve through `docs/reviews/`.
- `docs/reviews/` — dated review notes, including the implementation reviews
  every finding ID resolves to.
- `docs/api/surface.md` — the query API's **spec**: the REPL's querying as an
  HTTP surface for `~/Documents/model-browser` to search against, reviewed by
  both sides against their own code. Built 2026-08-19 as `src/query.py`,
  `src/collection.py`, `src/api.py` and the `serve_api.py` entry point;
  `docs/api/implementation.md` is the phased plan, with each phase's decisions
  recorded where they were made. Phases 0–2 are done, phase 3 (a live run at
  scale) and phase 4 are not.
- `docs/cache-rebuild.md` — the debt that is only payable when the whole cache
  is regenerated: shims that exist so an old cache keeps working, and which a
  rebuild is the one chance to delete. Read it **before** planning a rebuild,
  and add to it whenever you write a back-compat path.
- `eval/README.md` — the measurement harnesses. They call the production code
  through `eval/rig.py`, which builds the real `Renderer`/`Embedder` — true
  again since 2026-08-18, when the CLI's parallel single-process render/embed
  path was deleted and the harnesses moved onto `src/`.

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
- **Nothing imports `classify_stls.py`** (eval-debt cleanup, 2026-08-18). It is
  the CLI entry and exports nothing; `spawn` re-imports it as `__mp_main__` in
  the render child, so its module scope is the child's startup cost — stdlib,
  `tqdm`, `src.instrument`, `src.identity` and `src.cachedir`, with everything
  torch- or open3d-owning imported inside the function that uses it (measured:
  `import classify_stls` adds 138). What tools share
  lives in `src/cachedir.py` (cache layout, keys, run-params, the shared
  argparse block), `src/embed_store.py` (reading `.npy` back) and
  `src/embedder.py` (text embeddings). `src/pose.py` holds no rendering or
  model code either, and **`src/` reaches outside itself for nothing** — the
  root is on `sys.path` for the tools, not for the package.
- **Import weight is a design constraint here, and `interfaces.md`'s
  import-rule table is where it is enforced.** Two things that table settles,
  both learned by measurement: quote module counts as what an import *adds*
  over a bare interpreter's 48, never the total; and **a deferred import only
  fixes anything when the function's own signature already implies the
  dependency** — otherwise it moves the cost onto callers instead of removing
  it. `up_axis_scores(mesh)` qualifies (you cannot hold an Open3D mesh without
  open3d); a helper returning a *string* from a torch-owning module did not,
  and charged 836 modules to every tool that built an argument parser. The
  numbers, the decomposition that redirected the fix, and the third failure
  mode — a deferral whose comment names a beneficiary that pays anyway — are
  in the dated learnings entry.
- **Ctrl-C is the parent's alone; the render child ignores `SIGINT`.** A
  terminal signals the whole foreground process group, and the child's
  `except Exception` cannot catch a `KeyboardInterrupt` — so an unshielded
  child died mid-render and the parent wrote every in-flight file to the CSV
  as a render failure it never had. Error rows are retirements: they outlive
  the run that invented them.
- **Every join on the render child is bounded.** `fail_outstanding` reaches
  quiescence by *killing* the child, not by it going idle, and a child wedged
  in Filament's EGL/amdgpu ioctls sits in uninterruptible `D` state where
  SIGKILL neither kills nor reaps — an untimed `join()` there hangs past
  `done.flush()` and loses the run's work. See the dated learnings entry; note
  `multiprocessing`'s own `_exit_function` joins untimed too, which is why the
  child is discarded from `_children` on the give-up path.
- **Filament's output depends on the scene's draw history**, not just its
  current state. Measured three independent ways: across pose-cache states,
  across batch position (the *same* binary differs from itself by 7.0e-03 per
  embedding component when a model renders in company rather than alone), and
  between two code paths that agree on geometry, camera, material and framing.
  Renders are reproducible for a fixed sequence and not otherwise — never
  compare pixels or embeddings across runs that drew different things first.
- **`read_binary_stl`: the file outvotes a lying header** (commit `bd6be81`,
  for Materialise Magics files). The remaining guards — ASCII detection,
  finite/magnitude bound, whole-record remainder — are deliberate; don't
  loosen them.
- **ollama and SigLIP cannot share the 4060** (10.1 s reload vs 0.49 s of
  inference); never overlap them.
