## Old against new: what the refactor moved, and what moves anyway (2026-08-18)

The actor refactor replaced a 1277-line `classify_stls.py` with a CLI plus
eight modules under `src/`, a render child process and a sequential driver.
Wave 3's question was the only one that matters to a cache the collection has
already paid for: **does the new pipeline produce the same artifacts as the
old one?** Not "does it pass its tests" — the tests are written against the new
shapes. The comparison is `git show main:classify_stls.py` run for real,
against the branch run for real, on the same STLs into two cache directories,
compared byte for byte.

Nine comparisons, three models (`torus`, `blocky_building`, `bunny` — the
repo's `test-stls/`), `--render-size 512 --views 4 --elevations 20
--save-renders`, pose VLM off, cold caches unless stated. Arms: the three
models as one batch (twice per pipeline, for a noise floor, plus a
warm-pose-cache repeat), and each model alone (cold, and warm). Harness and raw
caches are in the session scratchpad; the figures below are the record.

### The part with money on it: nothing keys differently

| artifact | old vs new |
|---|---|
| embed cache keys (`embeds/<sha1>.npy`) | **identical**, every arm |
| `run-params.json`, `cache-meta.json` | identical |
| renders directory name (`512px-4v-e20`) | identical |
| pose-cache entry keys (`file_identity`) | identical |
| file inventory of the cache dir | identical |

So the live collection's ~2900 cached embeddings stay addressable: the new code
looks in exactly the places the old code wrote. That was the failure worth
fearing — a silent key change re-renders and re-embeds the whole library, hours
and real money — and it did not happen.

### Poses: the answer never moved; the margin sometimes did

In **every** comparison, for **every** model: `up`, `source`, `confidence` and
`v` are identical. The only fields that ever differ are `margin` and
`front_view` — the two things computed from pixels rather than from geometry.

### The decisive control: one model at a time, cold

Batch runs mix three models' work together, so a difference there could be the
refactor or could be the batch. Run each model alone and that confounder is
gone:

| model (cold, alone) | pose entry | 4 renders | `.npy` | CSV row |
|---|---|---|---|---|
| `torus` | identical | **byte-identical** | **byte-identical** | identical |
| `blocky_building` | identical | **byte-identical** | **byte-identical** | identical |
| `bunny` | identical | 186–723 subpixels differ, ≤5 levels | max abs 4.88e-04 | 4th decimal |

Two of three models are byte-identical end to end — pose entry, all four
renders, the embedding, the CSV row. A pipeline that had changed the pixels
could not produce that. And run each model alone against a *warm* pose cache —
no candidate tiles rendered at all — and the third joins them: `bunny`'s
embedding is byte-identical old vs new, as is `torus`'s.

`bunny` is the curved mesh, and it is cold-and-alone that separates it: 186 to
723 of 786,432 subpixels per view, at most 5 levels of 255, and 4.88e-04 on the
worst embedding component.

### The batch arms differ more — and so does the old code from itself

| comparison | max abs Δ per embedding component |
|---|---|
| old batch vs new batch | 1.5e-03 (`torus`), 6.6e-03 (`bunny`), 8.4e-03 (`blocky_building`) |
| **old batch vs old alone, same code** | 5.5e-03 (`torus`), 6.2e-03 (`bunny`) |
| **old warm batch vs old warm alone, same code** | **7.0e-03** (`bunny`) |

The second and third rows are the finding. They contain no new code at all:
`main:classify_stls.py` against itself, differing by the same order of
magnitude as the old-vs-new comparison, purely because the model was rendered
in company rather than alone. The pose cache moves with it — the same old
binary gives `torus` `margin` 0.0724 in the batch and **0.0791** alone, and
`front_view` **2** in the batch against **3** alone.

Both pipelines are *bit-reproducible against themselves*: two identical batch
runs of the old code agree on every byte of every artifact, and so do two of
the new. The noise floor within one code path and one arrangement is exactly
zero. What is not stable is the *arrangement*.

### Mechanism: the renderer's output depends on what it drew before

`OPEN_QUESTIONS.md` already records this as a pose-cache-state effect — a cold
run shoots six candidate-up tiles through the `OffscreenRenderer` before the
view renders and a warm run does not, and the views differ by up to 0.0098 per
embedding component as a result. These runs generalise it: **any** change to
the sequence of draws preceding a view changes that view, and batch position is
such a change. Measured at full strength on unmodified old code, so it is a
property of this renderer, not of the refactor.

The refactor changes that sequence too, by construction. The old path rotated
the resident mesh in place and re-uploaded it; the new `Renderer.views` copies
the mesh, rotates the copy, adds it under `ROTATED_NAME`, shoots, and removes
it (I11, `docs/learnings/2026-08-17-camera-rotation-and-the-world-fixed-fill.md`),
and the driver interleaves up to `WINDOW` files' work. So the new pipeline
reaches a known regime by a new route rather than opening a new one.

### The honest caveat

`bunny` cold-and-alone is the one number this framing does not fully absorb.
4.88e-04 is genuinely new: no old-vs-old arrangement measured here reproduces
it — the old code's self-comparisons across arrangements land at 5.5e-03 to
7.0e-03, an order of magnitude *larger*, and none of them lands near 4.9e-04.
It is 20× below the 0.0098 the pose-cache-state entry documents, and it moved
no rank, but "smaller than a known effect" is not "the same effect", and it is
recorded here as unexplained rather than dismissed.

### What it costs the output: nothing that ranks

Top-3 category **ordering is identical for all three models in all nine
comparisons**. The scores move in the 4th decimal (`blocky_building` 0.1192 →
0.1197, `bunny` 0.0965 → 0.0967, `torus` 0.0890 → 0.0888), which is a tenth of
the ~0.03 gap between competing categories.

### Wall clock: this sample cannot answer it

The new pipeline came in ~1–3 s slower on a three-model run. That number is
SigLIP's load and a spawned child's startup, amortised over three models
instead of a collection; the render child's overlap has nothing to overlap
with at n=3. The throughput claim for the boundary is the spikes' **1.17–1.21×**
on cold-run wall clock (LEARNINGS, "Overlap and the thermal ceiling"), measured
where it can be seen. Nothing here contradicts or confirms it.

### What this licenses

Keep the existing `embed-cache*/` directories. The keys match, the poses match,
the ranking matches, and the pixel differences that exist are the ones the
renderer was already producing between any two arrangements of the same work.
What it does **not** license is treating a margin difference at the 1e-03 level
as evidence about anything: this repo's renderer does not produce the same
image twice unless it drew the same things before it, and the pose margins
inherit that.
