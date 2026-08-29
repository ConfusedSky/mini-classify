# Archive

Documents whose work has shipped. They are kept because things still point at
them, not because they describe the code.

Two rules follow, and they are the whole reason this directory exists:

- **Line numbers in here are as-of-then.** They index the code as it stood on
  the document's date — mostly the pre-refactor `classify_stls.py`, which was
  1277 lines and is now 425. Do not repair them, and do not read them as
  navigation. Live docs and code cite by *symbol* instead
  (CLAUDE.md §Conventions).
- **Nothing in here is evidence about today's code.** Where something became a
  durable fact it has a dated write-up in `docs/learnings/` and an index line
  in `LEARNINGS.md`; quote that. What is left here is the argument that
  produced a decision, not the state it described.

## What is in it

- **`reviews/`** — the wave-1 design and implementation review rounds, the
  renderer measurements, the past-week retrospective. Kept for one concrete
  reason: **410 finding IDs cited from live code are defined in here and
  nowhere else**, so `(K1: unconditional)` in a comment has no referent
  without them. `reviews/README.md` indexes which file defines which ID
  family.
- **`superpowers/`** — the 2026-08-10 pose-pipeline spec and plan, from before
  the actor refactor. Its "Modify: `classify_stls.py:50-71`" lines were
  instructions for work that has since landed, not references; the pipeline
  they planned is described by `docs/actor-refactor/interfaces.md` now.
- **`tri-state-pass-2.md`** — the 2026-08-21 design for the `arbiter` retry
  contract, shipped as C1–C6. Same deal as the reviews: `C1`–`C6` are cited
  from `pose`, `poser`, `cache_checker`, `driver`, the CLI and six test
  modules, and are defined only here. What the contract *is* now lives in
  `pose.pose_is_sufficient`'s docstring and in LEARNINGS, "Tri-state pass 2,
  and the new primary cache".
