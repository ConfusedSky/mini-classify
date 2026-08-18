# Wave 1 implementation reviews — 2026-08-17

Review record for the actor-refactor wave-1 tracks (branch `actor-refactor`).
Implementations by fable agents, reviews by opus agents, coordinated in
session; this file is the durable copy the design notes' finding IDs
(`B-R1-*`, `C-R1-*`, `D-R1-*`, later `A-R1-*`/`E-R1-*`) resolve to.
Verdict rule used: findings only for spec violations, drift from
`classify_stls.py` where parity was claimed, or genuine bugs; a round of
only minors closes as CLEAN-WITH-NOTES, with the notes riding to the final
whole-branch review.

## Track D — embedder (`src/embedder.py`): CLEAN-WITH-NOTES

Reviewer re-ran the GPU parity suite on the 4060 and independently probed
beyond it (all four prompt banks bitwise-equal to the old path; PIL-vs-
ndarray identical at 384/512/800/600×400, not just the tested 256).

- **D-R1-1** (minor) `src/embedder.py` — `PROMPT_TEMPLATES`/`DEFAULT_MODEL`
  duplicated from `classify_stls.py:54-58`/`:682`; the only parity guard is
  gpu-marked, which routine `-m "not gpu"` runs skip. Fix: import from
  `src.embedder` in the CLI once wave 2 rewires it, or a non-GPU equality
  test.
- **D-R1-2** (minor) — spec's Embedder attr block lacked `front_T`/`back_T`.
  **Applied to interfaces.md same day.**
- **D-R1-3** (minor) — GPU parity test covers `up_T` only; loop it over all
  four banks.
- **D-R1-4** (minor) — `text_embeds` property blocks rebinding but not
  in-place tensor mutation; document "consumers must not mutate".
- **D-R1-5** (minor) — empty-image-list `ValueError` is exact parity with
  the old path; recorded so nobody "fixes" it into drift.

Cross-track observations: Poser `_stash` leak when `embed_tiles` raises
(became C-R1-5); `--no-up-ensemble` had no gate left anywhere but
`cache_checker` (fed C-R1-1); bare `pytest` loads SigLIP (~9.5 s) — use
`-m "not gpu"` while a pipeline runs.

## Track B — cache_checker (`src/cache_checker.py`): CLEAN-WITH-NOTES

Reviewer walked `classify_stls.process()` (:1096-1160) branch by branch
against the extracted decision table: no missed branch, no inverted
condition; key builders byte-identical; `--up-axis z|y` shortcut and
`--skip-embed` arms exact (J1/Q2).

- **B-R1-1** (minor) — `pytest.importorskip("classify_stls")` makes the
  key-parity pin skippable; must be a hard import.
- **B-R1-2** (minor) — no test case exercises `n_views` with more than one
  elevation; a partial second ring would pass as complete.
- **B-R1-3** (minor) — fresh-interpreter test doesn't assert the open3d
  rendering submodule stays unimported.
- **B-R1-4** (minor, cross-track) — D11 said `pose_is_sufficient` runs over
  `Pose | None`; the store is dict-valued (JSON in/out; `route` and
  `pose_is_sufficient` subscript entries). **The doc was wrong; D11
  amended same day.**

Escalation (out of scope, against the then-landed Poser): after a fresh
resolution the Poser returned `EmbedRenderTask` unconditionally — losing
today's post-resolution cache check (:1148-1155): the warm-`.npy` skip
(a `POSE_CACHE_VERSION` bump would re-embed the whole collection) and the
`pose_changed` forced redraw. **Resolved by spec amendment same day: the
Poser returns `Resolved` and the driver re-routes through
`route(..., pose_changed=...)`.** Also confirmed: `pose_changed` can never
fire on the cached/forced poses `route` sees (assigned only inside the
insufficient branch, :1146).

## Track C — arbiter + poser (`src/arbiter.py`, `src/poser.py`): CHANGES REQUESTED

Extraction verified line-exact against every cited anchor; concurrency
clean (parked/_stash main-thread-only, no lost wakeup in `settle`, `submit`
never blocks even saturated). Confirmed correct: record_pose at park time
with unconditional re-record on fold; `fold_done` dropping cancelled
futures uncounted (required for the abort narration's honesty); `settle`'s
`future.done()` timeout-vs-error disambiguation.

- **C-R1-1** (MAJOR) — `--no-up-ensemble` unimplementable: cold files still
  reach the Poser with the ensemble off (`pose_is_sufficient(None, False)`
  → `PoseRenderTask`), the flag was a no-op, the wiring couldn't construct
  a Poser without the banks, and the geometry-only `needs_arbiter` gate had
  no home. **Decision: flag retired** (with `--up-conf`); recorded in the
  proposal's migration notes.
- **C-R1-2** (minor, cross-track) — `instrument.arbiter_call` timer/
  in-flight gauge dropped, and the driver has no seam to restore it. Fix:
  `wrap` callable on `Arbiter`, applied on the worker path.
- **C-R1-3** (minor) — park-time record makes an abort-abandoned escalation
  permanent across runs (reads sufficient next run, never retried).
  **Accepted as the trade; recorded in interfaces.md I15.**
- **C-R1-4** (minor) — nothing rejected `backend="ollama"`, which would
  overlap ollama with SigLIP on the 4060 (hard constraint). Fix:
  `VlmConfig` raises at construction; `--no-defer-arbiter` recorded
  retired.
- **C-R1-5** (minor) — `_stash` entries (~9 MB of tiles at 512 px) leak
  when `TileEmbeds` never arrives (embed failure). Fix: `Poser.drop(index)`
  called from the driver's Failure arm.
- **C-R1-6** (minor, latent) — with `min_interval>0` a paced call could
  start after `shutdown` dropped the queue; a sleeping worker also degrades
  the window. Fix: stop-event checked after the pacing sleep.

Untested branches to pin (verified by hand this round): poll on a cancelled
future; settle when the call raises `TimeoutError`; settle folding a failed
call; submit under a saturated pool; `min_interval` with `workers>1`.

## Track A — child side: review pending

Implementation round measured the I11 failure (camera rotation vs
`mesh.rotate`, world-fixed IBL; see the dated learnings entry) which led to
the rotated-copy decision now in interfaces.md; the rework and its review
land after this record's first commit and append here as `A-R1-*`.

## Track E — done: review pending

Appends here as `E-R1-*`.
