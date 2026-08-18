# Wave 1–2 implementation reviews — 2026-08-17/18

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

**Delta round B-R2** (pose_changed + flag retirement + B-R1 minors):
CLEAN-WITH-NOTES. `pose_changed` verified against `:1157` including the
one leak path (cannot escape the renders_wanted guard — case-pinned);
the second-call termination invariant confirmed (ensemble-always means a
re-routed file is always sufficient, no loop back to `PoseRenderTask`).
**B-R1-3 retracted with measurement**: `import open3d` eagerly imports the
visualization tree, so the specified assertion could never hold; the
substitution (no torch/classify_stls/src.renderer + an eager-import canary)
accepted. Forced-axis `pose_changed` ruled unreachable-by-construction and
harmless — no guard, since raising would convert a driver bug into a lost
row. **B-R2-1** (minor, applied): the gc scan for renderer objects could
never fail (pybind11 instances are not GC-tracked); deleted, with "no
renderer object" resting on the documented abort backstop.

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

**Delta round C-R2** (fix round + the `Resolved.pose_changed` addendum):
CLEAN-WITH-NOTES, loop closed. All six C-R1 findings verified resolved
(the five hand-checked branches now pinned as tests); the retirement of
`--no-up-ensemble` confirmed complete in `src/` and the design notes;
the `wrap` placement ruled structurally identical to today's `timed`
closure. The `pose_changed` mapping matched `classify_stls.py:1146`
across the full eight-cell truth table — exact because it reads the
*recorded source*, not the exit taken (a paid confirmation of a SigLIP
pick is correctly `True`; that quadrant is the one unpinned cell). No
re-route loop (`_make_pose` always writes a margin). Minors riding to
the final review:
- **C-R2-1** (applied) — one §Poser sentence still named the deleted
  `EmbedRenderTask` return; now reads `Resolved`/driver.
- **C-R2-2** (applied, doc+docstring) — `drop` cancels the parked future,
  so wave 2's `fail_outstanding` must never call it: N3 depends on parked
  files surviving a dead child so their paid answers fold before flush.
  Spelled out at both the spec comment and the method docstring.
- **C-R2-3** (deferred; inert at `min_interval=0`) — the pacing sleep is
  uninterruptible: a worker parked in it at abort isn't `done()`, inflates
  the narration count, and burns `FOLD_S` on rate-limit wait. Fix when
  pacing goes live: `self._stopping.wait(wait)` instead of `sleep`.
- Untested quadrant worth one assertion: confirmation on a SigLIP park
  ⇒ `pose_changed=True` (the cell that distinguishes read-the-source from
  confirmation-means-unchanged).

## Track A — child side (`src/loader.py`, `src/renderer.py`, `src/render_child.py`): CLEAN-WITH-NOTES

Implementation round measured the I11 failure (camera rotation vs
`mesh.rotate`, world-fixed IBL; the dated learnings entry has the numbers)
which drove the rotated-copy decision; a rework round rebuilt `views()` to
it, byte-identical to the reference in 16/18 nopost cases with the two
exceptions exactly matching their own repeat floor (18/18 with IBL off).
Review verified the loader extraction byte-for-byte (every
`read_binary_stl` guard intact), walked the exception boundary (no
reachable gap), and closed a gap the eval structurally cannot see:
hidden resident originals left in the scene are **pixel-neutral**
(controlled interleaved rerun on the iGPU — hidden-present vs churn-only
byte-identical in every pairing). All eight flagged interpretations
sustained, including `save_renders` never raising (production parity; a
failed save self-heals via the next run's redraw).

- **A-R1-1** (minor, applied) — `mesh_nbytes` omitted `triangle_normals`
  (populated by `compute_vertex_normals`), undercounting residency ~13%
  (bunny: 86.7% of real bytes); the test had pinned the undercount as the
  contract. One term added.
- **A-R1-2** (minor, applied) — attribution counts were off: 11 (not 12)
  noibl cases collapse to 1/255, and the four that don't are bunny ±Y
  (7–8/255) and ±X (17–18/255), not "three at ±X". Deltas/percentages were
  correct; learnings entry + renderer docstring corrected.
- **A-R1-3** (minor, applied) — learnings self-contradiction ("all 18" vs
  its own 16/18 heading) fixed to name the two floor-bounded cases.
- **A-R1-4** (minor, applied) — interfaces.md still wrote
  `renderer.views(lm, pose.up)`; now `views(lm, index, up)` with the
  invariant-2 reason (residency and Release key on index).
- **A-R1-5** (minor, applied) — the eval's PASS criterion excluded
  repeat-unstable cases unconditionally, which could swallow a real
  difference riding on an unstable case; exclusion now requires the path
  delta to stay within the case's own repeat floor (today's data
  satisfies it: 27/27 and 70/70 exactly).

## Track E — done (`src/done.py`): CHANGES REQUESTED

Scoring/flush/retirement parity verified line-by-line (pool_sims,
view_config, error rows, BaseException-unlink, J2/Release single choke
point all exact); the front_view write goes through the store's entry
dict, so the `Pose.from_cache` copy hazard does not apply.

- **E-R1-1** (MAJOR) — forced `--up-axis z|y` reads a stale auto pose from
  the store: `_save_embeds` files fresh embeddings under the auto
  up-token (route looks under the forced token → every forced run
  re-embeds and strands `.npy`s), and `_score` returns the auto entry's
  cached `front_view`. Today's code never consults the store on the
  forced path (`classify_stls.py:1101-1102` ephemeral entry). Fix: derive
  the token from the message's authoritative `m.pose.up` in
  `_save_embeds`; gate `_score`'s lookup on `up_axis not in FORCED_UPS`.
  The one existing forced-axis test ran against an empty store — exactly
  the case that hides this.
- **E-R1-2** (MAJOR) — `flush` dropped today's nested-finally chaining
  (`classify_stls.py:1249-1268`, the full-disk incident): a pose-cache
  `os.replace` failure propagates before the CSV is written, and on the
  abort path takes `settle` + the second flush + `tasks.close` with it.
  Fix: `try: pose cache / finally: CSV`, last failure re-raises after
  every write has had its try.
- **E-R1-3** (minor) — `pose-cache.json.tmp` left behind when
  `os.replace` fails; `finally: tmp.unlink(missing_ok=True)`.
- **E-R1-4** (minor) — interfaces.md §Done `__init__` behind the code
  (categories + front/back banks are spec-sanctioned in substance).
  **Applied to interfaces.md same day.**
- **E-R1-5** (minor, deferred) — `done → cache_checker` import for
  `cache_key_from_identity` points the terminal stage at the admission
  module; the keying helper's natural home is the `identity` leaf. Later
  wave.

**Delta round E-R2**: CLEAN-WITH-NOTES, loop closed. E-R1-1's fix verified
structurally: `up_str(Pose.from_cache(d).up) == embed_cache_token(d,
"auto")` for anything `float()` accepts (9 entry shapes through a real
json round-trip, 0 mismatches), legacy from-disk entries DO reach
`_save_embeds` via the warm-pose/cold-`.npy` arm and agree end-to-end
against the real `route()`; the `_score` gate and flush guard agree by
construction (`not in FORCED_UPS` ≡ `== "auto"`); forced runs touch the
store on no path. E-R1-2's `finally` chaining matches the source's
`__context__` semantics. **E-R2-1** (minor, applied): the E-R1-4 doc fix
had named `front_T`/`back_T` where the code takes `front_embeds`/
`back_embeds`; ruling — keep the code's names (they are `pose.py`'s
`front_view_index` parameter names; `front_T`/`back_T` is the Embedder's
attribute), fix the doc.

## Wave 2 — driver + CLI (`src/driver.py`, `classify_stls.py`): FINDINGS → fix round

First end-to-end runs of the new pipeline passed (cold/warm/redraw/
skip-embed/SIGINT; warm CSV byte-identical; child torch-free by /proc
maps). Review re-verified every contract point (drain table, Resolved
re-route verbatim, admission window with no blocking send, wrap adapter
semantically identical to today's `timed` closure on both the timer and
the in-flight gauge, abort order + narration, SIGINT restore on every
exit path, retired flags gone) and the module-scope-torch rule
mechanically (0 torch modules after import, incl. under `__mp_main__`
re-exec). Name sweep independent: 42 names, 0 missing. All five listed
behaviour changes ruled acceptable (the ollama fallback's removal is
required by the hard constraint; measured cost ~1 model in 44).

- **W2-R1-1** (MAJOR) — the stall clock never resets: `child_owed()`
  silences the check during an arbiter tail (N1) but `last_progress`
  keeps aging, so the un-parking fold's own drain call sees
  `outstanding() ∧ child_owed() ∧ stale clock` and kills a healthy child
  zero ms after sending its task (reproduced on the test rig; 260 s park
  vs gemini's 300 s transport deadline). **The spec carried the same
  bug** — the drain pseudocode is amended with the fix: reset
  `last_progress` whenever `child_owed()` is empty; a wedge keeps
  `owed()` non-empty so O2's protection is intact.
- **W2-R1-2** (minor) — the poll-dispatch guard converts to `Failure`
  but omits `poser.drop`, contradicting the spec note added the same
  day; harmless today (poll pops parked first) but the guarantee should
  be real. Fix: add the drop; arms become literally identical.

Notes, no action: the old `stage("arbiter-wait")` has no successor
(driver's blocking wait is `results-wait` now); M4's "needs nothing
further from the child" is one drain-cycle optimistic in the
skip-embed+save-renders+dead-child+parked combination (self-correcting);
the pre-walk "N geometry-only poses will be re-resolved" diagnostic is
gone with the prefetcher (recorded in the migration notes).

**Delta round W2-R2**: CLEAN, no findings. The stall-clock fix verified
beyond report: the new regression test fails against a driver with the
reset stubbed out, and `test_a_wedged_child_is_killed_after_stall_s`
passes against *both* the fixed and unfixed driver — so O2/M3's wedge
path is genuinely unchanged rather than propped up by the new code.
Measured semantics: the un-parked file now gets exactly `STALL_S` of
fresh silence (kill at un-park + STALL_S against a child that never
answers), where pre-fix it was un-park + 0 ms. The bind-once deviation
(`owed = child_owed()`) accepted as an improvement, not merely
permitted: none of its three inputs (`admitted_files`, `retired_ids`,
`parked`) can mutate between the bind and the stall predicate — all
main-thread-only, and arbiter workers complete `Future`s without
touching `parked` — so the single bind makes the reset and the stall
half provably the same decision. `DriverConfig`'s `TYPE_CHECKING`
typing ruled correct and complete (no runtime use of the five names;
`import src.driver` still loads zero torch; both directions of the
`done ↔ driver` annotation pair stay annotation-only, so no runtime
cycle). Notes: `typing.get_type_hints(DriverConfig)` would now raise
`NameError` — matters only if a runtime-validating dataclass layer
(pydantic/typeguard) is ever added; a three-method `Protocol` for
`child` would state the fake-substitution contract more precisely than
`BaseProcess` + comment (taste, not correctness).

## Final whole-branch review (2026-08-18, fresh eyes): APPROVE WITH REQUIRED FIXES

A reviewer that saw no individual wave walked the branch diff
(`main..actor-refactor`) end-to-end and, crucially, *broke things to see
what the tests caught*. Cold/warm/redraw/forced-axis/skip-embed/SIGINT
all execute correctly; no message-type mismatch at any hand-off, no arm
that drops a file, no index retiring twice or not at all. E-R1-1's token
agreement confirmed end-to-end (forced `--up-axis z` re-run added zero
`.npy`), and the `Resolved → route` loop proven terminating
(`combine_up` always writes a margin, so the second call is always
sufficient).

**Fixed same day**: F-1 (MAJOR) the GPU suite had been failing since the
dedup pass — its parity test asserted a `classify_stls.PROMPT_TEMPLATES`
the dedup removed, so the branch's only bitwise embedding guard was
itself dead (parity intact; the guard was not). F-2 (MAJOR)
`eval/tile_and_vlm.py` still imported the top-level `pose` wave 0 moved
— the one file its 22-script sweep missed.

**Open, all minor**: F-3 invariant 5 (never block on a send / never a
blocking recv holding dispatchable work) has no test behind it —
`maxsize=1` on `tasks` and `drain(block=False)→True` both pass all 395,
because `FakeResults.recv is recv_nowait`; F-4 dead
`classify_stls.save_renders` (module-function vs method, which the dedup
AST sweep could not see); F-5/F-6 docstrings claiming no-copies and
not-yet-rewired; F-7 `--instrument` lost every child-side stage (the
child never calls `instrument.enable()`), so the flag's help and the
proposal's stage table promise attribution the code no longer produces;
F-8 47 `classify_stls.py:NNNN` citations are stale (40 past EOF, 7
landing on unrelated code — the dangerous kind), all correct against
`main:classify_stls.py`; F-9/F-10/F-11 design notes, OPEN_QUESTIONS and
`--pose-vlm-model` help contradicted by the shipped code; F-12 latent
`args.up_conf` read in `resolve_up`, reachable only from a caller that
supplies its own namespace.

**Invariant audit** — the value of the exercise was finding which
invariants are prose rather than pinned. Strongly pinned: 1 (retire
exactly once, though the headline test survived all four breaks — the
real coverage is in the boundary tests), 2, 3a, 4, 5c. **Not pinned**:
3b (Poser writes the store only via `record_pose` — true by
construction, untestable from outside), 3c (a *consistent* second writer
to `Admission.retired` passes), 3d (a genuine second writer to
`Done.rows` passes), 5a/5b (F-3).

**Riding notes resolved**: E-R1-5 verified closed by the dedup pass;
D-R1-3 closed with F-1's fix; D-R1-4 closed (the docstring exists);
C-R2-3 still correctly deferred (`min_interval` has no CLI surface, so
the paced branch is unreachable in production); M4's optimism confirmed
self-correcting by execution; the `pose_changed` SigLIP-confirmation
quadrant still unpinned but right by construction.

**Behavioural note, not a finding**: after `fail_outstanding`, a parked
file's fold can re-route to a `CachedHit` whose row overwrites the
`Failure` already written for it — the reverse of K5, and arguably an
improvement (the row is a real score from a real cached embedding).
Recorded so nobody rediscovers it as a bug.
