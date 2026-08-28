# Compose floor and count (`add7fd4`) — 2026-08-27

Single-reviewer pass (fable-5) over the change that made `min_score` and
`top` compose in `rank()` and retired the ten-row default that composition
would otherwise have weaponised. Verdict: **implementation correct, ship as
is**; four findings, all in the guards and the prose rather than in the
semantics, all fixed in `4b28af3` the same session. `(review, 2026-08-27)`
citations in the code resolve here.

The theme is worth stating because both findings that mattered have the same
shape: **a test that cannot fail for the reason it is named.** One asserted a
value the change cannot alter, the other asserted nothing about the code it
claimed to cover. Neither would have been caught by reading the diff, and both
were caught by mutating the source and watching the suite stay green.

## Findings (fixed, most severe first)

- **C1 — the REPL's floor translation was guarded by nothing.**
  `show_query`'s `query.rank` call sends its display ten away when a
  floor is in force, which is the whole reason `:min 0.1` still means "every
  model at or above 0.1" now that the bounds compose. Mutating it back to
  `rank(sims_1d, top=top, min_score=min_score)` — the exact regression, in the
  tool where querying actually happens — passed the **entire** suite. The
  parity oracles cannot cover it by construction: `repl_rank`
  (`tests/test_query.py`) is a *copy* of that call, so they confirm the copy
  while the original drifts, and the fuzz arm only ever runs `min_score=None`.
  Failure scenario: someone simplifies `show_query` back to the obvious
  one-liner; `:min 0.1` silently answers 10 rows of 875; nothing goes red.
  **Fix:** `test_the_repl_sends_its_ten_away_when_a_floor_is_in_force` pins it
  against `show_query` itself, in a subprocess for the reason the torch-free
  rule uses one. Falsified against the mutation before being trusted (prints
  `10 models >= 0.5` where the guard demands 41).

- **C2 — the composition-order test could not see the composition order, and
  its comment was false.** `tests/test_query.py`. The test asserted rows, and
  rows cannot distinguish floor-then-count from count-then-floor: the floor
  tests the same key the sort ordered, so the floor set is always a *prefix*
  of the descending order, and `order[mask][:t]` and `order[:t][mask]` select
  the same rows for every input — 20000 fuzzed tie-heavy cases, zero
  divergences. The comment claimed they "differ whenever the count is smaller
  than the floor set", which never happens. The behaviour *was* pinned, by the
  five `matched` tests, but not by the test named for it. **Fix:** the
  assertion is `matched` (20 here, 4 if the count had cut first), with the
  tautology named as one so it is not re-strengthened into a claim again.

- **C3 — stale prose in the jointly-owned contract.** `docs/api/surface.md`,
  the `rel_path` join note: the consumer's miss stats were described as
  "bounded by `top`". The consumer's default is floor-only and never sends a
  count, so `cap` is the actual bound — imprecise before this change (the
  replace rule ignored `top` under a floor too) and plainly wrong after it.
  **Fix:** names both bounds and which one the consumer's default actually
  meets. The only live false claim the sweep found; every other surviving
  "top-10" in the repo is a dated measurement, REPL prose that stays true
  under the REPL's kept exclusivity, or already struck through.

- **C4 — the measurement's conditions were incomplete by this repo's own
  convention.** `src/query.py`'s `rank` docstring named the cache, the volume,
  the model count and the date, but not the **pool** — and 0.146 / 0.143 /
  0.100 are softmax's numbers. Separately, "875 rows with no count" is
  `rank`'s number: over HTTP the same request answers 500 with
  `truncated: true` until `cap` is raised, which the commit message's
  "production path" framing overreached. **Fix:** both stated. The script is
  now `eval/compose_bounds.py` so the figures are re-runnable rather than
  merely attributed.

## Verified correct

Recorded so a later reader knows these were checked rather than skipped.

- Semantics: filter → `matched = len(order)` → slice is exactly what the
  docstring, `surface.md` and the consumer's D9 describe. Boundaries behave:
  `top` above the floor set returns the set; a floor nothing clears gives
  `matched=0` with `best` kept; empty `sims` returns before `robust_z`;
  `cap < top` truncates after `rank` with `matched` untouched.
- `truncated` remains the cap's alone, and the empty-scope early return
  carries `matched: 0`.
- Tie-break and stability: the stable argsort runs before either cut and both
  cuts take prefixes, so the membership-at-the-boundary guarantee holds.
- No other callers or readers anywhere, `eval/` and the tools included: only
  `src/query.py` constructs `Ranked`, nothing unpacks it positionally,
  `/similar`'s explicit `k` is untouched, and no script assumes the `/query`
  shape.
- Mutation coverage elsewhere is real: rank's default back to 10 (2 killed),
  `Field(10)` back (2), `matched` after the slice (3), `matched` before the
  floor (3), the exclusive branch restored (4), `matched` dropped from the
  empty-scope body (1).
- The REPL decision itself — keeping the ten as a display default rather than
  retiring it — is right and consistently implemented; `--min-score` and
  `:min` share one variable feeding one `show_query` call.

## Sent upstream, not fixed here

C2 invalidated two claims in model-browser's `floor-and-count-compose`, since
its design reasoned about an order difference that does not exist:

- D2 rejected count-then-floor for "returning fewer than N for no stated
  reason" — it returns exactly what floor-then-count returns.
- Task 4.1's contract test was described as "the test that fails if upstream
  composes the other way". Asserting rows, it would not have.

The replacement argument, which is stronger: the order is load-bearing
*because of* `matched`. Compose the other way and the reportable number is
bounded by the count — 60, never 875 — so "showing 60 of 875" becomes
unsayable. That makes D2 and D9 one decision rather than two. Both applied
upstream (`model-browser@5d0cf27`), where the contract test now asserts
`matched > top` against a live index and skips rather than passes when the
index is absent. That session also re-ran the fuzz independently: 0 row
divergences over 20000 cases, and the two orders disagreeing on `matched` in
**8926** of them.
