## The default that only became load-bearing when the semantics changed (2026-08-27)

`rank()` had two bounds that could not both apply: `min_score` **replaced** the
top-N cut. model-browser wanted them to compose — "the best N of everything at
least this similar" — which is a four-line change to one `if`. The four lines
were the easy part. What made the change interesting is that a default nobody
had thought about for two months became load-bearing the moment the semantics
under it changed, and that two of the tests written to guard the new behaviour
could not fail for the reasons they were named.

### A default is safe only under the semantics it was written for

`QueryRequest.top` was `int = Field(10, ge=1, le=1000)` and had been harmless
since the API shipped. It was harmless *because* `min_score` discarded it: a
floor-only caller — which is every request model-browser's default view sends,
since it omits `top` entirely — arrived with `req.top = 10` and the replace
branch threw it away.

Compose the two and that same request slices the floor set to ten rows.
Measured on embed-cache512 (3380 models, `/run/media/masa/STLLibrary`),
`fantasy character` templated, pool softmax, floor 0.1:

| request | rows |
|---|---|
| floor 0.1, no count | **875** |
| floor 0.1, `top=10` (the old default) | **10** |

The floor set runs 0.1463 down to 0.1000; the tenth score is 0.1426, so the
cut lands in the top 1.4% of a set the caller asked to be exhaustive. Nothing
would have errored. `eval/compose_bounds.py --floor-set` is the re-runnable
form.

**The default lived in four places, and only one of them was the schema.**
`rank()` carried `top=10` in its own signature, which no schema edit reaches;
`test_categories.show_query` carried another, which is the REPL — where
querying in this repo actually happens; and `docs/api/surface.md` stated the
rule in the table both repos review against. A change that retired only the
HTTP default would have left `:min 0.1` answering ten rows with no request
involved.

The REPL's ten survived, and that is the distinction worth keeping: it is a
*display* default, not a bound anyone chose, so `show_query` sends it away
when a floor is in force and `:min` still reads as "threshold instead of
top-10". The ten was only ever dangerous as a bound.

### The composition order is invisible in the rows

The obvious way to pin floor-then-count is to assert the rows come back
floor-filtered and count-capped. That assertion cannot fail. The floor tests
the same key the sort ordered, so the set above the floor is always a
*prefix* of the descending order, and

    order[mask][:t]   ==   order[:t][mask]

for every input, ties included — 0 divergences over 20000 tie-forced fuzzed
cases (`eval/compose_bounds.py --orders`). Slice first and you get the same
rows back; the count and the floor commute on the result set.

What does not commute is what can be *reported*. `matched` — how many models
cleared the floor — only means that if the floor was applied to the whole
collection first; compose the other way and the number is bounded by the
count, so "showing 60 of 875" becomes unsayable. The order is load-bearing
for the reportable number and for nothing else. (Under this script's
parameters the two orders disagree on `matched` in 11560 of 20000 cases; a
model-browser session fuzzing the same claim got 8926 under its own. The zero
is parameter-free, the other number is not — quote it only with its script.)

This one had teeth downstream: model-browser's design had rejected the reverse
order for "returning fewer than N for no stated reason", which cannot happen,
and its contract test was described as the test that fails if the index
composes the other way — asserting rows, it would not have. Both corrected
upstream, and the argument replaced with the `matched` one, which folds two of
its decisions into one.

### Both guards were caught by mutation, not by reading

Neither finding is visible in a diff. Both came from a reviewer copying the
tree, breaking the source on purpose, and watching what stayed green:

- Reverting `show_query` to compose with its display ten — the exact
  regression, in the tool that matters most — passed the **entire suite**. The
  parity oracle could not have caught it: `repl_rank` in the tests is a *copy*
  of `show_query`'s call, so the parity tests confirm the copy while the
  original drifts, and the only fuzz arm ran `min_score=None`. A test that
  mirrors production by copying tests the mirror.
- The composition test asserted rows, so slicing-first passed it. The
  behaviour was pinned — by the `matched` tests — but not by the test named
  for it, and its comment asserted a divergence that cannot occur.

The general form, which is the reason this entry exists: **a test whose
subject is a translation between two layers must be run against the
translation, not against a transcription of it.** Both failures here are that
sentence — one copied a call into the test file, the other copied an intuition
into a comment.

Reviewed in `docs/reviews/2026-08-27-compose-floor-and-count.md`; shipped as
`add7fd4` and `4b28af3`.
