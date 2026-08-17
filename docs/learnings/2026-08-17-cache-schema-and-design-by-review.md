## The cache learned its schema, and the design was reviewed into shape (2026-08-14 → 17)

Two arcs from the actor-refactor design session that the 08-14 write-ups
(ipc-transport, precision-and-compile) did not cover: the cache-key migration
that shipped mid-review, and what fourteen adversarial review passes over two
design notes taught about designing protocols on paper. The full finding
trail is `docs/reviews/2026-08-14-data-structures.md` (six passes) and
`docs/reviews/2026-08-14-interfaces.md` (eight).

### The cache schema migration (shipped, both live caches)

* **A moved key scheme fails silently — the stamp turns it into one line.**
  Every lookup just misses and the run re-renders and re-embeds the
  collection: hours, and money for VLM poses. `cache-meta.json` +
  `CACHE_VERSION` makes readers refuse instead. Hand-set integer, never a
  format hash: this repo deliberately makes byte-compatible key changes
  (`elev`, `|compiled`, `|evN` all append only when non-default) and an
  auto-derived stamp would punish exactly that discipline.
* **Key on what changes the pixels.** The embed token's source-based elision
  ("deterministic poses keep the `--up-axis` string") existed to keep an old
  cache valid and quietly filed identical pixels under two keys. The honest
  token — `up_str(pose.up)`, because only `up` moves the render — collapsed
  the duplication and took `source` out of the key entirely, which is what
  made the `heuristic→geometry` / `ensemble→siglip` rename a rename instead
  of a 1531-model re-embed.
* **Re-keying is metadata, not compute.** 2799 + 1148 + 144 `.npy` renamed,
  ~1.1 GB never read. The 144: the first migration iterated the *walk*, and
  144 pose entries had no walked file. The fix made the review's own
  verification trick load-bearing — `file_identity` is byte-identical to
  `cache_key`'s first three fields, so `cache_key_from_identity` re-keys
  straight from `pose-cache.json` with zero `stat()` calls, and a
  half-mounted collection cannot leave entries behind. The stamped door was
  reopened by making the tool probe regardless of the stamp.
* **Rename semantics: map on load, never bump.** `load_pose_cache` maps old
  source spellings in place; a `POSE_CACHE_VERSION` bump would have
  re-resolved (and re-billed) unchanged poses for a spelling. The trap the
  reviewer named as the likeliest silent cache-killer: the migration must
  read `pose-cache.json` as **raw JSON**, because reading through
  `load_pose_cache` would hand it new-spelling sources, `old_embed_cache_token`
  would miss every override, and 1531 embeddings would quietly orphan.

### What fourteen review passes taught about designing on paper

* **Protocols must be walked, not read.** The interfaces note's seams
  survived every pass; its *termination* did not — three of four run modes
  hung (sentinel before the arbiter tail, both queues bounded closing the
  render cycle, `--skip-embed` retiring through nothing), all three
  violating the note's own Invariant 1, all three invisible to reading and
  obvious to walking the four flag modes. The reviewer's method — trace each
  signature to the code path it replaces, walk cold/warm/redraw/skip — found
  every real defect; prose review found none of them.
* **Carve-outs generate findings until removed.** `needs_embed=False` sends
  no result" produced four findings across two passes (a hang, a
  double-retire, an unbounded backlog, a stale doc claim) and was closed by
  one uniform rule: every task gets exactly one result, `Rendered` is the
  ack. The dissolved-R7 claim generated work in *three* consecutive passes
  until the sentence was deleted rather than corrected. Special cases in a
  protocol are compound interest in reverse.
* **Invariants need mechanisms, not conventions.** "Retires exactly once"
  became true when `Done` got `retired_ids` and repeats became no-ops —
  before that, a double retirement drove `in_flight()` negative and turned
  the hang inside out: a run that finishes early with a complete-looking
  CSV. Same lesson at the exits: liveness checks belong in `drain`, the one
  place both blocking loops share, not in whichever caller remembered; a
  wedged child is bounded by a no-progress deadline, not assumed away.
* **The clean path is where the constraint bites.** The child's *successful*
  exit was its dangerous one — returning from `run_child` tears down the
  `OffscreenRenderer`, the repo's one hard-abort. `os._exit(0)` after stdio
  flush is "never destroy a renderer" expressed as code, and it is safe only
  because the sentinel follows quiescence — two fixes load-bearing for each
  other, now stated so neither is undone alone.
* **Writing calling conventions audits the message set.** The interfaces
  phase immediately found `PoseTiles` missing `geo_scores` (the mesh never
  crosses, its geometry evidence must) — a gap six data-structures passes
  had not surfaced, because it only exists when you ask *who computes what
  where*. Design layers find each other's holes; sequence them.
* **Severity falling is the convergence signal — counts are not** (amended
  twice: pass 7's O6 caught the claim contradicting its own data, pass 8's
  P5 fixed the arithmetic in the amendment). Neither sequence is
  monotonic: data structures 15 → 7 → 5 → 2 → 4 → 1-plus-good-news,
  interfaces 16 → 8 → 7 → 4 → 5 → 7 → 6 → 5. A stop-on-first-rise count
  test fires after pass 4 — right *after* the liveness hang it did catch,
  and before the wedge bound (pass 5), the stall clock's arithmetic (pass
  6), and the scoping bug that made both inert (pass 7) were ever
  written. The count test does not fail by missing the obvious hang; it
  fails by stopping while three passes of consequences are still
  unwritten. What fell monotonically is what the findings were *about*:
  protocol hangs (I/J), then failure-path holes (K–M), then one
  mechanism's arithmetic (N), then scoping and citations (O), then prose
  against a settled protocol (P) — at which point the reviewer called the
  stop, and the rule held.

### Conventions that made the loop work

Record each pass as received in its own commit, then take it with finding
IDs in the message. Answer a reviewer's gating questions with grounds, in
the note, so the decision survives the conversation. When the reviewer
states a lean (bound the wedge, drive from the pose cache), taking it is
usually right — they have walked the failure; when they offer either-or
(T1's reword-vs-compute), decide on what exists in this tree and say so.
Verify a reviewer's numbers before folding them in: they reproduced every
time until pass 5's "34 ms–2 s of child work" did not (the real range is
3–28 s, `actors_proposal.md`), and checking is what kept a stall deadline
from shipping at ~2× the work it was bounding — the one failure is the
best evidence for the convention. And the
asides matter: three of the most-cited harnesses had no README rows until a
reviewer had to find them the hard way.
