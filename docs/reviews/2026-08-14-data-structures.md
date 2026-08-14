# Review of `docs/actor-refactor/data_structures.md`

Review note, 2026-08-14. Covers the design note as written on 2026-08-14
(untracked at review time), read against `classify_stls.py`, `pose.py`, the two
pose caches on disk, and the three learnings write-ups it cites.

Method: every numeric claim was recomputed from the cited write-up or from the
data; every behavioural claim was checked against source, with `file:line` given
so the implementor can re-read rather than re-derive. Design decisions were
**not** relitigated — this checks whether the note describes the code and itself
correctly, not whether the shapes are the right shapes.

Findings carry IDs (`D1`…`D15`). Cite them in commits and in the note's own
revisions.

## Verdict

**The measurements are sound; the code claims and the edge coverage are not.**
The transport table, the roundtrip numbers, and the payload arithmetic all
reproduce exactly (§5). What fails is the layer above: four claims about current
code are wrong or stale (§1), the "frozen dataclass per edge" type set is missing
at least four edges and mis-shapes two more (§2), and the note specifies threaded
machinery for a v1 it opens by calling sequential (§4).

One finding changes a shape rather than a sentence: **`PoseTiles.tiles` as a flat
list cannot feed the ensemble** (D7). One changes a decision's stated
justification without changing the decision: **the mesh-residency argument rests
on a parse cost when the upload is what it was built for** (D13).

Two open questions gate three of the fixes (§6).

## 1. Wrong against the code — fix before handing the note to anyone else

### D1. `Pose.source` lists a value that does not exist — HIGH

The note gives `source: str  # "forced" | "geometry" | "vlm" | "ensemble"`.
`"geometry"` is never written. `resolve_up` initialises `source = "heuristic"`
(`classify_stls.py:466`) and only moves it to `"ensemble"` or `"vlm"`; the forced
path writes `"forced"` (`classify_stls.py:999`). Both caches on disk agree:

| cache | entries | `heuristic` | `ensemble` |
|---|---|---|---|
| `embed-cache2/pose-cache.json` | 2943 | 1949 | 994 |
| `embed-cache3/pose-cache.json` | 1149 | 612 | 537 |

This is not a typo with no consequence. `from_cache` is proposed as *the* place
old shapes are absorbed, i.e. the validating gate; written against this literal
set it rejects every entry in both caches. Two live consumers key off the real
strings — `embed_cache_token` on `source in ("vlm", "ensemble")` (`pose.py:170`)
and `pose_is_sufficient` on `entry["source"] == "vlm"` (`pose.py:154`).

Fix: `"forced" | "heuristic" | "ensemble" | "vlm"`.

### D2. Two of the "three atomicity fixes" have already shipped — HIGH

§Unchanged says the refactor adds "temp + `os.replace` for the pose cache and
`.npy` writes, partial CSV flush on Ctrl-C". As of this review:

* **CSV flush on abort: already done.** The write sits inside `finally`, in a
  nested chain that attempts all three artifacts (`classify_stls.py:1134-1169`).
* **`.npy` torn write: already handled**, differently but for the stated failure
  mode — `np.save` unlinks the file on `BaseException` so a truncated `.npy`
  cannot pass the `.exists()` check next run (`classify_stls.py:1086-1092`).
  Temp + `os.replace` would still be stronger against SIGKILL; say that, rather
  than describing an open hole.
* **`save_pose_cache` is still a bare `write_text`** (`pose.py:133-138`). This
  one is real, and it is the artifact whose loss costs money.

The staleness is inherited: `actors_proposal.md` §Shutdown still asserts "the
write sits *after* the `finally`". Correct both notes in the same pass, or the
next reader re-finds this.

### D3. The `Pose` motivation cites the wrong lines — MEDIUM

The argument that "decided it" makes three claims; two mis-cite.

* "the bare-int `front_view` fallback is inlined at its read site
  (`classify_stls.py:1101`)" — the read-site fallback already has exactly one
  home, `pose.front_view()` (`pose.py:196`). `classify_stls.py:1100-1103` is the
  *write* site, merging a legacy value into the per-config dict.
* "`pose_is_sufficient` inspects `v` and `source` on a raw dict" — it inspects
  `source` and `margin` (`pose.py:141-156`). `v` is filtered in
  `load_pose_cache` (`pose.py:126`), which is where version handling belongs and
  already lives.
* "every reader does `.get("margin")` defensively" — correct.

The conclusion survives on independent evidence: `embed-cache3` really does hold
bare-int `front_view: 0` entries alongside per-config dicts. Keep the
conclusion, re-cite the argument.

### D4. The transport payload is not the production payload — MEDIUM

The table's header says "24 tiles + 16 views uint8 at 384 px, 17.7 MB/model".
`UP_TILE_AZIMUTHS = 2` is already in the code (`classify_stls.py:190`), so a
model is 12 tiles + 16 views ≈ 12.4 MB — 30% less. The arithmetic and every
throughput figure are correct *for what the spike blasted*; the framing as the
current payload is what is wrong.

This only strengthens the `mp.Queue` decision. Label the row as the spike's
payload and note the production figure beside it.

## 2. Edges the type set does not cover

The note's own claim is "frozen dataclasses per edge, no `kind` field". Held to
that standard:

### D5. Four edges have no message type — HIGH

Missing entirely: **Cache Checker → `Done`** (the cached-embedding hit, which
`actors_proposal.md` calls the whole point of that cache), **Embedder → `Done`**
(embeddings plus whatever `Done` scores against), **Poser → Embedder** (the
ensemble's tile-embed request), and its answering back-edge. §Queues names
`Embedder → Poser` as one of the three unbounded back-edges while defining no
message that travels on it.

### D6. `Arbiter → Poser` is specified twice, incompatibly — MEDIUM

§Queues calls it an unbounded back-edge queue. §Poser continuation state gives
`ParkedFile.future: Future`. Both cannot be the transport. The `Future` matches
today's `ThreadPoolExecutor` (`classify_stls.py:976, 1032`) and is the cheaper
build — but if it wins, the Arbiter is not a queue-fed actor and the back-edge
rule does not apply to it. Gated on Q1 (§6).

### D7. `PoseTiles.tiles` as a flat list loses shape the Poser needs — HIGH

The ensemble is
`np.asarray(score_upright(flat)).reshape(len(grid), -1).mean(axis=1)`
(`classify_stls.py:475`) — it needs the `[6][n_az]` grouping, not 6·n_az images
in a row. A flat `list[np.ndarray]` can only be regrouped if the parent knows
`n_az`, and `n_az` is exactly the parameter that just moved 4 → 2 (D4). Encode
it: `tiles: list[list[np.ndarray]]`, or keep the flat list and carry `n_az: int`.

Second half of the same edge: the arbiter path needs the six first-column tiles
as PIL Images for `make_contact_sheet` (`pose.py:300-317`), and today gets them
from the same grid (`classify_stls.py:472`, and `render_up_candidate_tiles`
at `classify_stls.py:283` for the `--no-up-ensemble` path). Arrays are the right thing to ship; say who converts
back.

### D8. No message expresses "render needed, embedding cached" — MEDIUM

That is the `redrawn` path: `--save-renders` set, an image missing from the
render index or the pose just changed, but the `.npy` still live
(`classify_stls.py:1054, 1074`). `actors_proposal.md`'s Cache Checker routes it
to the render side, and the only return type is `EmbedViews → Embedder` — so it
re-embeds what is already on disk, at the cost of the run's single most
expensive stage. Either `EmbedRenderTask` carries `needs_embed: bool`, or the
child saves the renders and returns nothing for this case. Gated on Q2 (§6).

### D9. The reason `Pose` is unfrozen does not survive the boundary — HIGH

`Pose` is left mutable so `Done` can write `front_view` after scoring, "on the
hottest write path". But `EmbedViews.pose` has been pickled into the child and
pickled back — it is a copy. Mutating it updates nothing that reaches
`save_pose_cache`.

This lands precisely on the comment `# echoed through so Done needn't look it
up`: `Done` **must** look up the canonical entry by identity to write
`front_view`, whatever the echo is for. Either state that the echo is
read-only convenience and `Done` writes through the pose dict, or freeze `Pose`
and have `Done` replace the entry — the copy is being made by `mp.Queue` either
way, so freezing costs nothing it does not already cost.

### D10. `v: int = POSE_CACHE_VERSION` as a field default launders versions — MEDIUM

`from_cache` is specified as the place legacy shapes are absorbed. A dict that
reaches it without a `v` gets stamped with the current version, and `to_cache`
writes it back as freshly resolved — silently defeating `load_pose_cache`'s drop
rule (`pose.py:126-130`), whose whole purpose is that a pose decided under a
different ensemble is not this version's pose.

It is safe today only because `load_pose_cache` filters before `from_cache`
would ever see an entry. Do not leave a shape whose correctness depends on call
order: have `from_cache` carry `v` through from the dict, or drop the default.

### D11. `pose_is_sufficient` as a method loses its `None` case — LOW

It is called with a possibly-absent entry at `classify_stls.py:964` and
`classify_stls.py:1003`, and `entry is None → False` is load-bearing (it is the
miss test). A method on `Pose` has no receiver for a miss. Say explicitly that
absence is the Cache Checker's dict lookup, not `Pose`'s job.

### D12. `PoseTask`/`EmbedTask` and `PoseRenderTask`/`EmbedRenderTask` are identical pairs — LOW

Field-for-field the same. The note also says the Loader "is not an inter-actor
edge at all", which leaves Cache Checker → Renderer as the only consumer of the
first pair — so one pair is dead, or something translates and the note does not
say what. Name the translator or collapse them.

## 3. Renderer-child mesh residency

### D13. "A miss is cheap now anyway" attributes the cost to the wrong half — MEDIUM

The parse got cheap; the upload did not. `_upload` is measured at 275 ms on an
800k-triangle STL (`classify_stls.py:259-260`) against 34 ms to re-show a hidden
geometry — a miss is ~8× a hit, and the upload is the entire reason a device
tier was proposed. "~10 ms parse plus re-upload" reads as though the 10 ms were
the whole miss.

The conclusion is likely still right — the only revisit in a run is the pose →
embed round trip, and the roundtrip spike resolved residency to a three-entry
dict at 88% busy — but it needs to be reached through the upload number, not
around it.

### D14. `ResidentMesh` cannot express its own eviction rule — MEDIUM

Three gaps in five lines:

* **"never evict in-flight"** needs an in-flight marker. `ResidentMesh` has
  `name`, `center`, `radius`, `nbytes` and no way to say a mesh is spoken for.
* **FIFO is not the LRU the proposal specified.** `OrderedDict` without
  `move_to_end` evicts in insertion order, which with a round trip in flight can
  drop exactly the mesh about to come back. If FIFO is the deliberate
  simplification, say so against the proposal's LRU; if not, it is a bug in the
  shape.
* **`budget_bytes` stops being a bound** once in-flight entries are exempt:
  three heavy meshes at the spike's `--inflight 3` is ~450 MB whatever the
  budget says. `actors_proposal.md` tied residency depth to the Supervisor's
  admission window for exactly this reason; the link is dropped here and worth
  restoring in one sentence.

Minor: `name` duplicates the dict key.

## 4. Coherence

### D15. v1 is called sequential, then specified with threads — MEDIUM

The preamble settles on "a **sequential driver**, and exactly one process
boundary at the Renderer". §Supervisor accounting then specifies
`threading.Condition`; §Queues specifies bounded forward edges, unbounded
back-edges, and a deadlock rule. A sequential driver has none of those.

Most of the note *is* v1-real — `Pose`, the message types, `ResultRow`, `parked`
(the deferral is already threaded today via `ThreadPoolExecutor`), the child's
resident dict, the boundary queues. The rest is the threaded successor. Mark
which is which; the implementor otherwise builds a condition variable for a loop
that cannot contend.

Related and unresolved: the boundary is "bounded both ways at depth 4", the
Supervisor has an admission limit, and the roundtrip spike ran `--inflight 3`.
Three bounds on the same in-flight window, no stated relation between them.

Trivial: 2048 px is 28.4× the bytes of 384 px, not 29×.

## 5. What checked out — do not re-verify

Recomputed or re-read, all correct:

* **The transport table**, against `docs/learnings/2026-08-14-ipc-transport.md`
  and its own arithmetic: 17.7 MB ÷ 13.5 ms = 1.31 GB/s, ÷ 2.8 ms = 6.3 GB/s,
  ÷ 5.5 ms = 3.2 GB/s, ÷ 4.1 ms = 4.3 GB/s. Every row consistent.
* **The payload arithmetic**: 384² × 3 = 442 KB/view ("~440 KB"); 40 images ×
  442 KB = 17.7 MB. (See D4 for the tile count itself.)
* **384 px is production** — `run_classify.sh:3`. It is `actors_proposal.md`'s
  `--render-size 2048` that is stale, not this note.
* **16 views** = `--views 8 --elevations 20,-20` through `total_views`
  (`classify_stls.py:618`).
* **~0.85 s of the 6–8 s parent wait, ~0.5% of a cold run** — matches the
  ipc-transport write-up's decomposition, including that the estimate it
  replaces was 5%.
* **Three resident meshes, 88% busy, 1.11× against overlap's 1.21×** — matches
  `docs/learnings/2026-08-13-roundtrip-tiles-and-the-full-label-set.md`.
* **`Failure` is load-bearing, and today's error rows are malformed** —
  `rows.append({"file": ..., "top1": f"RENDER_ERROR: {e}"})` at
  `classify_stls.py:1024` and `1066`, surviving only because `DictWriter` fills
  missing keys from `restval`.
* **The legacy bare-int `front_view` is real on disk** — `embed-cache3` holds
  `front_view: 0` entries. (D3 corrects the citation, not the fact.)
* **`ParkedFile.resolved` is the right tuple** — `(up, ratio, source, margin)`,
  matching `deferred.append(...)` at `classify_stls.py:1032-1033` and
  `apply_arbiter` at `classify_stls.py:506`.
* **`front_view: dict[str, int]` keyed by view config** — matches
  `view_config()` (`classify_stls.py:300-308`) and the `'8v-e20,-20'` keys on
  disk.

## 6. Questions that gate fixes

**Q1. Is the Arbiter a queue-fed actor, or a `Future` the Poser holds?**
Decides D6, and part of D15 (a `Future` removes one of the three back-edges the
deadlock rule is written for).

**Q2. Does the render child save renders itself?** `actors_proposal.md`'s
Renderer says yes. If so, D8 is a `needs_embed` flag on `EmbedRenderTask` and
the child simply returns nothing on that path; if not, `Done` needs a
render-only message and a save step.

A third, smaller: **is `rows` the retirement record or the output record?**
Under `--skip-embed` a file retires with no row (`classify_stls.py:1094`), so
`rows: dict[int, ResultRow | Failure]` will have holes while
`admitted == retired` still holds. Correct as designed — worth one sentence so
nobody asserts `len(rows) == admitted`.

## Suggested order

1. **D1, D2, D3, D4** — pure corrections to the note, no design input needed.
   D2 also touches `actors_proposal.md` §Shutdown.
2. **D9, D10, D13, D14** — shapes and justifications that are wrong
   independently of Q1/Q2.
3. **D5, D7, D11, D12** — complete the type set. D7 is the one that would
   otherwise be found at implementation time.
4. **D6, D8** — after Q1 and Q2 are answered.
5. **D15** — last, since answering Q1 changes what is left to mark.
