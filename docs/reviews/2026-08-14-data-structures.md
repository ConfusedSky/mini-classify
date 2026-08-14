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

---

# Pass 2 — 2026-08-14, against `76be854`

Six commits since pass 1: `898635e` took D1–D15, `bfa63c6` and `76be854`
revised the shapes further, and `3717534`/`17d0d5a`/`b563fd7` are the
precision arc and the `--compile` flag. Suite green at review time: **180
passed, 1 skipped**.

Same method as pass 1 — claims re-derived from source and from the caches on
disk, not read. New findings carry `R` IDs. This pass also records two
**decisions** taken during the review conversation (§P2.3) and the migration
they imply (§P2.5), because both outran what a findings list can hold.

## P2.0 Verdict

**Pass 1's findings all landed; the follow-on work opened one new hole in
shipped code and one in the design.** The doc's revision is materially better
— D7's grid, D9's frozen `Pose`, D10's version handling and D14's LRU are all
correct now, and the two gating questions are answered inline.

The new code hole is `R1`: `--compile` was correctly made part of the
embedding cache's identity and was **not** made part of the pose cache's,
which consumes the same compiled tower.

The design hole is `R2`, and chasing it produced the more valuable result.
Asking why `"geometry"` read badly exposed that nobody had written down what
`source` actually means — it records *what moved the answer*, not *what ran* —
and that `embed_cache_token` keys on it only as a **proxy for determinism**.
Both are now decided (§P2.3) and cost a migration, not a re-embed (§P2.5).

## P2.1 Disposition of pass 1

| finding | status |
|---|---|
| D1 | **reversed deliberately** — see `R2`, then superseded by decision `P2.3-A` |
| D2 | taken; `actors_proposal.md` §Shutdown corrected in the same commit |
| D3 | taken — citations moved to `pose.py:196` / `pose.py:126` |
| D4 | taken — table relabelled as the spike's payload, production noted |
| D5 | taken — `EmbedTilesRequest`/`TileEmbeds`/`CachedHit`/`Embedded` added |
| D6 | answered (Q1): the `Future` wins, the back-edge list is corrected |
| D7 | taken — `tiles: list[list[np.ndarray]]`, `[candidate][azimuth]` |
| D8 | answered (Q2) — `needs_embed: bool`, child owns saving. See `R7` |
| D9 | taken — `Pose` frozen, `Done` writes through the canonical dict |
| D10 | taken — `v` carries through, no field default |
| D11 | taken — stays a module function over `Pose \| None` |
| D12 | taken — pair collapsed, and the Loader *module* seam kept deliberately |
| D13 | taken — the 275 ms upload leads the argument, parse is the aside |
| D14 | taken — `move_to_end` LRU, `in_flight` exemption, soft-bound stated |
| D15 | taken — explicit v1 / threaded-successor split at the top |

## P2.2 New findings

### R1. `--compile` re-keys the embedding cache but not the pose cache — HIGH

`b563fd7` makes the numeric regime part of `cache_key` on exactly the right
argument: a permanent cache must not mix regimes. The pose cache is also
permanent, and it also consumes the compiled tower.

`main()` replaces the bound method (`classify_stls.py:887`), and
`score_upright` reaches it through `embed_images` →
`model.get_image_features` (`classify_stls.py:963-965`). So under `--compile`
the pose ensemble's tile embeddings carry the drift — median 7.3e-04, max
3.1e-03 by the write-up's own table — into `upright_scores` → `combine_up`,
which decides both the argmax and the margin. `pose.file_identity` is
`rel_path|mtime|size` (`pose.py:110`): no regime token, no version bump. Two
runs at different regimes share pose entries, and whichever resolves a file
first pins it.

The exposure is not a wrong category, it is the escalation gate: a combined
margin crossing `MARGIN_THRESHOLD = 0.45` (`pose.py:30`) changes whether a
*paid* Gemini call happens, on the artifact the code singles out as the only
one whose loss costs money (`classify_stls.py:1166-1169`). And
`eval/compile_flips.py` measured top-1 **category** flips — the pose-side rate
is unmeasured, so "1 of 341" does not cover this path.

Accepting it is defensible: poses are resolved only on cold and upgrade files,
and a mixed cache is inside the noise `parser_gate` concluded is permanent.
But `docs/learnings/2026-08-14-precision-and-compile.md` covers `.npy`
identity and text embeddings and is silent here, so it currently reads as an
omission rather than a decision. Fold it into the entry proposed in
[M3](#m3-close-the-pose--embedding-key-gaps).

### R2. The `"geometry"` rename picked a name a measured write-up had rejected — HIGH

Taking D1 by changing the code rather than the literal is legitimate, and
`from_cache` mapping the old value is a *better* answer than the migration
script the open question assumed. The target was wrong.

`OPEN_QUESTIONS.md:118-120` and
`docs/learnings/2026-08-12-where-a-7-hour-run-went.md:869-875` had already
decided this rename with a measurement behind it: `"heuristic"` means *the
ensemble ran and agreed* (verified by re-resolving 15 `heuristic`-marked
models — **0 of 15 moved**), and the name "reads as 'the ensemble was skipped'
— it actively misled during a previous session".

`data_structures.md:238-251` justified `"geometry"` as "it reads better",
citing neither. Superseded by `P2.3-A`, which resolves it differently than
either the doc or the open question proposed.

### R3. The rename's blast-radius list is incomplete — MEDIUM

`data_structures.md:247-250` names the write site, two test files and the CSV
column. Also affected:

* `cleanup.sh:36` and `:106` — usage text and the deletion message. Worth
  naming **because** it is the destructive path, and worth stating that its
  filter keys on `"vlm"` (`cleanup.sh:104`), so the rename cannot break it.
* `classify_stls.py:926` — "ambiguous poses keep the heuristic guess".
* `classify_stls.py:451` — the `resolve_up` docstring.
* `eval/siglip_up.py:142,148` — prints `cached['source'][:4]`; that column
  silently becomes `geom`.

Historical specs under `docs/superpowers/` also quote the old value; leave
them, they are records.

### R4. `Pose` frozen with a `dict` field — LOW

`hash(pose)` raises `TypeError` (a frozen dataclass generates `__hash__` from
its fields; `front_view` is a dict), and `pose.front_view[cfg] = i` still
mutates through the freeze. Only matters if anything ever keys on a `Pose` —
one line in the note.

### R5. `Done` does not stay on the device either — LOW

`data_structures.md:161-170` is accurate that the **Poser** does the one
conversion, but `Done` needs numpy regardless: `front_view_index`
(`classify_stls.py:1117`) and the `.npy` write (`:1105`). Don't let it read as
"Done never leaves the GPU".

### R6. Typo — LOW

`pre---compile` at `classify_stls.py:616`.

### R7. `needs_embed=False` retires outside the admission window — LOW

On that path the file retires via `CachedHit` while the child still holds
render work, so the admission counter is not what bounds it — the bounded task
queue is. Self-limiting, but "one window, three consumers"
(`data_structures.md:340`) is doing slightly more work than stated.

## P2.3 Decisions taken during this pass

Recorded here because they change `pose.py`'s contract and belong in the
design note, not only in a findings list.

### P2.3-A. Source vocabulary: `forced | geometry | siglip | vlm`

**What `source` actually means**, which no document stated plainly before:

```python
# classify_stls.py:466-478
up, source, margin = pose.UP_CANDIDATES[geo_idx], "heuristic", None
...
idx, margin = pose.combine_up(geo_scores, sig)
if idx != geo_idx:
    up, source = pose.UP_CANDIDATES[idx], "ensemble"
```

Every label but `forced` sits on the axis *did this tier **move** the answer* —
not *did it run*. The ensemble runs on every model, so `"ensemble"` cannot
mean "the ensemble decided"; it means "the combined pick differed from
geometry's". `apply_arbiter` (`classify_stls.py:506-510`) behaves the same
way: a paid VLM call that **confirms** the pose leaves the label alone, so
`pose_source` undercounts arbiter usage too.

The doc's `"forced" | "geometry" | "ensemble" | "vlm"` mixes axes —
`geometry` names an input, `ensemble` names a mechanism — which is what makes
it read as a tier ladder. Renaming **both** halves fixes the axis:

| label | meaning |
|---|---|
| `forced` | the user's `--up-axis` |
| `geometry` | geometry's pick stood |
| `siglip` | SigLIP moved it off geometry's pick |
| `vlm` | the arbiter moved it off what the ensemble concluded |

Chosen over the open question's `"confirmed"` for two reasons: it keeps one
axis (whose answer prevailed) where `confirmed` mixes a state with two
mechanisms, and it stays true for the `--no-up-ensemble` case, where
`confirmed` would be actively false — nothing confirmed it. The "did SigLIP
run at all" question stays where it already lives and is already load-bearing:
`margin is not None` (`pose.py:141-156`). Put that sentence in the `Pose`
docstring so it is never re-derived.

Two caveats to carry:

* **`siglip` slightly overclaims.** `combine_up` takes the argmax of
  `geo_weight * unit(geo) + unit(siglip)` (`pose.py:268`), so a compromise
  candidate ranked second by both can win — a case where `ensemble` was
  literally accurate. Expected to be rare, **not measured**; neither cache
  stores the component scores, so quantifying it needs a re-resolve pass
  (`eval/tile_count.py` already computes both arms).
* **The latent overload is real but not live.** `--no-up-ensemble` also
  produces the agreement label, with `margin: None`. **0 of 4092** entries
  across both caches have a null margin, because ensemble-available runs
  upgrade them in place (`classify_stls.py:1021`).

### P2.3-B. The embedding token becomes `up_str(pose.up)`

**Why the token exists at all**, which was likewise unwritten. The cache is
not keyed on pose *source*; it is keyed on `up`, and `source` is a **proxy for
determinism**. Only `up` changes the pixels
(`mesh.rotate(rotation_to_z_up(entry["up"]))`, `classify_stls.py:1081`), and
`up_token` is the only pose-dependent component of `cache_key`.

Geometry's answer is a deterministic function of the file — `up_axis_scores`
is seeded (`pose.py:54`) — and the file's identity is already in the key, so
its up vector is *redundant* and elides to the legacy `--up-axis` string.
SigLIP and VLM answers are not reproducible from the file, so they splice the
vector in. The elision existed to keep a populated pre-pose-pipeline cache
valid (`docs/superpowers/plans/2026-08-10-pose-pipeline.md:194`,
`docs/learnings/2026-08-11-canonical-pose.md:15-16`).

**Decision: drop the elision.** The token becomes the render identity for
every pose:

```python
def embed_cache_token(pose):
    """The render identity of a pose. Only `up` changes the pixels."""
    return up_str(pose.up)
```

Rationale beyond tidiness — the elision costs real duplication today.
`--up-axis z` yields token `"z"`; an auto run whose geometry resolves to
`[0,0,1]` yields `"auto"`. Same file, same `up`, `rotation_to_z_up([0,0,1])`
the identity in both: **identical pixels, two keys, two `.npy` files.** Under
the honest token they are one entry, and a pose that changes label without
changing axis stops re-embedding. It also takes the source string out of the
cache key entirely, which is what makes `P2.3-A` a plain rename instead of a
1531-model re-embed (994 + 537 `ensemble`-sourced poses).

## P2.4 What checked out — do not re-verify

* **All of pass 1's D-findings**, per the table in §P2.1.
* **`11–66 ms` parse** — `classify_stls.py:399`, `actors_proposal.md:128`.
* **`--inflight 3`, queue depth 4** — `eval/overlap_spike.py:250-251`.
* **275 ms upload vs 34 ms re-show ≈ 8×**, and **28.4×** at 2048 px.
* **`make_contact_sheet` at `pose.py:300`.**
* **`Pose` field order is valid Python** — the four non-default fields precede
  `margin` and `front_view`.
* **`--compile` mechanics**: pre-existing eager keys stay byte-identical
  (`classify_stls.py:616`); `save_run_params`' `is not None` filter correctly
  preserves `compile: false` (`:749`); the **bound method** is compiled, not
  the wrapper (`:887`) — the null-canary lesson from the write-up applied.
* **Cache census**, for sizing the migration:

| cache | pose entries | agree / override | `.npy` | on disk |
|---|---|---|---|---|
| `embed-cache2` | 2943 | 1949 / 994 | 2990 | 788 MB |
| `embed-cache3` | 1149 | 612 / 537 | 1148 | 299 MB |

## P2.5 Migration plan

Five coupled changes. **The order is forced** — each step needs something the
next one destroys, or provides something the next one depends on.

| step | what | why it is here |
|---|---|---|
| M0 | cache-root schema stamp | makes M1 detectable instead of a silent re-embed |
| M1 | honest `up_str` token + migration arm | needs the pose cache intact for the `up` mapping |
| M2 | source rename | free only once M1 takes `source` out of the key |
| M3 | pose ↔ embedding key gaps | `--compile`, render size, no `EMBED_CACHE_VERSION` |
| M4 | doc corrections | independent |

### M0. Stamp the cache root with a schema version — FIRST

**The stamp is what makes M1 detectable; without it a key-scheme change fails
silently.** A cache whose keys moved does not error — every lookup simply
misses, and the run re-renders and re-embeds the whole collection. That is not
hypothetical: it is the failure `migrate_cache_keys.py`'s own docstring exists
to prevent — *"Without this every entry misses and the next run re-renders,
re-embeds and re-resolves the whole collection — hours, and real money once a
pose entry is VLM-sourced."*

It also costs the migration tool real complexity today. Because nothing records
which scheme a cache was written under, `migrate_cache_keys` cannot read it —
it takes the old scheme as a **parameter** and reconstructs keys from it
(`old_base(f, old_root, new_root, absolute)`, `old_identity`, `old_cache_key`,
`old_render_key` — `migrate_cache_keys.py:57-93`). Every future migration
inherits that: another `old_*` family, and a caller who has to know which one
applies.

```python
# classify_stls.py
CACHE_META_FILE = "cache-meta.json"
CACHE_VERSION = 1     # 1 = up_str token (M1) + embeds/ & renders/<cfg>/ layout
                      # 0 = unstamped: the up-token elision, pre-M1

def cache_version(cache_dir):
    """0 for any cache written before the stamp — i.e. every current one."""
    p = Path(cache_dir) / CACHE_META_FILE
    return json.loads(p.read_text())["cache_version"] if p.exists() else 0
```

On mismatch, **refuse and name the migration command.** The whole value is
turning a silent 2943-model re-embed into one line of output.

Four design points, each with a reason:

* **A separate file, not `run-params.json`.** That file is merge-only and
  accumulates keys forever — `embed-cache2/run-params.json` still carries
  `"pool": "mean"`, a key no longer in `RUN_PARAMS_KEYS`
  (`classify_stls.py:678-681`), surviving because `save_run_params` does
  `load_run_params(...) | {...}` (`:748-749`). Useful for durability, wrong
  for an authoritative schema record; and its semantics are "defaults for the
  next run", a different lifecycle from "how to read this cache".
* **A manual integer, not a hash of the key format.** The repo *deliberately*
  makes byte-compatible changes to `cache_key` — `elev` and `|compiled` appear
  only when non-default, precisely so existing keys survive
  (`classify_stls.py:610-617`). An auto-derived schema hash would fire on
  exactly those, forcing a pointless migration each time and punishing the
  discipline the repo already practises.
* **Root-level, unlike `POSE_CACHE_VERSION`.** Poses are stamped per entry
  (`v: 4`), which lets `load_pose_cache` drop selectively and carry mixed
  versions (`pose.py:126`) — right for one JSON file read whole. Stamping
  4138 `.npy` files would mean opening every one to decide, so root-level
  all-or-nothing is the correct trade here.
* **Write the stamp last.** `migrate_cache_keys` should bump it only after all
  moves succeed, so an interrupted migration stays at the old version and
  stays re-runnable — which the tool already is by design.

Then `cleanup.sh` must never delete `cache-meta.json`, the same rule
`run-params.json` already has. If `--clear-caches` removed it, an
already-migrated cache would read as version 0 and the next migration would
compute old keys for files that no longer have them.

Optionally record the `cache_key` format string alongside the integer, marked
**informational only, never compared** — it makes a mass-miss diagnosable by
eye without inviting an automatic check that would trip on the compatible
changes above.

### M1. Honest token + a `migrate_cache_keys.py` arm — AFTER M0

`P2.3-B` changes every key: `"auto"` → `"0,0,1"`, `"ensemble:0,0,1"` →
`"0,0,1"`, `"z"` → `"0,0,1"`. **This is a rename, not a re-embed.** The `.npy`
content is a pure function of `(file, views, render_size, up, model, elev,
compile)` — only the sha1 of the key string moves. 4138 files, ~1.1 GB never
read; metadata-only on one filesystem.

`migrate_cache_keys.py` is the home: it already re-keys a cache whose scheme
moved, is dry-run by default, re-runnable, and leaves unmatched files in
place. Its docstring already carries the constraint this needs — *"Order is
forced: poses first, because an embedding's key contains the pose's up-token,
then embeds, then renders."* Adding an old-token arm is an extension of that
tool, not a new migration.

Two constraints:

* **Run before any `POSE_CACHE_VERSION` bump.** The new key needs each file's
  `up`, which only the pose cache knows, and `load_pose_cache` drops
  mismatched versions on load (`pose.py:126`). Bump first and the mapping is
  gone and every `.npy` orphans — that *is* the expensive re-embed.
* **The 47-file discrepancy is not orphaned embeddings.** `embed-cache2` holds
  2990 `.npy` total, but `embeds/` holds exactly **2943 — 1:1 with the pose
  cache**. The other 47 sit loose in the cache *root*, residue from the
  earlier layout migration that `migrate_cache_keys` reported and left by
  design. `embed-cache3` has 1148 in `embeds/` and none in the root. So the
  M1 mapping is complete for every file it needs to move; do not read the
  count as migration failure.

### M2. The source rename — AFTER M1

Once the token is `up_str(pose.up)`, `source` is out of the cache key, so
`heuristic → geometry` and `ensemble → siglip` are both plain renames with no
compatibility shim. `from_cache` maps the two old values; `to_cache` writes
the new ones; disk converges on the first load+save cycle.

Sites: `classify_stls.py:466` (write), `:1061` (`pose_changed` — behavioural,
drives render refresh only), `:451`, `:926`; `pose.py:170` disappears with the
token; `tests/test_pose.py`, `tests/test_migrate_cache_keys.py`;
`cleanup.sh:36,106` (text only — the filter at `:104` keys on `"vlm"`);
`eval/siglip_up.py:142,148`.

**Amend `OPEN_QUESTIONS.md:118-120` in place** rather than striking it: the
entry proposed `"confirmed"` and "Needs a pose-cache migration", and both
parts resolved differently. Per the repo convention the answer goes into the
entry.

### M3. Close the pose ↔ embedding key gaps

Three instances of one bug: **an input that moves the pose but not the key.**
They read as one entry, not three.

1. **`--compile`** — `R1` above; unmeasured on the pose path.
2. **Render size** — already recorded
   (`docs/learnings/2026-08-12-where-a-7-hour-run-went.md`, "the pose cache is
   not keyed on render size"): the ensemble's tiles come through the main
   renderer, so its answer depends on `--render-size`, which `file_identity`
   does not carry.
3. **No `EMBED_CACHE_VERSION`.** After M1 the embedding key is honest about
   `up`, but nothing versions the *derivation*. If `load_mesh` →
   `up_axis_scores` → `rank_up_scores` ever changes its answer for unchanged
   bytes, the pose cache notices (bump the version, re-resolve) and the
   embedding cache does not. It has held by luck of scope — v2→v3→v4 changed
   `geo_weight` and the arbiter sheet, never `up_axis_scores` — and the numpy
   parser swap was the near-miss, passing only because triangle counts and
   bounding boxes came out exact.

M1 shrinks (3) — an honest `up` in the key means a changed geometry answer
re-keys itself — but does not close (1) or (2), which move the pose *before*
`up` is written.

### M4. Doc corrections — any time

`R3`, `R4`, `R5`, `R6`, `R7`, plus folding `P2.3-A` and `P2.3-B` into
`data_structures.md` with their reasoning, since both replace text the note
currently argues for.

---

# Pass 3 — 2026-08-14, against `8b35316`

Three commits: `d159494` recorded pass 2, `b0ea59a` is the code half (M0–M2,
one third of M3), `8b35316` the doc half. Suite **185 passed, 1 skipped** — up
five from pass 2, the new ones covering the token migration and the stamp.

This pass reviews a **migration that has already run against both live
caches**, so it verifies outcomes on disk rather than reasoning about intent.
New findings carry `S` IDs.

## P3.0 Verdict

**The migration math is correct and independently confirmed; both defects are
in the guard around it, not in the re-keying.** Nothing landed at a wrong key,
nothing was clobbered, and `embed-cache3` migrated 100% cleanly. The commit
also avoided the one mistake most likely to have destroyed a cache silently
(§P3.4).

What is wrong is the boundary: the migration only visits files in the current
walk, and left 144 embeddings behind without reporting them (`S1`); and the
stamp's own "is this cache empty" test can mark a genuinely unmigrated cache
as current (`S2`), which is the exact failure M0 exists to prevent.

## P3.1 Independent verification of the migration

Worth recording as a method, because it audits any future key change with no
run and no collection access: **`pose.file_identity` is byte-identical to
`cache_key`'s first three fields** (`rel_path|mtime_key|size`). So every
embedding key — old scheme and new — is reconstructible from
`pose-cache.json` plus `run-params.json` alone, without `stat()`ing a single
STL.

Reconstructing both schemes for every pose entry and intersecting with
`embeds/`:

| cache | on disk | at new keys | still at old keys | neither |
|---|---|---|---|---|
| `embed-cache2` | 2943 | 2799 | **144** | 0 |
| `embed-cache3` | 1148 | 1148 | 0 | 0 |

"Neither = 0" is the load-bearing column: no file landed at a key belonging to
neither scheme, so nothing was mis-keyed or half-renamed. The commit message's
"2799 + 1148 re-keyed" is accurate; what it does not say is that 144 files
were left behind (`S1`).

## P3.2 Disposition of pass 2

| item | status |
|---|---|
| R1 | **recorded as a decision** in the precision write-up, escalation gate named as the exposure. Substance still open — see `P3.5` |
| R2 | resolved by `P2.3-A` and implemented |
| R3 | taken — all behavioural sites; `eval/siglip_up.py` correctly left (display-only `[:4]`, now reads `geom`/`sigl`) |
| R4, R5, R7 | taken in `data_structures.md` |
| R6 | fixed |
| P2.3-A | implemented — `forced\|geometry\|siglip\|vlm`, mapped on load |
| P2.3-B | implemented — token is `up_str(pose.up)` |
| M0 | implemented — see `S2`, `S3`, `S5` |
| M1 | implemented and run — see `S1`, `S4` |
| M2 | implemented |
| M3 | **one of three closed** — `EMBED_CACHE_VERSION` landed; `--compile` and render-size remain (`P3.5`) |
| M4 | taken |

## P3.3 New findings

### S1. 144 embeddings left at v0 keys in a v1-stamped cache, unreported — MEDIUM

`plan_token_moves` iterates `files` from `load_file_list`, so it re-keys only
embeddings whose file is in the current walk. `embed-cache2` holds 2943 pose
entries against 2801 walked files; the 144 difference was never visited. The
`.npy` files are intact and their poses are still known — **nothing is lost
and nothing is broken today.** Two things make it worth fixing anyway:

* **Nothing reports them.** `missing_t` counts *files with no cached
  embedding*; there is no reverse scan for *embeddings no file claimed*. The
  root migration path has exactly that (`orphans`), and the module docstring
  promises it — *"unmatched .npy files and renders are reported and left where
  they are."* The token path does not honour its own contract.
* **The door is now shut.** The cache is stamped v1, so `need_token` is False
  and a re-run prints "nothing to migrate" and returns. Should those models
  return (remount, restore, re-add) they miss and re-embed, and the only route
  back is hand-editing `cache-meta.json` to 0.

**Fix — smaller than the current code: drive the token migration from the pose
cache, not the walk.** Per §P3.1 both keys are computable from
`file_identity` + `run-params.json` with no filesystem access and no
`f.stat()`. That is not a proposal, it is what the verification above did, and
it reached all 2943. It also drops the coupling to a possibly-partial mount,
which is what caused this.

### S2. `require_cache_version` stamps an unmigrated old-layout cache as current — MEDIUM

The "is this cache empty" test is `pose-cache.json` or `embeds/`. A
pre-layout cache has neither — root-level `.npy` and nothing else — so it is
treated as empty and stamped current:

```
$ ls cache/            # one root .npy, one run-params.json
$ python -c "...; print(cache_version(d)); require_cache_version(d); print(cache_version(d))"
before: 0
after : 1 <-- stamped current, never migrated
```

Every key then misses, the collection re-embeds, and the guard never fires
again because the stamp reads v1. That is precisely the failure M0 exists to
prevent, in the one case where the cache predates everything else.

Not hypothetical: `embed-cache2` still holds **47 loose root `.npy` files**,
so the layout exists in this tree. The realistic route in is a forced
`--up-axis z\|y` cache, which writes no `pose-cache.json` at all
(`classify_stls.py:1234-1235`).

Widen the test to include root `*.npy` and `renders/`. The suite covers
"unstamped populated cache is refused" and "empty cache is stamped current"
but not this middle case.

### S3. The stamp's `cache_key_format` is stale on arrival — LOW

It reads `sha1(rel|mtime|size|views|render_size|up_token|model|pv[|e:...][|compiled])`
and omits `|ev{N}`, which M3 added in the same commit. Marked informational
and never compared, so nothing breaks — but eyeball diagnosis of a mass-miss
is its only job.

### S4. The deliberate collapse is indistinguishable from "already migrated" — LOW

`plan_token_moves` folds the forced/geometry key collapse into `already_t`, so
the output cannot tell a collapsed duplicate from a file that was already at
its new key, and the superseded `.npy` stays in `embeds/` as unreported dead
bytes. Same reverse-scan fix as `S1`.

### S5. The guard runs after `cache_root` — LOW

`require_cache_version` is called after `cache_root` in both `main()`
(`classify_stls.py`) and `test_categories.py`. `cache_root` can prompt and
re-anchor, so a user with an unreadable cache may be asked to answer a
re-anchor question before being told the cache cannot be read.

## P3.4 What is right — do not re-verify

**The trap that was avoided is worth naming explicitly**, because it was the
most likely way this commit could have quietly destroyed a cache.
`load_pose_cache` renames sources on load (`P2.3-A`), so a migration reading
poses through it would compute *new*-spelling sources, `old_embed_cache_token`
would not match `("vlm", "ensemble")`, and all 1531 override-sourced
embeddings would have failed to be found — a silent 1531-model re-embed. The
code reads `pose-cache.json` as **raw JSON** in the migration path and
`old_embed_cache_token` accepts both spellings, with a docstring saying why.

Also confirmed correct:

* The stamp is written **last** in both migration paths, so an interrupted run
  stays at the old version and re-runnable.
* `migrate_cache_keys` imports `cache_version` and `stamp_cache_version` but
  **not** `require_cache_version` — which would have made the migration refuse
  the very cache it exists to fix.
* The source rename is mapped on load rather than behind a
  `POSE_CACHE_VERSION` bump, so no pose is re-resolved or re-billed for a
  spelling.
* `EMBED_CACHE_VERSION` suppresses at 1, the same byte-compatible trick as
  `elev` and `|compiled`.
* `cleanup.sh` documents the never-delete rule for `cache-meta.json`; its
  filter still keys on `"vlm"` and is untouched, so the rename cannot reach
  the destructive path.
* Remaining `"ensemble"` strings under `eval/` are local harness arm labels
  (`("sig", …), ("ens", …)`), unrelated to pose source.

## P3.5 Still open

* **`R1` — the pose cache does not carry the compile regime.** Recorded as an
  accepted decision in `docs/learnings/2026-08-14-precision-and-compile.md`
  with the escalation gate named, which was the ask; `eval/compile_pose_flips.py`
  is now measuring the rate that decision rests on. Note that `b0ea59a`'s
  message lists R1 among what it takes — the substance (`file_identity`
  carrying no regime) is untouched, correctly, pending that number. Do not let
  the commit trail read as closed.
* **M3's render-size gap** — same shape, still open.
* **`S1`–`S5`** above. `S1` and `S2` before the next cache migration; the rest
  any time.
