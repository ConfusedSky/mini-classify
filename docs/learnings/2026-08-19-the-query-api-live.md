# The query API, live against the primary cache — 2026-08-19

Phase 3 of `docs/api/implementation.md`: run the real server against
`embed-cache2` and measure. Everything below is from one session with the
library volume mounted, `serve_api.py --cache-dir embed-cache2`, SigLIP on the
4060.

## What it does

| | |
|---|---|
| models loaded | 2801 (matrix `2801 × 16 × 1152`) |
| time to `ready` | **16.0 s** — the SigLIP load is nearly all of it |
| time to first `/status` | **immediate**, well before ready |
| `/query` latency | **median 49 ms** (n=12 distinct queries, min 47, max 63) |
| `/similar` latency | median 50 ms (n=5) |
| first query after ready | 202 ms — a one-off, the steady state is 47 |

49 ms is the text forward plus a `2801 × 16 × 1152` matmul, and it means the
GPU lock is not the bottleneck anyone feared: the card is busy for tens of
milliseconds per request, and model-browser sends one request per Enter rather
than one per keystroke.

## Two SigLIP instances fit on the 4060

The open question was whether the server may run while `classify_stls.py`
does, since `/reload` exists for exactly that workflow. Measured by running
`test_categories.py` (which loads its own model) against the live server:

| | MiB |
|---|---|
| server alone | 2374 |
| server + a second SigLIP process | **4740** |
| card | 8188 |

So ~3.4 GB of headroom. A classify run's SigLIP is the same model; its render
child is on the **AMD iGPU**, not the 4060, so it adds nothing here. Its
measured embed peak is ~2.5 GB rather than a bare load, which still leaves
~0.9 GB. **They coexist.** This is unlike the ollama constraint in CLAUDE.md,
which is about a 10.1 s model *reload* thrash rather than capacity — two
resident SigLIPs never evict each other.

Not measured: throughput under real contention. Both processes sharing SMs
will slow each other; the claim here is only that neither fails.

## The API agrees with the REPL exactly

The parity that matters, since `test_categories.py` is where querying actually
happens. Same cache, same `--pool softmax`, query `"skeleton"`:

**Identical top-10 — same models, same order, same scores to 3 dp.** Both go
through `src/query.py`, which is the point of extracting it; this confirms the
server adds nothing of its own to the answer.

## Scoping narrows the judgement, not just the list

Same query at three scopes:

| scope | models | best z | top hit |
|---|---|---|---|
| whole collection | 2801 | 4.5 | `32mm_Skeleton` |
| `DM Stash` | 133 | 2.4 | `32_Unsupported_Theldranax_BodyWhole` |
| `DM Stash/Crimson Masquerade` | 71 | 2.3 | same |

This is the designed behaviour and worth seeing on real data: robust z is
computed over the *scoped* subset, so the same text is judged against the kit
being browsed. The collection has a real skeleton and `DM Stash` does not, and
the z drop from 4.5 to 2.4 is the API saying so — a UI scoped to that kit
should present the result far more weakly than the raw score suggests.

## `/reload`, and what a fresh walk found

| call | time | result |
|---|---|---|
| `/reload` | 1.2 s | 2801 models, `missing 0` |
| `/reload {"rescan": true}` | 1.6 s | 2801 models, **`missing 595`** |

**And a correction, because this run is where it surfaced.** Earlier notes
justified "no walk in a request" with a ~32 s cold walk — a figure borrowed
from model-browser's measurement of a *spinning USB exfat* drive. This library
is not that: `/dev/sda1`, **ext4**, `rotational=0`. A full `find_stls` over its
19133 entries takes **0.07 s** (three consecutive runs, warm). A per-request
walk would have been affordable, and the design argument had to be rebuilt.

The rebuilt version is the more durable one anyway: **request cost is
independent of the storage**, so the interface does not get slower if the
library moves to an HDD — a real possibility, and exactly the degraded case
worth insuring against, at a cost of nothing. "A walk would be affordable" was
a fact about today's disk rather than about the design. The two
speed-independent reasons stand alongside it: `n_scanned` is a stable claim
about the index rather than about the tree right now, and request cost never
scales with the collection.

The full picture for this volume, once model-browser re-checked its own rows
against it:

| | |
|---|---|
| device | `/dev/sda1`, ext4, `rotational=0`, 476.9 GB, label `STLLibrary` |
| contents | 19,134 entries / 5,077 dirs / 453 zips |
| `find_stls` warm (this repo) | **0.07 s** |
| full walk warm, via model-browser's API | 0.54–0.86 s (it also reads zip central directories) |
| full walk **cold** | **2.92 s** |

So cold is ~3 s, not ~32 s. The 32 s belongs to a *second* volume — spinning
USB exfat — which model-browser's proposal has always carried as its own
labeled row. The measurement was never wrong and was never a claim about this
disk; **the error was entirely in the transfer**, reading across the wrong
column.

The lesson is narrower than "measure your own hardware": the borrowed number
made a *correct* decision look like it rested on a fact it did not, and nothing
in the code would have revealed that. It also had a cost on the other side —
that repo's `listing-tree-cache` is priced against the 32 s row, and if the
spinning drive is retired the cache buys about three seconds for a
revalidation protocol and an exfat timestamp-granularity risk. Pushing the
correction back rather than quietly fixing my own files is what surfaced it.

And the decision here is *more* justified than before, not less. Masa, in this
session: "there is still the possibility of using the hdd later and this design
works better in that degraded scenario." Whether that drive actually returns is
unsettled and tracked in `OPEN_QUESTIONS.md` — this line originally cited that
file before the entry existed, which is the same class of mistake as the one
this write-up is about, made while writing it up.

The 595 is the interesting part and not a defect: a fresh walk sees **3396**
STLs where the last classify run cached 2801, so the library has grown by 595
models since. The scope block reports it rather than hiding it:

| scope | indexed | scanned | status |
|---|---|---|---|
| whole collection | 2801 | 3396 | `partial` |
| `DM Stash` | 133 | 133 | `indexed` |
| `Loot Studios` | 1206 | 1275 | `partial` |

This is exactly what the field was for. A UI can now say "1206 of 1275 models
here are searchable" instead of quietly returning results from a fraction of
the folder — and `partial` is actionable, since the fix is a classify run.

Note the side effect: `--rescan` rewrites the walk cache, so every tool
reading this cache now sees 3396 files and reports 595 uncached until the next
classify run. That is more accurate, not less, but it is a change to shared
state made by a *read* endpoint.

## Odds and ends confirmed live

* The three scope rejections return 422 / 400 / 404 on real paths.
* `/similar` on a green dragon returns its own variants first, then
  `32mm_XenowyvernRider_Dragon` — a different dragon.
* Poses come back as designed: the dragon is `up: [0,1,0]`, `source: siglip`,
  front view 5 at azimuth 225° / elevation 20°. Y-up is one of the three axes
  where the azimuth shortcut is 90° wrong (surface.md §pose), so the most
  common model in this collection is also the case a consumer must not
  shortcut.
