# The refactored pipeline at scale (2026-08-18)

The first real run of the actor architecture after it was merged: DM Stash,
production settings (384 px, 8 views × 2 elevations, `--save-renders`,
`--pose-vlm off`, `--compile`), cold cache.

| | |
|---|---|
| files walked / after tag filtering | 292 → **133** |
| cold run | **6:34**, 2.97 s/model |
| warm re-run | **12.4 s** — 32× faster, CSV **byte-identical** |
| renders written | 2128 = 133 × 16, exactly |
| pose sources | 88 geometry, 45 siglip |
| cache size | 37 MB for 133 models |
| errors | none; clean child exit, no leaked VRAM |

**This is a baseline, not a speedup.** There is no old-pipeline number for
this collection, so nothing here says the refactor made anything faster. The
throughput claim that does exist is the overlap spike's 1.17–1.21×, measured
before any of this was built and thermally capped; see LEARNINGS "Overlap and
the thermal ceiling".

## What the numbers actually confirm

**The warm path is the product working.** 12.4 s against 6:34, with a CSV
identical to the byte and zero re-embedding, is the whole `route()` decision
table — pose cache hit, embedding cache hit, renders present — resolving
correctly 133 times. It also means the cache keys round-trip: every key
written by the cold run was recomputed identically from the same inputs a run
later. That is the single cheapest end-to-end check available and it should be
the first thing run after anything touches keying.

**2128 renders is exactly 133 × 16**, and rows, `.npy` files and pose entries
are all exactly 133. Invariant 1 — every admitted index retires exactly once —
holds at scale, not just in the driver's unit tests.

**The ensemble carries a third of the collection.** 45 of 133 poses came from
SigLIP rather than geometry, so on this collection geometry alone would have
been guessing on a third of the files. With `--pose-vlm off` nothing escalated
further, which is also why the run billed nothing.

**Tag filtering removes more than half.** 292 STL files, 133 models: the rest
are supports and non-model parts caught by `naming.skip`. Any per-model cost
estimate taken from a raw file count is roughly 2× too high on this
collection.

## Incidental

`--compile` announces "embeddings keyed as a separate cache regime" — the
compile flag is part of the cache key, so a compiled run cannot contaminate an
uncompiled cache. That is the mitigation for the drift OPEN_QUESTIONS records
as disqualifying (`torch.compile`'s 1.10× "would seed the permanent `.npy`
cache inconsistently"); the keying makes the two regimes independent rather
than the flag unusable.

At 37 MB per 133 models the cache is small because renders are jpg — the
decision recorded under "`compress_level=1` on saved renders", which took the
collection's render footprint from ~27 GB to ~2 GB.
