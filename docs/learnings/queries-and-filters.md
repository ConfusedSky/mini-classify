## Open-set queries: detecting "not in the collection"

- Cosine scores are only comparable *within* a query — some phrasings run
  hot, some cold — so a raw threshold alone can't tell "present" from
  "absent"; something always ranks first.
- Z-score against the collection's own distribution per query. Plain
  mean/std z fails when a category is *well*-represented (8 robots inflate
  the mean → the query looks weak); robust z (median/MAD) fixes it.
- **Measured: no z cutoff separates modest correct matches from semantic
  near-misses.** Correct "skeleton" → TatteredTroopers at z 2.4–2.7;
  wrong-but-nearest "witch on a broomstick" → AurochRider (a mounted rider)
  at z 3.7. Layer the defenses instead: z < 2.0 = whole query is noise,
  suppress output entirely; displayed z + clickable render link covers the
  judgment calls no threshold can make. A raw-score floor (`--min-score` /
  `:min`) trims weak individual matches, but it is opt-in — the top-10 cut
  already hides most of what it would catch, so it earns its keep only on
  exhaustive listings.
- Near-misses are often *semantically legitimate* ("wizard with a staff" →
  OrcShaman, who carries a staff) — treat threshold tuning as UX, not truth.

## Filter gotchas

- Substring tags bite: `"supported"` matched inside "**un**supported", which
  in miniature packs means NO supports — the files you want. Strip the
  exception word before tag matching.
- When filter semantics change, bump the walk-cache key or stale cached file
  lists silently keep the old behavior.

