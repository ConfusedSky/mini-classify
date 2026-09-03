## Parts, duplicates, and whether SigLIP can see part-ness (2026-09-03)

Prompted by 75mm models and duplicates surfacing in search results. Census
and probe both ran against embed-cache512's walk (3,380 files, STLLibrary);
the probe over its cached embeddings, `eval/part_probe.py`, no rendering.

### The duplicate census

- **The filter's scale tag has three competitors it never met.** `"75mm"`
  catches Loot Studios' directories, but DM Stash spells the duplicate as a
  bare **`75_` filename prefix** (`75_Unsupported_AlphaAlm_Body.stl` beside
  its `32_` twin — **57 files, every one with an exact 32_ sibling in the
  walk**), and Bestarium spells it `_75 mm scale` / `150mm` (18 files —
  which have **no** 32mm twin, so they are unique kits the tag's rationale
  does not cover, and they stay). `_searchable` keeps separators, so a
  spaced spelling can never hit an unspaced tag.
- **Byte-identical duplicates: 98 md5-identical groups** (166 by the
  stem+size proxy; the other 68 differ in bytes — `(repaired)` re-exports).
  Three mechanisms: the same accessory copied into one character's Kit I and
  Kit II loadouts (`Requiem_Holding_Hand_L` ×8); a kit copy plus a
  "Standalone Weapons & Hands" copy; and **whole subtrees unpacked twice**
  (`Nagarot_Thrall_F_(f)`, a 25 MB complete character, indexed twice) —
  which is why dedupe cannot be a part-vocabulary filter.
- **Scale twins are invisible to byte-dedupe by construction** — scaling
  rewrites every vertex, so 32/75 pairs share neither stem nor size nor
  bytes. That class is name-structural only.

### The probe: complete-model vs part, zero-shot over cached embeddings

Margin = best part-prompt sim − best complete-prompt sim, softmax pool,
terrain and busts deliberately in the *complete* bank. Two constraints set
in advance: a miniature whose hands ship separately (DM Stash `*Body*`) must
read complete, and terrain must not read as parts.

| cohort | n | flagged at margin>0 |
|---|---|---|
| Standalone-Weapons dirs (parts by construction) | 197 | 89.8% |
| `_L`/`_R` handed stems | 445 | 91.9% |
| DM Stash `*Body*` (complete, hands separate) | 90 | **0.0%** (median −0.057) |
| Loot "Environment" trees | 154 | 13.0% — almost all kit *fragments* (plugs, pins, door flaps) |
| everything else | 2,578 | 30.6% |

Of 519 clean single-file character models, 4.0% flag — nearly all
standalone objects (Ladder, Trident, Portal, Streetlight); true character
mistakes are 2 (~0.4%). The probe's misses mirror the same ambiguity:
Chalice, Fireball, Grimoire — parts that look like complete objects. At
margin>0 the corpus reads **~41% parts**, which the modular-kit structure
supports. The floor trades cleanly: 0.02 takes character FPs 4.0% → 1.3%
and terrain 13% → 5.8% at 91% → 75% part recall — and every part lost to
that floor carries name evidence, which is the case for a **two-tier floor**
(name evidence → flag at >0; no evidence → flag only above 0.02) rather
than one scalar. Verdict: not too wide; both constraints upheld.

### What shipped, what didn't

Shipped (this date): `"standalone"` joins `NON_MODEL_TAGS` — five directory
spellings plus 80 loose `Standalone_*.stl`, no non-accessory occurrence in
the library — and `SCALE_PREFIX` (`^75_`) joins `skip()` as an anchored
rule, not a substring (`1975_Tank.stl` must survive). Together: 3,380 →
**3,092** files (231 + 57). The spaced Bestarium kits stay, decided
explicitly. Falsified: all 8 new naming tests fail against the old filter.

Deferred to the repo issue "Part/accessory handling": the two-tier
partedness flag (exclude-by-default in API/REPL, `Basing/`+Environment
exempt), the byte-dup skip-list, VLM arbitration of the ambiguous band.
