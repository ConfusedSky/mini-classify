## Elevation rings + the run manifest (2026-08-11)

Higher `--render-size` (512 → 2048) visibly improved classification and the
speed cost was acceptable, which made a second view axis worth paying for too.

- **"Turntable" was already the azimuth.** The original loop orbited
  `az = 2πi/n` at a hardcoded 20° pitch, so the axis actually pinned was
  *elevation*. Worth naming precisely: "add another rotation" is ambiguous
  between the two, and the fix is in the parameter nobody had exposed.
- `--elevations 20,-10,55` renders a full `--views` ring per elevation
  (product, not sum). Ordering is **elevation-major** so views
  `0..n_views-1` remain the first ring — `view0.png` keeps meaning the same
  camera as every previous run, which is what lets saved renders and
  `front_view` indices stay meaningful across the change.
- **A constant `up` silently kills azimuth near the poles.** Passing world
  `[0,0,1]` to `setup_camera` looks safe at any elevation short of ±90, but
  Filament's `lookAt` calls `up` degenerate once `|up · view| > 0.999` — that
  is **|elev| > 87.44°, not 90°** — and substitutes a *fixed* fallback up.
  The fallback doesn't rotate with azimuth, so a whole ring collapses: a
  `--elevations 20,89,-89,-20` run rendered its two polar rings as 8 copies
  of one camera (mean |view₈ − view₁₂| was 3/255, all of it shading; the ±20°
  rings differ by 13–17). The ±89° clamp was reasoning about the ±90°
  singularity and landed *inside* the broken band.
  Fix: carry `up` around the orbit instead of holding it fixed —
  `(-cos az·sin elev, -sin az·sin elev, cos elev)`. It is exactly orthogonal
  to the view direction, so `lookAt` never falls back, and below 87° it is
  the same frame `[0,0,1]` already produced (verified: max pixel delta 2/255,
  the warm cache stays valid). The poles are now ordinary cameras, so the
  clamp relaxes to ±90.
  Generally: a look-at `up` that is *near* the view direction is already
  broken, and it fails silently — identical-looking frames, no error. Derive
  `up` from the same parameters as `eye`, don't hardcode it.
- The pose contact sheet stays pinned at one 20° tile. Up-detection input
  shouldn't change shape as a side effect of a classify-side render flag.
- **Append-only cache keys, again.** Same trick as the pose token (above): a
  single default 20° ring appends *nothing* to the key string, so every key
  written before elevations existed stays byte-identical. Measured: 24/24
  real files still hit the warm 2048px cache (the 25th is the known
  FloatingRock2 RENDER_ERROR, never cached). Two uses in two sessions —
  treat "extend the key without disturbing the default path" as the default
  approach for cache-format changes here, not a one-off.
- `front_view` is an argmax over *all* views, so with multiple rings the hero
  shot can land on a non-primary elevation. Acceptable (more angles to find a
  face in) but it means heroes can shift on already-processed files.

**Config belongs next to the cache, not in the repo.** The three scripts each
re-declared `--views/--render-size/--model/--up-axis` under a comment reading
"must match the classify_stls.py run that built the cache" — a comment that
names a drift hazard is a design smell, not documentation. Options considered:
a committed `.env`, or a manifest written by the run itself.

- **A committed config can drift from the cache it describes** (edit the file,
  forget to re-run classify, downstream silently reads the wrong tiles). A
  manifest written *by* the run cannot: whatever built the cache is what
  describes it. Chose `<cache-dir>/run-params.json`, gitignored along with the
  cache — it's derived state holding an absolute path to an external drive,
  meaningless on another machine.
- Mechanism: `parser.set_defaults(**manifest)` before `parse_args()`, after a
  `parse_known_args()` pass to learn `--cache-dir`. Explicit flags still win
  because set_defaults only moves the fallback. Print which keys came from the
  file — silent hidden state is worse than the retyping it replaces.
- **Record the input directory too.** It's the same class of parameter (the
  walk cache is keyed on it) and it's the one argument you can't tab-complete:
  `/run/media/masa/Files\ and\ S/STL/Loot\ Studios`. Both downstream tools now
  run with zero arguments. Guard: a single-file classify run leaves the
  recorded collection root alone rather than clobbering it with a file path.
- Declaring the shared args once (`add_cache_args`) matters more than the
  defaulting: a new cache-identity flag can no longer be added to the writer
  and forgotten in the readers.
- Bonus catch: `cluster_models.py` never had a `--renders-dir` default, so
  contact sheets were opt-in. It inherits the classifier's from the manifest.

