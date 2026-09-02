# Week in review — 2026-09-01 (as received; resolutions at the end)

mini-classify week in review — 27 commits, 3 streams, suite green (733 passed, 1 skipped)

Cross-cutting good news first: the cross-contract check passed on every route
(/query, /similar, /poses, /under, /status) — field names, nullability, status
codes, warming 503s, POSES_MAX=1024 on both sides, and the realpath-vs-lexical
pose matching all match what model-browser's semantic.ts comments assert.
Runtime probe confirmed this producer can never emit a bare-null 200 body, so
model-browser's null-body hole (my earlier report) is consumer-side defense,
not a live bug.

## Should fix before next use

1. MAJOR — migrate_cache_keys.main writes the pose cache through a non-atomic
   writer: pose.save_pose_cache is a bare write_text, and done.py's docstring
   claiming "nothing in the pipeline calls it" is now false. A crash
   mid---apply tears pose-cache.json and the tool dies re-running — on the
   most expensive file in the cache. Route through cachedir.write_atomic (and
   likewise the plain write_text of run-params.json two lines later).
2. Migration bypasses resolve_root's anchoring — a subdir-scoped invocation
   re-keys the whole cache relative to the kit and silently drops every pose
   outside it on --apply. Call resolve_root and refuse the "subdir" case
   (nothing needs migrating there anyway).
3. --skip-embed run with divergent flags still rewrites run-params.json
   (classify_stls.main writes unconditionally) — the next bare serve_api.py
   misses every key: the CacheUnusable shape c05c14d closed for pre-flight
   deaths, surviving a run that embeds nothing.
4. Gemini arm never got the pin validation 26f601f gave the glm arm
   (resolve_pose_vlm) — a foreign --pose-vlm-model under --pose-vlm gemini
   reaches the Vertex URL and folds into permanent arbitrated: "rejected",
   the exact catastrophe the glm check exists to stop.
5. GLM solo tiles are sent uncapped (pose.ask_vlm_up's glm branch, no
   thumbnail cap) — fine at the default 512, but a --render-size 1024+ run
   feeds unmeasured 6–12× token costs; cap at SHEET_THUMB (bit-identical at
   the default).

## Worth doing (minor)

- Transport error-split duplicated between _ask_gemini and _ask_glm — three
  review passes have each found a transient pinned permanent; one shared
  split_transport helper gives the retry contract a single home.
- surface.md's /under prose example predates n_scanned/covers and contradicts
  the jsonc block in its own section; and "/poses echoes every requested path
  as a key" overpromises on duplicate batch keys.
- FOV coupling between _shoot's hardcoded 45.0 and tight_view_cams' default
  is pinned nowhere — one FOV_DEG constant for both.

## Nits

- parse_tile_answer admits JSON true as tile 1
- arbiter_id stamps literal "claude/None"
- two stale comments in the degrade path
- --repose announcement prints before the forced-axis no-op check
- EMBED_CACHE_VERSION's changelog and RECIPE_VERSION both claim the tight
  framing (one cross-reference sentence closes it)
- load_embed prints where 445b2b0 moved to logging
- /status's covers builds a throwaway np.arange
- OSError retry breadth in load_siglip
- dead c unpack in a test

## Cleared

The _parts symlink-spelling heal heals both directions without clobber;
EMBED_CACHE_VERSION 2 is consistent at every writer/reader (no half-loads);
the c7b65b5→c05c14d run-params placement is right with both negative halves
pinned; dtype/device following is parametrized-pinned; /under//query input
confinement is tight (verified at runtime, including 422s and the .3mf
honesty discriminator).

## Housekeeping

The working tree has uncommitted edits to the three run_*.sh scripts
(cache-dir retarget to embed-cache512 + a typo fix) — deliberate-looking,
but uncommitted.

## Resolutions (2026-09-01, four commits; suite 743 passed, 1 skipped)

Every diagnosis was re-verified against the code before briefing; two claims
adjusted below. Each behavioral fix carries a regression test falsified
against the unfixed code.

- ~~1 (MAJOR, atomicity)~~ + ~~2 (anchoring)~~ — "Migrate the cache without
  tearing it or re-anchoring it to a kit". **Fixed at the call site, not as
  proposed**: routing `pose.save_pose_cache` through `cachedir.write_atomic`
  is forbidden by interfaces.md's import-rule table (`pose` is the leaf both
  process sides import) and its guard test, and would have overturned J7.
  `migrate_cache_keys.main` now writes pose-cache.json and run-params.json
  itself via `write_atomic` (mirroring `Done.flush`'s one-liner), which makes
  done.py's "nothing in the pipeline calls it" true again — the docstring
  stands unedited; `save_pose_cache` instead gained a docstring warning off
  the next tool author. New `migration_root` asks `identity.resolve_root`
  and refuses the "subdir" case before any write.
- ~~3 (--skip-embed manifest)~~, ~~4 (gemini pin)~~, ~~--repose nit~~ —
  "Three refusals in the CLI". `save_run_params` is guarded by
  `not args.skip_embed`; the gemini arm rejects an org/model-shaped pin ahead
  of the gcloud checks, symmetric with glm's; `--repose` under a forced
  `--up-axis` is refused like the rest of its family. The manifest tests
  needed a new harness (`a_run_that_reaches_the_driver`): the existing
  helper dies in the model load, *before* the write site, so it could not
  falsify the guard — a positive control now pins the write half too.
- ~~5 (GLM tile cap)~~, ~~transport split~~, ~~parse_tile_answer~~,
  ~~arbiter_id~~ — "pose: cap GLM's solo tiles, and give the transport split
  one home". Solo tiles ride `SHEET_THUMB` like every other send path
  (bit-identical at the default size); `_split_transport` is the retry
  contract's single home, pinned by the two byte-unchanged contract tests;
  bool is excluded from tile answers; `arbiter_id("claude", None)` stamps
  bare "claude". A pre-existing "claude/None" stamp would read as foreign to
  `--repose`, but no cache in this repo's use has one (every embed-cache512
  escalation was answered by gemini-3.5-flash).
- ~~FOV coupling~~, ~~/poses overpromise~~, ~~version cross-ref~~,
  ~~load_embed print~~, ~~/status covers~~, ~~load_siglip retry~~,
  ~~dead unpacks~~ — "Pin the couplings a comment was holding".
  `renderer.FOV_DEG` now feeds both sides (no RECIPE_VERSION bump — same
  float, same pixels); surface.md promises *distinct* paths as keys with the
  duplicate collapse stated and tested; the narrowed retry propagates a
  device-transfer OSError instead of silently going online.
- **/under prose (minor, first half): already fixed** before this pass —
  commit c075498 rewrote exactly that section; the claim described a state
  the tree had moved past. Only the /poses sentence needed work.
- **"Two stale comments in the degrade path" (nit): not located.** A close
  read of every degrade-related comment in src/pose.py found them internally
  consistent. Needs the reviewer's line references to action. Nearest
  candidate is not a comment: the retired ollama backend is still fully
  present as unreachable code (`_ask_ollama`, `ollama_available`, the
  `ask_vlm_up` branch) while `poser.VLM_BACKENDS` rejects it — removing it
  is a separate decision, left open here.
