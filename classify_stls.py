"""Zero-shot STL classification: multiview renders scored against text categories with SigLIP.

Usage:
  python classify_stls.py /path/to/stls --categories categories.txt --out results.csv
  python classify_stls.py model.stl --save-renders   # single file, keep debug renders
                                                     # under <cache-dir>/renders/<camera config>/
  python classify_stls.py ... --instrument           # per-stage timings, both GPUs
  python classify_stls.py ... --profile              # torch trace of the parent, to ./log

Renders each mesh from several viewpoints (Open3D offscreen, in a render child
process), embeds the views with SigLIP in this process, and ranks the pooled
similarities against text embeddings of the categories.

**This file is the CLI entry, and now only that** (docs/actor-refactor/): its
argparse, the run-params it writes, the cache guards it runs before touching
anything, and the wiring. The loop lives in `src/driver.py` and every stage it
drives is one of the `src/` modules.

Nothing here is imported by anything else. That is the point of the eval-debt
cleanup that emptied it: the evals, the tests and the sibling tools used to
import a *script* for the cache layout (`src/cachedir.py` now), for reading
cached embeddings back (`src/embed_store.py`), for a text embedding
(`src/embedder.py`'s `embed_raw`/`embed_texts`), and for names this file only
forwarded on behalf of `src/identity.py`, `src/loader.py`, `src/renderer.py`
and `src/done.py`. Each consumer imports the owner now, so `classify_stls.py`
can be read as one program rather than as a library with a `main()` attached.
The single-process render helpers the pose evals once called
(`render_views`, `render_up_candidate_grid`, `resolve_up`) went with it —
`eval/rig.py` drives the production `Renderer` instead.

Viewpoints are a turntable of --views azimuths at each --elevations pitch, so
--views 4 --elevations 20,-10 gives 8 renders per mesh. Every run records its
parameters in <cache-dir>/run-params.json; cluster_models.py and
test_categories.py default from that file, so cache-identity flags (and the
input directory) only have to be typed once, here.

Meshes are stood upright first, from three tiers of evidence: flat print-base
geometry with a confidence ratio, a SigLIP vote over the six up-candidate tiles
(the two averaged, always), and a VLM arbitrating low-confidence cases
(--pose-vlm). The front-facing view index is recorded per file (front_view
column) so downstream tools can show the render that actually faces the viewer,
and resolved poses persist in <cache-dir>/pose-cache.json.

**Nothing heavy at module scope, deliberately.** `mp.get_context("spawn")`
makes the render child re-import this file as `__mp_main__` before it runs
`run_child`, so anything imported at module scope is imported in the child
too — and a torch import there is exactly what the child-side import rule
forbids (interfaces.md's import table: SigLIP lives in the parent, and torch
in the child costs VRAM and startup for nothing). Since the cleanup the same
reasoning is applied one step further: the module scope is stdlib plus
`instrument`, `src.identity` and `src.cachedir`, all of which are cheap, and
everything else — torch, open3d, numpy, PIL, transformers, and every `src/`
module that pulls one of them — is imported inside `main()` or
`resolve_pose_vlm()`, where it is used. `import classify_stls` is therefore
free, which is what the child's `__mp_main__` import gets.
"""
import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from src import instrument
from src.instrument import stage
from src.cachedir import (RUN_PARAMS_FILE, add_cache_args, apply_run_params,
                          cache_root, embeds_dir, load_file_list,
                          render_index, renders_dir, require_cache_version,
                          save_run_params, total_views)
from src.identity import render_key

# What the render child may hold in host-side meshes before its LRU evicts
# (RenderConfig.budget_bytes). A soft bound: in_flight meshes are never
# evicted, so the hard worst case is the admission window x the heaviest mesh
# (~450 MB at 3 x 150 MB — data_structures.md §residency). Not a flag: the one
# knob is the admission window, and this follows it.
RESIDENT_BUDGET_BYTES = 512 * 1024 * 1024

# Where --profile writes its tensorboard trace when given no argument.
PROFILE_DIR = "./log"


def profile_dir(argv=None):
    """--profile, read before `main()` runs.

    The flag is declared on main()'s own parser as well — that is what puts it
    in --help and what stops argparse rejecting it — but the profiler has to be
    entered *around* main(), so the value is also read here from a throwaway
    parser that ignores every other argument.

    Off by default, and that is the point: this used to be an unconditional
    `torch.profiler.profile` around main(), which wrote a ~280 MB trace on
    every single run."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--profile", nargs="?", const=PROFILE_DIR, default=None)
    known, _ = p.parse_known_args(argv)
    return known.profile


def _confirm_degraded(why):
    """`--pose-vlm auto` probed and found no arbiter: ask, default No
    (docs/archive/tri-state-pass-2.md, 2026-08-21).

    A silent degrade is what made an `auto` run indistinguishable from an
    `off` one, and every pose it resolves is marked `arbitrated: false` — a
    whole run's escalations deferred on a gcloud failure nobody was told
    about.

    Two deliberate deviations from `cache_root`'s prompt, which is precedent
    for the shape and not for these: **stdin AND stderr** must be ttys, and
    the prompt is written to **stderr** — `input()` writes its own prompt to
    stdout, so a redirected-stdout run would appear to hang. A backgrounded
    job with a tty stdin still SIGTTIN-stops, the same hazard `cache_root`
    accepts."""
    print(f"pose VLM: gemini unavailable ({why})", file=sys.stderr)
    if not (sys.stdin.isatty() and sys.stderr.isatty()):
        raise SystemExit(
            "--pose-vlm auto found no arbiter and this run is not interactive. "
            "Re-run with --pose-vlm off to accept the ensemble's answers "
            "(gated poses are marked `arbitrated: false` and revisited by the "
            "first arbiter-on run), or fix gcloud: "
            "`gcloud auth application-default login`")
    print("continuing keeps the ensemble's answer for every ambiguous pose; "
          "each one is marked `arbitrated: false` and revisited by the first "
          "run with an arbiter.", file=sys.stderr)
    if input("continue without the arbiter? [y/N] ").strip().lower() not in ("y", "yes"):
        raise SystemExit("--pose-vlm auto: no arbiter, declined")
    return None


def _openrouter_ready():
    """Is there an OpenRouter key for `auto` to fall back to? Reading it is
    the probe — `pose.openrouter_key` raises for missing, unreadable and empty
    alike, and reading is what the backend does per call anyway."""
    from src import pose
    try:
        pose.openrouter_key()
        return True
    except pose.VLMUnavailable:
        return False


def resolve_pose_vlm(args):
    """--pose-vlm to the backend the Poser is built with, announcing the choice.

    `ollama` is retired (2026-08-17, C-R1-4): the Arbiter is a thread pool with
    no inline arm, and a pooled ollama call would overlap SigLIP on the 4060 —
    10.1 s of model reload against 0.49 s of inference, this repo's one hard
    GPU constraint. `VlmConfig` refuses the name at construction if it ever
    reaches it another way.

    `auto` degrades before it asks: gemini, then the OpenRouter arbiter if a
    key is readable, and only then the question (`_confirm_degraded`). The
    explicit choices are unchanged in shape — `gemini` and `glm` are errors if
    unavailable, `off` is silent."""
    from src import pose        # module-local: pose pulls open3d (docstring)
    backend = args.pose_vlm
    if backend == "off":
        return None
    if backend == "auto":
        # gemini or nothing: it is the only arbiter measured to beat the
        # ensemble (43/44 against 40/44), where haiku/sonnet on a 256px sheet
        # score below running no arbiter at all. It bills per call — $2.7 for
        # the 354 escalations a 602-model run fires, measured usage at list
        # prices (LEARNINGS 2026-08-30) — so the choice is announced.
        try:
            args.gemini_project = args.gemini_project or pose.gcloud_project()
            pose.gcloud_token()
            backend = "gemini"
        except Exception as e:
            # A measured second choice beats a prompt: GLM-5.3-Flash on solo
            # tiles rescues +3 -> 41/44 where gemini rescues +4 -> 42/44
            # (LEARNINGS, "Arbiter backends, sheet sizes, presentations and
            # effort", 2026-08-30), for $0.06 a run and no GPU. Degrading to
            # no arbiter at all — which defers a whole run's escalations — is
            # only the question to ask when there is no key either.
            if not _openrouter_ready():
                return _confirm_degraded(e)
            backend = "glm"
            # The backend degraded; a --pose-vlm-model pin chosen for gemini
            # does not, unless it is dropped here. OpenRouter ids are
            # org/model and a gemini id never is, so a surviving pin reaches
            # OpenRouter as an HTTP 400 — which `pose._ask_glm` must map to
            # `VLMRejected`, since a non-transient 4xx is the API judging the
            # request, and `poser.Poser._fold` records that permanently as
            # `arbitrated: "rejected"`. One per escalation for the whole run,
            # and unstoppable: each rejection *resets* the breaker
            # (adversarial review, 2026-08-31).
            pin = args.pose_vlm_model
            drop = bool(pin) and "/" not in pin
            if drop:
                args.pose_vlm_model = None
            print(f"pose VLM: gemini unavailable ({e}) — falling back to the "
                  f"OpenRouter arbiter"
                  + (f", ignoring --pose-vlm-model {pin} — it names a gemini "
                     f"model, and glm models are org/model ids" if drop else ""))
    vlm_model = args.pose_vlm_model or pose.DEFAULT_VLM_MODELS.get(backend)
    if backend == "gemini":
        # Fail here rather than on the first ambiguous model, thousands of
        # renders into a run: resolving the project and minting a token are the
        # two things that go wrong, and both are cheap to check up front.
        # Explicit --pose-vlm gemini is an error if unavailable; auto already
        # returned above and never reaches this.
        try:
            args.gemini_project = args.gemini_project or pose.gcloud_project()
            pose.gcloud_token()
        except Exception as e:
            raise SystemExit(f"--pose-vlm gemini: {e}")
        print(f"pose VLM: {vlm_model} on Vertex AI, project {args.gemini_project} "
              f"— billed per escalation")
    elif backend == "glm":
        # The foreign pin `auto` drops silently is an error when both halves
        # were typed: a gemini id sent to OpenRouter comes back HTTP 400, and
        # the retry contract has to read a non-transient 4xx as a verdict — so
        # the run would write a permanent `arbitrated: "rejected"` per
        # escalation rather than fail (adversarial review, 2026-08-31).
        if args.pose_vlm_model and "/" not in args.pose_vlm_model:
            raise SystemExit(
                f"--pose-vlm glm: --pose-vlm-model {args.pose_vlm_model} is not an "
                f"OpenRouter model id — those are org/model (e.g. {pose.GLM_MODEL}). "
                f"Drop the pin or give it an OpenRouter id")
        # Symmetric with the gemini arm: the one thing that goes wrong is the
        # key, and checking it costs nothing against finding out thousands of
        # renders into a run. `auto` reaches this having already probed —
        # cheap, and it keeps the announcement in one place.
        try:
            pose.openrouter_key()
        except pose.VLMUnavailable as e:
            raise SystemExit(f"--pose-vlm glm: {e}")
        print(f"pose VLM: {vlm_model} on OpenRouter via "
              f"{pose.openrouter_provider()} — billed per escalation")
    else:
        print(f"pose VLM: {vlm_model or backend}")
    # The arbiter sheet scales each tile to SHEET_THUMB, and Image.thumbnail
    # never enlarges — so tiles rendered smaller than that sit padded in their
    # cells and the arbiter sees a smaller sheet than the number implies. Worth
    # saying out loud: sheet size is the knob that moved sonnet 10 of 44.
    if args.render_size < pose.SHEET_THUMB:
        print(f"  note: --render-size {args.render_size} is below the {pose.SHEET_THUMB}px "
              f"arbiter tile, so the arbiter sees {args.render_size}px tiles, not "
              f"{pose.SHEET_THUMB}px ones — on the sheet backends they sit padded into "
              f"{pose.SHEET_THUMB}px cells")
    return backend


def main():
    # Both pull open3d/numpy (`pose` transitively, `renderer` directly), and
    # neither is wanted at module scope — see the docstring.
    from src import pose
    from src.renderer import RENDER_FORMATS

    parser = argparse.ArgumentParser()
    add_cache_args(parser, "STL file or directory of STL files "
                           "(defaults to the last run's directory)")
    parser.add_argument("--categories", default="categories.txt")
    parser.add_argument("--out", default="results.csv")
    parser.add_argument("--save-renders", action="store_true",
                        help="keep the render images for debugging, under "
                             "<cache-dir>/renders/<camera config>/ as "
                             "<stem>_<path hash>_view<i>.<ext>, plus <stem>_<path hash>"
                             "_pose.png for each model whose up axis the VLM had to "
                             "arbitrate (the hash keeps two models that share a "
                             "filename from overwriting each other)")
    parser.add_argument("--render-format", choices=sorted(RENDER_FORMATS), default="jpg",
                        help="encoding for --save-renders images (default jpg). Nothing "
                             "reads these back — the classifier embeds the in-memory "
                             "render — so lossy is safe here, and jpg encodes ~180x "
                             "faster and ~16x smaller than png at 2048 px")
    parser.add_argument("--reanchor", action="store_true",
                        help="accept a collection root that differs from the one this "
                             "cache was built against, re-keying every entry. Right after "
                             "the library moves, wrong when the cache belongs to another "
                             "collection")
    parser.add_argument("--pool", choices=["mean", "max", "softmax"], default="softmax",
                        help="how per-view scores combine: mean = whole-object consensus, "
                             "max = single-view features decide, softmax = in between")
    # the backends this CLI will accept — `ollama` is not among them (C-R1-4),
    # which is also what --pose-vlm-model's help below filters on: advertising
    # a default for a backend argparse rejects is worse than saying nothing
    pose_vlm_choices = ["auto", "claude", "gemini", "glm", "off"]
    parser.add_argument("--pose-vlm", choices=pose_vlm_choices,
                        default="auto",
                        help="arbiter for uncertain up detection: gemini on Vertex AI, "
                             "glm on OpenRouter, claude CLI, or off. auto (default) = "
                             "gemini if gcloud ADC resolves, else glm if an OpenRouter "
                             "key is readable, else ask. gemini-3.5-flash is the best "
                             "arbiter measured (43/44 standalone, +4 -> 42/44 as the "
                             "tier) and bills ~$2.7 per full-collection run (the "
                             "354 escalations a 602-model run fires); "
                             "GLM-5.3-Flash is the measured fallback (+3 -> 41/44) at "
                             "~1/40th the cost ($0.06) and no GPU. `ollama` is retired: "
                             "the arbiter is a thread pool with no inline arm, and a "
                             "pooled ollama call would share the 4060 with SigLIP")
    parser.add_argument("--pose-vlm-model", default=None,
                        help="model for --pose-vlm; defaults per backend "
                             f"({', '.join(f'{k}={v}' for k, v in pose.DEFAULT_VLM_MODELS.items() if v and k in pose_vlm_choices)})")
    parser.add_argument("--gemini-project", default=None,
                        help="GCP project for --pose-vlm gemini (default: "
                             "$GOOGLE_CLOUD_PROJECT or `gcloud config get-value project`)")
    parser.add_argument("--repose", action="store_true",
                        help="re-arbitrate cached poses whose recorded arbiter "
                             "differs from this run's. Provenance-selective: it "
                             "re-buys only entries a *different* backend/model "
                             "judged (and entries judged before provenance was "
                             "recorded), not geometry or ensemble answers — a run "
                             "that degraded to GLM (+3 -> 41/44) can be re-judged "
                             "by gemini (+4 -> 42/44) without re-posing the "
                             "collection. Flag-gated on purpose: doing this on "
                             "every run would make two backends re-buy each "
                             "other's entries forever whenever `auto` degrades. "
                             "Run it WITHOUT --skip-embed: a re-opened entry is "
                             "rewritten without its front_view, and only the "
                             "scoring path recomputes and merges that back "
                             "(S8). The judge's identity is backend/model and "
                             "deliberately not the serving provider — the same "
                             "model on a different provider is the same judge, "
                             "and no re-judgment is owed. "
                             "To re-pose everything instead, delete pose-cache.json")
    parser.add_argument("--embed-batch", type=int, default=0,
                        help="images per SigLIP call (0 = the whole view list at once). "
                             "Raise to keep the GPU busier on long lists; lower if "
                             "SigLIP has to share the card")
    parser.add_argument("--prefetch", type=int, default=2,
                        help="accepted and inert in v1: the render child loads each mesh "
                             "inline, and this becomes its loader_worker_count when the "
                             "child grows loader workers (actors_proposal.md migration "
                             "notes). Worth ~1-2%% of a run when it did apply")
    parser.add_argument("--arbiter-workers", type=int, default=4,
                        help="concurrent pose-VLM calls for network backends — the "
                             "Arbiter's window. The call averages 24s against 3-28s of "
                             "local work per model, so waiting inline leaves the run "
                             "mostly idle. Was 8 until 2026-08-19, when a collection-"
                             "scale run hit Vertex quota (HTTP 429)")
    parser.add_argument("--arbiter-min-interval", type=float, default=1.0,
                        help="minimum seconds between pose-VLM call *starts*. At 4 "
                             "workers and a ~24s call the healthy rate is ~10/min, so "
                             "this never binds on success — it exists because a 429 "
                             "returns in milliseconds, which frees a worker to fail "
                             "again immediately and turns a quota refusal into a "
                             "self-sustaining storm. 0 disables pacing")
    parser.add_argument("--up-margin", type=float, default=pose.MARGIN_THRESHOLD,
                        help="escalate to the pose VLM when the ensemble's winning "
                             "candidate leads the runner-up by less than this (0-2). "
                             "Lower = fewer VLM calls")
    parser.add_argument("--skip-embed", action="store_true",
                        help="skip embedding and scoring the classification views; pose "
                             "resolution, including the SigLIP up-ensemble, still runs")
    parser.add_argument("--instrument", nargs="?", const="instrument.json",
                        default=None, metavar="PATH",
                        help="record per-stage timings and CPU/NVIDIA/amdgpu "
                             "utilization to PATH (default instrument.json), and "
                             "print the breakdown at the end. Rendering runs on the "
                             "amd iGPU and embedding on the nvidia card, so both "
                             "are sampled. The render child times its own stages "
                             "(mesh-load, pose-render, view-render, save-renders) "
                             "and reports them as their own table — it is a separate "
                             "process, so its time overlaps the parent's rather than "
                             "adding to it")
    parser.add_argument("--profile", nargs="?", const=PROFILE_DIR,
                        default=None, metavar="DIR",
                        help=f"run the torch profiler over the whole run and write a "
                             f"tensorboard trace to DIR (default {PROFILE_DIR}), then "
                             f"print the top CPU-time table. Off by default: a trace is "
                             f"~280 MB. Parent process only — the render child is a "
                             f"separate process, so its work appears here as time spent "
                             f"waiting on the results queue, not as rendering. Read by "
                             f"`profile_dir()` before main() starts; declared here so it "
                             f"shows in --help")
    args = apply_run_params(parser)
    if not args.input:
        sys.exit("no input given, and no directory recorded in "
                 f"{Path(args.cache_dir or '.') / RUN_PARAMS_FILE}")

    if args.instrument:
        instrument.enable(args.instrument)

    inp = Path(args.input)
    # before cache_root: an unreadable cache should be one line of output,
    # not a re-anchor prompt followed by a refusal (S5)
    require_cache_version(args.cache_dir)
    root = cache_root(inp, args.cache_dir, reanchor=args.reanchor)
    # sticky, and only a directory run may set it: a loose file describes no
    # collection, and save_run_params drops None rather than overwriting
    args.collection_root = str(root) if inp.is_dir() else None
    with stage("walk"):
        files = load_file_list(inp, args.cache_dir, args.rescan) if inp.is_dir() else [inp]
    if not files:
        sys.exit(f"no STL files found under {inp}")
    n_views = total_views(args)
    print(f"{n_views} views per model: {args.views} azimuths at "
          f"{', '.join(f'{e:g}' for e in args.elevations)} degrees")
    categories = [l.strip() for l in open(args.categories) if l.strip()]

    # The manifest before the model load, not after the run: an OOM or a
    # kill -9 inside Embedder construction must still leave a cache that says
    # what its .npy files are keyed under. Written at exit it did not, and a
    # cache being filled by a live run — or by one that was killed — held real
    # entries under keys no reader could reconstruct, so every tool fell back
    # to parser defaults (--model, --views), missed every key, and reported a
    # half-built cache as unusable (hit live, 2026-08-31). Every
    # RUN_PARAMS_KEYS value is settled by argparse plus `collection_root`
    # above and none of them changes during the run, so this writes the bytes
    # the exit-time write did. Last before the run commits to work, and after
    # the checks above, so a typo'd empty directory or an unreadable
    # --categories still exits without overwriting a good manifest.
    save_run_params(args)

    # Every import below is deferred, and for one reason: src.done, src.embedder
    # and src.poser own torch, this module is re-imported by the spawned render
    # child (module docstring), and none of this runs there.
    # `cache_checker` owns no torch, and is deferred under the other half of
    # the same rule: this module's scope is the render child's startup cost
    # (CLAUDE.md's import-weight constraint), only `main` needs the seam's
    # name, and `src.driver` right below already imports it — so deferring
    # here removes the cost rather than moving it onto a caller.
    from src import cache_checker, driver
    from src.arbiter import Arbiter
    from src.done import Done
    from src.driver import Admission, DriverConfig
    from src.embedder import Embedder
    from src.messages import CacheContext, Failure, RenderConfig
    from src.poser import Poser, VlmConfig

    vlm_backend = resolve_pose_vlm(args)
    # --repose's comparison value, on the namespace because `route` reads the
    # run's flags through `CacheContext.args` and nothing else crosses — under
    # `cache_checker`'s own REPOSE_ARBITER_ATTR, so the writer and the reader
    # cannot drift apart. It must be built the same way the Poser builds its
    # stamp, which is why both go through `pose.arbiter_id` — and after
    # `resolve_pose_vlm`, which is where an `auto` degrade settles both the
    # backend and the model pin.
    if args.repose and vlm_backend is None:
        raise SystemExit(
            "--repose needs an arbiter — it exists to re-judge, and this run "
            "has none (--pose-vlm off, or auto found neither gcloud nor an "
            "OpenRouter key)")
    if args.repose and args.pose_vlm == "auto" and vlm_backend != "gemini":
        # A degraded `auto` is the one arbiter nobody typed. Re-judging the
        # better judge's work with the measured-worse backend has to be a
        # decision, not a morning when ADC happened to expire: gemini rescues
        # +4 -> 42/44 where the OpenRouter arbiter rescues +3 -> 41/44
        # (LEARNINGS 2026-08-30, the 44-model labelled subset), so an
        # auto-degraded --repose run would walk the collection *replacing*
        # gemini judgments with glm ones and report it as progress. Explicit
        # `--pose-vlm glm` is still allowed — then it is what was asked for
        # (adversarial review, 2026-08-31).
        raise SystemExit(
            f"--repose with --pose-vlm auto, but auto degraded to "
            f"{vlm_backend}: re-judging with the fallback would replace "
            f"gemini's judgments (+4 -> 42/44) with the measured-worse "
            f"backend's (+3 -> 41/44), for every entry gemini settled. Fix "
            f"the gemini credentials, or pass `--pose-vlm {vlm_backend}` "
            f"explicitly if re-judging with the fallback is what you mean")
    repose_arbiter = (pose.arbiter_id(vlm_backend, args.pose_vlm_model)
                      if args.repose else None)
    setattr(args, cache_checker.REPOSE_ARBITER_ATTR, repose_arbiter)
    if args.repose:
        print(f"--repose: re-arbitrating cached poses not judged by "
              f"{repose_arbiter}")
    print(f"loading {args.model} ...")
    with stage("model-load"):
        # the Embedder is the only owner of torch models: the fp16 load, the
        # --compile wrap on the image forward, the category text embeddings and
        # the four prompt banks all happen in here
        embedder = Embedder(categories, args.model,
                            compile_image_forward=args.compile,
                            embed_batch=args.embed_batch)
    if args.compile:
        print("torch.compile on the image forward; embeddings keyed as a "
              "separate cache regime")

    # the .npy files sit in their own subdirectory: they are the bulk of the
    # entries, and keeping them out of the cache root leaves pose-cache.json,
    # the walk lists and run-params.json legible in a listing
    edir = embeds_dir(args.cache_dir)
    if edir:
        edir.mkdir(parents=True, exist_ok=True)
    rdir = renders_dir(args.cache_dir, args) if args.save_renders else None
    # route()'s read-only world. `poses` is THE store Done owns from here on —
    # the same object, never a copy, so route sees this run's resolutions (I9).
    ctx = CacheContext(poses=pose.load_pose_cache(args.cache_dir), embeds_dir=edir,
                       render_index=render_index(rdir), args=args, root=root)

    # tasks unbounded, results bounded at the admission window (I2/Q1): the
    # parent never blocks on a send, and admission is the only forward
    # pressure. Constructed by the driver, not here — the wiring IS the
    # invariant, and a test pins it there (F-3)
    tasks, results = driver.make_transports()
    # ONE Admission per run (P2) — `admitted` is the driver's field, `retired`
    # is Done's, and the driver takes this very object back off Done rather
    # than being handed a second one.
    done = Done(Admission(), embedder.text_embeds, ctx, tasks,
                categories=categories, front_embeds=embedder.front_T,
                back_embeds=embedder.back_T)
    arbiter = Arbiter(workers=args.arbiter_workers, wrap=driver.instrumented,
                      min_interval=args.arbiter_min_interval)
    poser = Poser(embedder.up_T, embedder.down_T, arbiter, done.record_pose,
                  VlmConfig(backend=vlm_backend, model=args.pose_vlm_model,
                            scratch_dir=args.cache_dir or ".",
                            project=args.gemini_project,
                            margin_threshold=args.up_margin,
                            # keeps each escalation's contact sheet beside that
                            # model's renders; the claude backend's scratch
                            # copy is a unique temp name per call, unlinked
                            # after the CLI reads it (review, 2026-08-20)
                            sheet_path=(lambda f: rdir / f"{render_key(f, root)}_pose.png")
                                       if rdir else None))
    child = driver.spawn_render_child(tasks, results, RenderConfig(
        render_size=args.render_size, views=args.views,
        elevations=tuple(args.elevations), save_renders_dir=rdir,
        render_format=args.render_format, budget_bytes=RESIDENT_BUDGET_BYTES,
        collection_root=root,
        # the child times its own stages under --instrument and ships the
        # totals back on EndOfInput (F-7); without this the flag reports only
        # what the parent does, which since the refactor is mostly waiting
        instrument_path=args.instrument))
    driver.run(DriverConfig(
        # the bar advances on admission, so it runs at most WINDOW files
        # ahead of what has actually retired
        walker=tqdm(files, desc="classifying"), ctx=ctx,
        tasks=tasks, results=results, child=child, poser=poser,
        embedder=embedder, done=done, arbiter=arbiter,
        skip_embed=args.skip_embed))
    errors = sum(1 for r in done.rows.values() if isinstance(r, Failure))
    print(f"wrote {args.out} ({len(done.rows)} rows"
          + (f", {errors} of them render errors" if errors else "") + ")")
    # The run's product is reportability: a breaker-tripped or arbiterless run
    # must not read like a healthy one in the log tail
    # (docs/archive/tri-state-pass-2.md, 2026-08-21).
    if poser.gate_fired_no_call:
        print(f"{poser.gate_fired_no_call} ambiguous poses kept the ensemble's "
              f"answer with no arbiter call, marked `arbitrated: false` — the "
              f"first run with an arbiter revisits them"
              + (" (run that one WITHOUT --skip-embed: re-resolution wipes "
                 "front_view and only the scoring path recomputes it)"
                 if args.skip_embed else ""))
    instrument.report()


if __name__ == "__main__":
    trace_dir = profile_dir()
    if not trace_dir:
        main()
    else:
        # torch stays inside this block. `spawn` gives the render child
        # `__mp_main__`, not `__main__`, so nothing here runs there — but a
        # module-scope import would still land in the child and break the
        # import rule (module docstring).
        import torch
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
            record_shapes=True
        ) as prof:
            main()
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
