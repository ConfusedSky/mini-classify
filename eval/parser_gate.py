"""Does the numpy STL parser change pose decisions on the labelled set?

A binary STL is a triangle soup and `read_triangle_mesh` welds a handful of its
vertices where the numpy parser does not, so the two produce different vertex
normals and therefore different pixels — measured at 4.5% of pixels above 2/255
against a 0.004% noise floor (docs/actor-refactor/renderer_alternatives.md). That is well
above jitter, so it has to be scored against labels before it ships.

Each arm runs the production pose path with the arbiter off, changing *only*
the loader:

    geometry scores -> 6 x UP_TILE_AZIMUTHS candidate tiles -> SigLIP -> combine_up

The three tiers are composed here, out of the same `src/pose.py` functions the
Poser calls (`up_axis_scores` -> `rank_up_scores` -> `combine_up` ->
`needs_arbiter_margin`) over the same `Renderer.pose_tiles` pixels the render
child produces and the same `Embedder` forward. That is fifteen lines of
composition rather than a call into `src/poser.py`, because the Poser is
message-shaped — it consumes `TileEmbeds`, parks files on in-flight arbiter
calls and writes a cache — and none of that is under test here. What *is*
under test is the loader, so everything downstream of it is production's.

This used to call `classify_stls.resolve_up`, a single-process re-arrangement
of the same three tiers kept alive in the CLI for this one caller. Two things
went with it: geometry's ratio gate (`pose.needs_arbiter`, reached only when
no SigLIP scorer is supplied — which this harness never does, so it was dead
code behind a dead `--up-conf`), and the inline arbiter tier, which this
harness switched off anyway.

Two modes run per invocation over the same models:

    A/A   the same loader in both arms — the noise floor. One shared
          OffscreenRenderer carries scene state from render to render
          (LEARNINGS.md:505-524), so *zero* variables changed still moves
          margins: a reviewer measured 0.0169 mean / 0.0411 max that way,
          larger than the parser's own 0.0095. No margin-level claim here is
          attributable without this number printed beside it.
    A/B   `read_triangle_mesh` against the numpy parser — the variable on test.

Arm order alternates per model in both modes, so scene-state bias lands on both
arms equally instead of systematically on the arm that always rendered second.

No other harness gates this: `tile_count.py` sweeps azimuths and
`backbone_sweep.py` sweeps towers, neither of which the parser touches, and
both score cached pixels rather than re-parsing the meshes.

One OffscreenRenderer per process, so pass the size rather than sweeping it:

    .venv/bin/python eval/parser_gate.py [render_px]   # default 384, production

The labels record the collection root they were written against; when that root
has moved, re-root without editing the labels:

    COLLECTION_ROOT=/run/media/masa/STLLibrary .venv/bin/python eval/parser_gate.py
"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d

from common import AX, OUT, collection_root, load_labels   # puts REPO on sys.path

import rig
from src import pose
from src.loader import load_mesh

PX = int(sys.argv[1]) if len(sys.argv) > 1 else 384
MID = "google/siglip2-so400m-patch14-384"
SETS = ("orig", "hard", "holdout")   # reported separately, never pooled

# The one threshold the pose path still reads: the ensemble's margin gate.
# Production's default, so the gate moves when production does.
ARGS = SimpleNamespace(up_margin=pose.MARGIN_THRESHOLD)


def old_reader(path):
    """What load_mesh did before the swap."""
    m = o3d.io.read_triangle_mesh(str(path))
    m.compute_vertex_normals()
    return m


MODES = (("A/A", (load_mesh, load_mesh)),    # control: nothing changed
         ("A/B", (old_reader, load_mesh)))   # the loader, and nothing else


def current_root():
    """Where the collection lives now. up_axis_labels.json records the root the
    labels were written against; COLLECTION_ROOT overrides it when the drive has
    moved (same override idiom as EVAL_OUT), so a re-root needs no label edit."""
    return (Path(os.environ["COLLECTION_ROOT"]) if os.environ.get("COLLECTION_ROOT")
            else collection_root())


def labels_on_disk():
    """(present, missing) labelled models, paths under the current root.

    Only the root prefix is swapped — the path inside the collection is the
    label's. A mesh that is not there is its own condition, counted and printed:
    the old `except Exception: continue` scored it as a parse failure while
    still announcing the full label count.
    """
    old, new = collection_root(), current_root()
    labels = sorted(load_labels(), key=lambda l: l["stem"])
    for l in labels:
        l["path"] = new / l["path"].relative_to(old)
    return ([l for l in labels if l["path"].exists()],
            [l for l in labels if not l["path"].exists()])


def resolve(loader, path, renderer, score_upright):
    """The production pose path minus the arbiter — (geo_idx, ens_idx, margin).

    The two evidence tiers, in production's order and with production's
    functions: `pose.up_axis_scores` off the mesh the loader under test
    produced, `Renderer.pose_tiles` for the pixels, `Embedder` for the
    embeddings, `pose.combine_up` for the ensemble. Geometry's own pick falls
    out of the same score vector — `up_axis_scores` is seeded, so this is the
    same number the second call used to re-derive.
    """
    mesh = loader(path)
    geo_scores = pose.up_axis_scores(mesh)
    geo_idx, _ratio, _best = pose.rank_up_scores(geo_scores)
    grid = rig.pose_tiles(renderer, mesh)
    flat = [im for row in grid for im in row]
    sig = np.asarray(score_upright(flat)).reshape(len(grid), -1).mean(axis=1)
    idx, margin = pose.combine_up(geo_scores, sig)
    return geo_idx, idx, float(margin)


def run_mode(name, loaders, labels, renderer, score_upright):
    """One pass of both arms over every model. (rows, failures)."""
    print(f"\n{'=' * 78}\nmode {name}: arm a = {loaders[0].__name__}, "
          f"arm b = {loaders[1].__name__}\n{'=' * 78}")
    hdr = f"{'model':30} {'set':8} {'gold':>5} {'a':>6} {'b':>6} {'margin':>14} 1st"
    print(hdr + "\n" + "-" * len(hdr))

    rows, failed = [], []
    for i, l in enumerate(labels):
        # alternate which arm renders into the dirty scene first
        order = (0, 1) if i % 2 == 0 else (1, 0)
        arm = {}
        try:
            for k in order:
                arm[k] = resolve(loaders[k], l["path"], renderer, score_upright)
        except Exception as e:
            print(f"{l['stem'][:30]:30} {l['set']:8} ERROR {type(e).__name__}: {e}")
            failed.append(dict(stem=l["stem"], set=l["set"], error=f"{type(e).__name__}: {e}"))
            continue
        (g_a, e_a, m_a), (g_b, e_b, m_b) = arm[0], arm[1]
        gold = l["gold"]
        mark = lambda p: AX[p] + ("" if p == gold else "*")
        flag = "" if e_a == e_b else "   <-- DIFFERS"
        print(f"{l['stem'][:30]:30} {l['set']:8} {AX[gold]:>5} {mark(e_a):>6} "
              f"{mark(e_b):>6} {m_a:6.3f}->{m_b:6.3f} {'ab'[order[0]]}{flag}",
              flush=True)
        rows.append(dict(stem=l["stem"], set=l["set"], gold=gold, geo_a=g_a,
                         geo_b=g_b, a=e_a, b=e_b, m_a=m_a, m_b=m_b,
                         first="ab"[order[0]]))
    return rows, failed


def crossed(r):
    """Did this model cross the production escalation threshold between arms?"""
    return (pose.needs_arbiter_margin(r["m_a"], ARGS.up_margin)
            != pose.needs_arbiter_margin(r["m_b"], ARGS.up_margin))


def summarise(name, rows):
    """Print one mode's scorecard and return its headline numbers."""
    print(f"\n--- {name} over {len(rows)} scored models ---")
    for s in ("all",) + SETS:
        sel = rows if s == "all" else [r for r in rows if r["set"] == s]
        if not sel:
            continue
        n = len(sel)
        print(f"{s:9} n={n:<3} ensemble a {sum(r['a'] == r['gold'] for r in sel)}/{n}"
              f"   b {sum(r['b'] == r['gold'] for r in sel)}/{n}"
              f"   geometry a {sum(r['geo_a'] == r['gold'] for r in sel)}/{n}"
              f"   b {sum(r['geo_b'] == r['gold'] for r in sel)}/{n}")

    moved = [r for r in rows if r["a"] != r["b"]]
    geo_moved = [r for r in rows if r["geo_a"] != r["geo_b"]]
    dm = np.abs([r["m_b"] - r["m_a"] for r in rows])
    print(f"\nensemble pick changed on {len(moved)}/{len(rows)}"
          f"   geometry pick changed on {len(geo_moved)}/{len(rows)}")
    print(f"margin shift: mean {dm.mean():.4f}  max {dm.max():.4f}")

    # The margin gate decides what the VLM ever gets to see, so a crossing is
    # invisible in ensemble accuracy and reported on its own — per set, because
    # a pooled count hides which population moved.
    print(f"escalation at margin < {ARGS.up_margin} (per set, never pooled):")
    for s in SETS:
        sel = [r for r in rows if r["set"] == s]
        if not sel:
            continue
        esc = lambda k: sum(pose.needs_arbiter_margin(r[k], ARGS.up_margin) for r in sel)
        print(f"  {s:9} n={len(sel):<3} arm a {esc('m_a')}   arm b {esc('m_b')}"
              f"   crossings {sum(crossed(r) for r in sel)}")

    for r in moved:
        print(f"  pick moved: {r['stem']} ({r['set']}) gold {AX[r['gold']]} "
              f"{AX[r['a']]} -> {AX[r['b']]}")
    for r in rows:
        if crossed(r):
            was = pose.needs_arbiter_margin(r["m_a"], ARGS.up_margin)
            print(f"  escalation {'lost' if was else 'gained'}: {r['stem']} ({r['set']}) "
                  f"margin {r['m_a']:.3f} -> {r['m_b']:.3f}"
                  f"{'' if r['b'] == r['gold'] else '  (ensemble is WRONG here)'}")
    return dict(n=len(rows), mean=float(dm.mean()), max=float(dm.max()),
                moved=len(moved), geo_moved=len(geo_moved),
                crossings={s: sum(crossed(r) for r in rows if r["set"] == s)
                           for s in SETS})


def main():
    labels, missing = labels_on_disk()
    emb = rig.embedder(MID)

    def score_upright(tiles):   # the Poser's ensemble input, same banks
        return pose.upright_scores(rig.embed(emb, tiles), emb.up_T, emb.down_T)

    renderer = rig.rig(PX)

    print(f"{len(labels)} of {len(labels) + len(missing)} labelled models on disk "
          f"| {PX}px tiles | {MID}\ncollection root: {current_root()}")
    if missing:
        print(f"\nMISSING: {len(missing)} labelled meshes are not under that root "
              f"— not a parse failure, not scored:")
        for l in missing:
            print(f"  missing: {l['stem']} ({l['set']}) {l['path']}")

    out = {"px": PX, "backbone": MID, "labels": len(labels) + len(missing),
           "scored": len(labels), "up_margin": ARGS.up_margin,
           "missing": [dict(stem=l["stem"], set=l["set"], path=str(l["path"]))
                       for l in missing], "modes": {}}
    stats = {}
    for name, loaders in MODES:
        rows, failed = run_mode(name, loaders, labels, renderer, score_upright)
        stats[name] = summarise(name, rows)
        stats[name]["failed"] = len(failed)
        out["modes"][name] = {"arms": [f.__name__ for f in loaders],
                              "rows": rows, "failed": failed}

    aa, ab = stats["A/A"], stats["A/B"]
    print(f"\n{'=' * 78}\nparser effect against the noise floor\n{'=' * 78}")
    print(f"{'':14}{'A/A (floor)':>14}{'A/B (parser)':>14}")
    print(f"{'margin mean':14}{aa['mean']:>14.4f}{ab['mean']:>14.4f}")
    print(f"{'margin max':14}{aa['max']:>14.4f}{ab['max']:>14.4f}")
    print(f"{'picks moved':14}{str(aa['moved']) + '/' + str(aa['n']):>14}"
          f"{str(ab['moved']) + '/' + str(ab['n']):>14}")
    for s in SETS:
        print(f"{'cross ' + s:14}{aa['crossings'][s]:>14}{ab['crossings'][s]:>14}")
    if ab["mean"] > aa["max"]:
        verdict = "parser effect exceeds the A/A max — attributable"
    elif ab["mean"] > aa["mean"]:
        verdict = ("parser effect exceeds the A/A mean but not its max — "
                   "margin-level claims stay weak")
    else:
        verdict = ("parser effect is at or below the A/A noise floor — margin "
                   "shifts and threshold crossings are NOT attributable to the parser")
    print(f"\nverdict: {verdict}")
    out["stats"] = stats
    out["verdict"] = verdict

    path = OUT / f"parser_gate_{PX}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1, default=int)
    print(f"wrote {path}")
    rig.exit_without_teardown()   # the live OffscreenRenderer must not be destroyed


if __name__ == "__main__":
    main()
