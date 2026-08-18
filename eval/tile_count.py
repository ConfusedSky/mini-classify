"""Can the pose ensemble's tile count be cut without losing accuracy?

  python tile_count.py                     # 4, 2, 1 azimuths
  python tile_count.py --az 4,2 --verify 0

`pose-embed` — SigLIP over 6 up candidates x `UP_TILE_AZIMUTHS` azimuths — is
the largest GPU line item in the run (29-42% of wall), and the work is exactly
proportional to the tile count: 24 -> 12 -> 6. This scores the ensemble at
`UP_TILE_AZIMUTHS` 4 (production), 2 and 1 against the hand-labelled up axes,
with probes, geometry and the min-max combination frozen — only how many
azimuths are averaged into the SigLIP vote changes.

Why the cached 24-tile grid can be subsetted rather than re-rendered:
`Renderer.pose_tiles` takes its cameras from `view_angles(n_az, [20.0])`, i.e.
azimuths 2*pi*i/n_az at one fixed elevation, and nothing else in that method
depends on n_az. So n_az=2 asks for {0, pi} and n_az=1 for {0}, both exact
subsets of n_az=4's {0, pi/2, pi, 3pi/2} — columns (0, 2) and (0). This is
*checked* against `src.renderer.view_angles` at run time rather than assumed
(`azimuth_columns`), and a non-subset request aborts instead of quietly scoring
the wrong pixels.

The `n_az` parameter on `Renderer.pose_tiles` exists for this harness: the tile
count is a measured parameter, so the sweep has to sweep the production call
rather than a copy of it. Production always passes `UP_TILE_AZIMUTHS`.

`--source` chooses the pixels. `orbit` reuses `common.build_orbit_tiles`
(already cached for all 49 labels, and the pixels every published azimuth number
in LEARNINGS was measured on), which rotates the mesh per candidate — the
`Renderer.views` path — where the pose tiles carry the camera back through R.T.
`production` renders through `Renderer.pose_tiles` itself — the real code path,
the pixels the render child produces — and needs the STLs reachable. The two
are *not* interchangeable: measured here they agree to the noise floor only for
the +Z candidate, where R is the identity.

`--verify N` re-renders N of the scored models through `Renderer.pose_tiles`
and reports the pixel delta and whether any pick or margin moves. Against
`--source orbit` that measures the path difference; against `--source production`
it measures the renderer's own noise floor, which every pixel diff has to be read
against.

Per eval/README: `orig` is tuned, `holdout` is honest, `hard` was picked for
being failure-prone — reported separately, never pooled into one figure.

What the 2026-08-13 run found, so the next reader knows which arm to trust:

- On production pixels (n=43, the reachable labels) `n_az=2` changes **no**
  ensemble pick at all — orig 18/20, holdout 17/20, hard 3/3 at both 4 and 2 —
  for half the `pose-embed` work. `n_az=1` costs 2 of 40 (`Propane_Tank`,
  `Mortimer_BodyNoMask`) and doubles arbiter firing, 6 → 12 of 40.
- The renderer is bit-exact: re-rendering the same model twice through
  `Renderer.pose_tiles` gives mean|dpx| 0.0000. So any pixel difference
  measured against it is real, not noise.
- The two render paths are **not** pixel-identical: rendered fresh, camera-carry
  against rotate-the-mesh is ~1.9 mean|dpx| on both models tried — the finding
  that later became I11 (`eval/views_camera_rotation.py`) and moved the view
  path onto a rotated copy. The `+Z` candidate is
  the exception, where R is the identity and the two paths agree exactly.
- The cached `orbit384x4` tiles are additionally **stale** for 39 of the 43
  reachable models — 35 by a uniform ~0.45 mean|dpx| (a shading constant moved
  under them) and 4 by 9-13 (`tile9`, `Concrete Chunk (6)`, `Body`,
  `Pressurized_Container_2`). Every published azimuth number sits on those
  pixels. `--compare` reprints this.
- Picks are robust to all of that (0-3 of 43 differ between sources), but gate
  crossings are not: 2-7 of 43 land on the other side of `MARGIN_THRESHOLD`
  depending only on which pixel source fed them. Read any escalation count here
  with that error bar.
"""
import argparse
import json
import time

import numpy as np

from common import (AX, ORBIT_N_AZ, OUT, build_orbit_tiles,  # puts REPO on sys.path
                    build_tiles, load_labels)

from src import pose

BACKBONE = "google/siglip2-so400m-patch14-384"   # what production runs
RENDER_PX = 384                                  # what run_classify.sh renders at
GRID_AZ = 4                                      # azimuths in the cached grid


def azimuth_columns(n_az, grid_az=GRID_AZ):
    """Columns of a `grid_az`-azimuth grid that reproduce an n_az render, or None.

    Asks `src.renderer.view_angles` — the function `Renderer.pose_tiles` itself
    calls — for both camera lists and matches them, so this stays correct if the
    angle scheme or the tile elevation ever changes. Returning None means the
    angles are not a subset and the caller must render that n_az itself rather
    than slicing.
    """
    from src.renderer import UP_TILE_ELEVATION, view_angles
    have = view_angles(grid_az, [UP_TILE_ELEVATION])
    want = view_angles(n_az, [UP_TILE_ELEVATION])
    cols = []
    for az, elev in want:
        hit = [i for i, (a, e) in enumerate(have)
               if np.isclose(a, az) and np.isclose(e, elev)]
        if not hit:
            return None
        cols.append(hit[0])
    return cols


def render_production_grids(labels, render_px=RENDER_PX, out_dir=None):
    """`Renderer.pose_tiles` itself, cached: {stem: [6][GRID_AZ] Path}.

    The production call, so these are the pixels the pipeline actually embeds.
    Unreachable STLs are dropped and returned as the second element rather than
    faked — the labels carry an absolute collection root and the library moves.
    """
    from PIL import Image
    import rig
    d = out_dir or (OUT / f"upgrid{render_px}x{GRID_AZ}")
    d.mkdir(parents=True, exist_ok=True)
    paths = {l["stem"]: [[d / f"{l['stem']}_u{u}a{k}.png" for k in range(GRID_AZ)]
                         for u in range(6)] for l in labels}
    missing = [l["stem"] for l in labels if not l["path"].exists()]
    todo = [l for l in labels if l["stem"] not in missing
            and any(not p.exists() for row in paths[l["stem"]] for p in row)]
    if todo:
        print(f"rendering {len(todo)} models x {6 * GRID_AZ} tiles at {render_px}px "
              f"through Renderer.pose_tiles -> {d}")
        r = rig.rig(render_px)
        for n, l in enumerate(todo, 1):
            grid = rig.pose_tiles(r, rig.load(l["path"]), n_az=GRID_AZ)
            for u, row in enumerate(grid):
                for k, im in enumerate(row):
                    Image.fromarray(im).save(paths[l["stem"]][u][k])
            print(f"  [{n}/{len(todo)}] {l['stem']}", flush=True)
    ok = [l for l in labels
          if all(p.exists() for row in paths[l["stem"]] for p in row)]
    return {l["stem"]: paths[l["stem"]] for l in ok}, missing


def upright_grid(labels, verify=0, source="orbit"):
    """{stem: {"up": (6, GRID_AZ) upright scores, "geo": (6,)}} plus a render check.

    `source` is "orbit" (the cached rotate-the-mesh tiles) or "production"
    (`Renderer.pose_tiles`, which needs the meshes and drops unreachable
    ones — hence the filtered label list in the return).

    Render phase first, embed phase second — they evict each other on an 8 GB
    card (eval/README). `verify` re-renders that many models through the
    production grid call and embeds those pixels too; against source="production"
    that is a same-path re-render, i.e. the renderer's noise floor.

    Returns (scores, verify_rows, seconds_per_model, labels_actually_scored).
    """
    import torch
    from PIL import Image
    import rig

    if ORBIT_N_AZ != GRID_AZ:
        raise SystemExit(f"common.ORBIT_N_AZ is {ORBIT_N_AZ}, expected {GRID_AZ}")

    if source == "production":
        paths, gone = render_production_grids(labels)
        if gone:
            print(f"note: {len(gone)} of {len(labels)} labelled STLs are not "
                  f"reachable and are dropped from this run: {', '.join(gone)}")
        labels = [l for l in labels if l["stem"] in paths]
    else:
        paths = build_orbit_tiles(labels, RENDER_PX)      # cached; 24 per model
    geo_src = build_tiles(labels, RENDER_PX)              # geometry only; cached
    # geo comes off the mesh, not the pixels: tiles384/geo.json and
    # tiles2048/geo.json are byte-identical, so the render size here is free.

    # Only models whose STL is reachable can be re-rendered. The labels record an
    # absolute collection_root and the library moves between mounts (identity.py
    # exists for exactly that), so a missing file is a moved drive, not a bad label.
    check = [l for l in labels if l["path"].exists()][:verify]
    if verify and not check:
        print(f"verify: skipped — no labelled STL is reachable under "
              f"{labels[0]['path'].parents[-2] if labels else '?'} (pass --root)")
    prod = {}
    if check:
        d = OUT / "tile_count_verify"
        d.mkdir(parents=True, exist_ok=True)
        print(f"verify: re-rendering {len(check)} models through "
              f"Renderer.pose_tiles at {RENDER_PX}px")
        r = rig.rig(RENDER_PX)
        for l in check:
            grid = rig.pose_tiles(r, rig.load(l["path"]), n_az=GRID_AZ)
            prod[l["stem"]] = [[d / f"{l['stem']}_u{u}a{k}.png" for k in range(GRID_AZ)]
                               for u in range(6)]
            for u, row in enumerate(grid):
                for k, im in enumerate(row):
                    Image.fromarray(im).save(prod[l["stem"]][u][k])

    t0 = time.time()
    e = rig.embedder(BACKBONE)
    print(f"{BACKBONE} on {e.device}, loaded in {time.time()-t0:.0f}s")

    def scores(flat):
        imgs = [Image.open(p).convert("RGB") for p in flat]
        return pose.upright_scores(rig.embed(e, imgs),
                                   e.up_T, e.down_T).reshape(6, GRID_AZ)

    out, t0 = {}, time.time()
    for n, l in enumerate(labels, 1):
        s = l["stem"]
        out[s] = {"up": scores([p for row in paths[s] for p in row]),
                  "geo": np.asarray(geo_src[s]["geo"])}
        if n % 10 == 0 or n == len(labels):
            print(f"  embedded {n}/{len(labels)}", flush=True)
    per_model = (time.time() - t0) / max(len(labels), 1)
    print(f"  {GRID_AZ * 6} tiles/model: {per_model:.2f}s per model "
          f"({per_model / (GRID_AZ * 6) * 1000:.0f} ms per tile)")

    ver = []
    for l in check:
        s = l["stem"]
        a = np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)
                      for row in paths[s] for p in row])
        b = np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)
                      for row in prod[s] for p in row])
        g_scored, g_prod = out[s]["up"], scores([p for row in prod[s] for p in row])
        rows = []
        for n_az in (GRID_AZ, 2, 1):
            cols = azimuth_columns(n_az)
            rows.append((n_az,
                         pose.combine_up(out[s]["geo"], g_scored[:, cols].mean(1)),
                         pose.combine_up(out[s]["geo"], g_prod[:, cols].mean(1))))
        # Per-candidate pixel delta, because the +Z candidate is the one where R is
        # the identity and the two render paths are the same cameras — its row is
        # the noise floor the other five have to be read against.
        ver.append({"stem": s, "source": source,
                    "mean_abs_px": float(np.abs(a - b).mean()),
                    "max_abs_px": int(np.abs(a - b).max()),
                    "identical": bool((a == b).all()),
                    "mean_abs_px_per_candidate":
                        [float(np.abs(a[u * GRID_AZ:(u + 1) * GRID_AZ]
                                      - b[u * GRID_AZ:(u + 1) * GRID_AZ]).mean())
                         for u in range(6)],
                    "max_score_delta": float(np.abs(g_scored - g_prod).max()),
                    "picks": [{"n_az": n, "scored": [AX[o[0]], round(o[1], 4)],
                               "reference": [AX[p[0]], round(p[1], 4)]}
                              for n, o, p in rows]})

    del e
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out, ver, per_model, labels


def compare_sources():
    """Read both source runs and report how much the pixel path moved the answer.

    Restricted to the models both runs scored, because `--source production` drops
    whatever the mount cannot reach and comparing across different sets would
    measure the sets. Reports three things: accuracy per set under each source,
    how many picks and gate crossings differ, and — from the +Z column, where both
    render paths use the identical camera — how much of the orbit cache is stale.
    """
    from PIL import Image
    a = json.loads((OUT / "tile_count_orbit.json").read_text())
    b = json.loads((OUT / "tile_count_production.json").read_text())
    lab = b["labels"]
    stems = [s for s in lab if s in a["labels"]]
    az = [n for n in a["azimuths"] if n in b["azimuths"]]
    sets = [(k, [s for s in stems if lab[s]["set"] == k]) for k in ("orig", "holdout")]
    sets.append(("orig+hold", [s for s in stems if lab[s]["set"] in ("orig", "holdout")]))
    sets.append(("hard", [s for s in stems if lab[s]["set"] == "hard"]))

    print(f"ensemble accuracy on the {len(stems)} models both sources scored")
    print(f"{'set':16}" + "".join(f"{src + ' n_az=' + n:>18}"
                                 for src in ("orbit", "production") for n in az))
    for name, sel in [(n, s) for n, s in sets if s]:
        row = [f"{sum(d['azimuths'][n][s]['ens'] == lab[s]['gold'] for s in sel)}/{len(sel)}"
               for d in (a, b) for n in az]
        print(f"{name + f' (n={len(sel)})':16}" + "".join(f"{c:>18}" for c in row))

    print("\nwhere the two pixel sources disagree")
    for n in az:
        pick = [s for s in stems if a["azimuths"][n][s]["ens"] != b["azimuths"][n][s]["ens"]]
        esc = [s for s in stems
               if a["azimuths"][n][s]["escalates"] != b["azimuths"][n][s]["escalates"]]
        print(f"  n_az={n}: pick differs on {len(pick)}/{len(stems)}"
              + (f" — {', '.join(s[:30] for s in pick)}" if pick else ""))
        print(f"          gate crossing differs on {len(esc)}/{len(stems)}"
              + (f" — {', '.join(s[:30] for s in esc)}" if esc else ""))

    # +Z is candidate 0, where rotation_to_z_up is the identity and both paths
    # compute the same cameras — so a difference there is not the path.
    o, p = OUT / f"orbit{RENDER_PX}x{GRID_AZ}", OUT / f"upgrid{RENDER_PX}x{GRID_AZ}"
    arr = lambda f: np.asarray(Image.open(f).convert("RGB"), dtype=np.int16)
    stale = []
    for s in stems:
        d = np.mean([np.abs(arr(o / f"{s}_u0a{k}.png") - arr(p / f"{s}_u0a{k}.png")).mean()
                     for k in range(GRID_AZ)])
        if d > 0.01:
            stale.append((s, float(d)))
    print(f"\norbit cache staleness, from the +Z column (identical cameras): "
          f"{len(stale)} of {len(stems)} differ")
    for s, d in sorted(stale, key=lambda t: -t[1])[:8]:
        print(f"  [{lab[s]['set']:7}] {s[:44]:44} mean|dpx| {d:6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--az", default="4,2,1", help="UP_TILE_AZIMUTHS values to score")
    ap.add_argument("--verify", type=int, default=3,
                    help="models to re-render through the production grid call")
    ap.add_argument("--source", default="orbit", choices=("orbit", "production"),
                    help="which pixels to score: the cached rotate-the-mesh orbit "
                         "tiles every published azimuth number was measured on, or "
                         "Renderer.pose_tiles itself (needs the STLs)")
    ap.add_argument("--out", default=None, help="results filename in eval/out/")
    ap.add_argument("--compare", action="store_true",
                    help="don't score anything: read both source runs' JSON and "
                         "report how far the pixel path moved the answer")
    ap.add_argument("--root", default=None,
                    help="collection root to read STLs from, when the library has "
                         "moved off the mount recorded in up_axis_labels.json. "
                         "--source orbit only touches meshes for --verify; the cached "
                         "tiles and geometry are keyed by stem and need no root.")
    args = ap.parse_args()
    if args.compare:
        return compare_sources()
    az_list = [int(a) for a in args.az.split(",")]

    labels = load_labels()
    if args.root:
        from pathlib import Path
        from common import collection_root
        old, new = collection_root(), Path(args.root)
        for l in labels:
            l["path"] = new / l["path"].relative_to(old)
    gold = {l["stem"]: l["gold"] for l in labels}
    which = {l["stem"]: l["set"] for l in labels}

    cols = {n: azimuth_columns(n) for n in az_list}
    for n, c in cols.items():
        if c is None:
            raise SystemExit(f"n_az={n} is not a subset of the {GRID_AZ}-azimuth "
                             f"grid — it has to be rendered, not sliced")
        print(f"  n_az={n} -> grid columns {c}")

    grids, ver, per_model, labels = upright_grid(labels, args.verify, args.source)
    # Print the composition *after* the render phase — with --source production a
    # missing mount silently shrinks the set, and a bare "n=44" is how a number
    # outlives the set it was measured on.
    order = [l["stem"] for l in labels]
    counts = {s: sum(1 for l in labels if l["set"] == s)
              for s in dict.fromkeys(l["set"] for l in labels)}
    print(f"\nsource {args.source} — labels: "
          + ", ".join(f"{k} {v}" for k, v in counts.items()) + f" (total {len(order)})")
    for v in ver:
        print(f"verify {v['stem'][:40]:40} mean|dpx| {v['mean_abs_px']:.4f} "
              f"max {v['max_abs_px']:3d}  max score delta {v['max_score_delta']:.5f}")
        print("    mean|dpx| per candidate " + " ".join(
            f"{AX[u]} {d:.3f}" for u, d in enumerate(v["mean_abs_px_per_candidate"])))
        for p in v["picks"]:
            same = "same" if p["scored"][0] == p["reference"][0] else "DIFFERENT"
            print(f"    n_az={p['n_az']}: scored {p['scored'][0]} m={p['scored'][1]:.3f}"
                  f"  re-render {p['reference'][0]} m={p['reference'][1]:.3f}  {same}")

    # score every azimuth count off the same embeddings
    res = {}
    for n in az_list:
        per = {}
        for s in order:
            sig = grids[s]["up"][:, cols[n]].mean(1)
            idx, margin = pose.combine_up(grids[s]["geo"], sig)   # the real combination
            per[s] = {"ens": idx, "margin": margin, "sig": int(sig.argmax())}
        res[n] = per

    geo = {s: int(pose.rank_up_scores(grids[s]["geo"])[0]) for s in order}
    sets = [(k, [s for s in order if which[s] == k]) for k in ("orig", "holdout", "hard")]
    sets = [(n, sel) for n, sel in sets if sel]
    sets.insert(2, ("orig+hold", [s for s in order if which[s] in ("orig", "holdout")]))

    def acc(pick, sel):
        return f"{sum(pick[s] == gold[s] for s in sel)}/{len(sel)}"

    print("\ngeometry alone (azimuth-independent): "
          + "  ".join(f"{k} {acc(geo, sel)}" for k, sel in sets))
    for key, label in (("sig", "SigLIP alone"), ("ens", "ensemble")):
        print(f"\n{label} — accuracy by UP_TILE_AZIMUTHS")
        print(f"{'set':14}" + "".join(f"{'n_az=' + str(n):>12}" for n in az_list)
              + "     tiles/model")
        for name, sel in sets:
            row = [acc({s: res[n][s][key] for s in order}, sel) for n in az_list]
            print(f"{name + f' (n={len(sel)})':14}" + "".join(f"{c:>12}" for c in row))
        print(f"{'tiles/model':14}" + "".join(f"{6 * n:>12}" for n in az_list))

    base = az_list[0]
    print(f"\nwhich models flip, against n_az={base}")
    for n in az_list[1:]:
        flips = [s for s in order if res[n][s]["ens"] != res[base][s]["ens"]]
        if not flips:
            print(f"  n_az={n}: no ensemble pick changes")
        for s in flips:
            b, c = res[base][s]["ens"], res[n][s]["ens"]
            verdict = ("fixed" if c == gold[s] else
                       "broken" if b == gold[s] else "wrong->wrong")
            print(f"  n_az={n} [{which[s]:7}] {s[:40]:40} {AX[b]} -> {AX[c]} "
                  f"(gold {AX[gold[s]]}) {verdict}")

    print(f"\nensemble margin distribution (gate is margin < {pose.MARGIN_THRESHOLD})")
    print(f"{'set':14}{'n_az':>6}{'median':>8}{'p25':>8}{'p75':>8}"
          f"{'med|right':>11}{'med|wrong':>11}{'fires':>7}")
    for name, sel in sets:
        for n in az_list:
            m = np.array([res[n][s]["margin"] for s in sel])
            r = np.array([res[n][s]["ens"] == gold[s] for s in sel])
            med = lambda v: f"{np.median(v):.3f}" if len(v) else "--"
            fires = int((m < pose.MARGIN_THRESHOLD).sum())
            print(f"{name if n == az_list[0] else '':14}{n:>6}{np.median(m):>8.3f}"
                  f"{np.percentile(m, 25):>8.3f}{np.percentile(m, 75):>8.3f}"
                  f"{med(m[r]):>11}{med(m[~r]):>11}{f'{fires}/{len(sel)}':>7}")

    print(f"\nescalation changes at MARGIN_THRESHOLD={pose.MARGIN_THRESHOLD} "
          f"(each firing is one paid arbiter call)")
    fired = {n: {s for s in order if res[n][s]["margin"] < pose.MARGIN_THRESHOLD}
             for n in az_list}
    for n in az_list:
        by_set = "  ".join(f"{k} {len(fired[n] & set(sel))}/{len(sel)}" for k, sel in sets)
        print(f"  n_az={n}: {len(fired[n])}/{len(order)} fire — {by_set}")
    for n in az_list[1:]:
        for s in sorted(fired[n] - fired[base]):
            ok = "ensemble already right" if res[n][s]["ens"] == gold[s] else "ensemble wrong"
            print(f"  n_az={n} STARTS escalating [{which[s]:7}] {s[:40]:40} "
                  f"margin {res[base][s]['margin']:.3f} -> {res[n][s]['margin']:.3f} ({ok})")
        for s in sorted(fired[base] - fired[n]):
            ok = "ensemble right" if res[n][s]["ens"] == gold[s] else "ENSEMBLE WRONG, now unchecked"
            print(f"  n_az={n} STOPS  escalating [{which[s]:7}] {s[:40]:40} "
                  f"margin {res[base][s]['margin']:.3f} -> {res[n][s]['margin']:.3f} ({ok})")

    # An escalation change is only a cost or a saving once the arbiter answers, so
    # replay the recorded VLM runs through the production gate. No API access —
    # arbiter_gate.load_arbiters reads answers already on disk.
    from arbiter_gate import load_arbiters
    arbiters = {a: p for a, p in load_arbiters(order).items()
                if all(s in p for s in order)}      # full coverage only
    pipeline = {}
    print(f"\npipeline accuracy at the production gate (margin < {pose.MARGIN_THRESHOLD}), "
          f"replaying recorded arbiter answers — correct/calls")
    print(f"{'arbiter':32}" + "".join(f"{k + ' n_az':>16}" for k, _ in sets))
    print(f"{'':32}" + "".join("".join(f"{n:>5}" for n in az_list).rjust(16)
                               for _ in sets))
    for a in sorted(arbiters):
        cells, rec = [], {}
        for name, sel in sets:
            row = []
            for n in az_list:
                ok = calls = 0
                for s in sel:
                    v = res[n][s]["ens"]
                    if res[n][s]["margin"] < pose.MARGIN_THRESHOLD:
                        calls += 1
                        v = arbiters[a][s]
                    ok += v == gold[s]
                row.append(f"{ok}/{calls}")
                rec[f"{name}@{n}"] = {"correct": ok, "n": len(sel), "calls": calls}
            cells.append("".join(f"{c:>5}" for c in row).rjust(16))
        pipeline[a] = rec
        print(f"{a[:32]:32}" + "".join(cells))
    print("  (denominator is the call count, not n — set sizes are in the table above)")

    print("\npose-embed GPU work (proportional to tile count)")
    for n in az_list:
        print(f"  n_az={n}: {6 * n:>2} tiles/model, {6 * n / (6 * base):.2f}x of n_az={base}"
              f"  (measured {per_model * n / base:.2f}s/model at this backbone)")

    out = OUT / (args.out or f"tile_count_{args.source}.json")
    json.dump({"backbone": BACKBONE, "render_px": RENDER_PX, "source": args.source,
               "margin_threshold": pose.MARGIN_THRESHOLD,
               "labels": {s: {"set": which[s], "gold": AX[gold[s]]} for s in order},
               "geometry": {s: AX[geo[s]] for s in order},
               "grid_columns": {str(n): cols[n] for n in az_list},
               "embed_seconds_per_model_24_tiles": per_model,
               "verify": ver, "pipeline": pipeline,
               "azimuths": {str(n): {s: {"ens": AX[p["ens"]], "sig": AX[p["sig"]],
                                         "margin": round(p["margin"], 5),
                                         "escalates": p["margin"] < pose.MARGIN_THRESHOLD}
                                     for s, p in res[n].items()} for n in az_list}},
              open(out, "w"), indent=1)
    print(f"\nwrote {out}")
    import rig                      # a renderer is live; teardown would abort
    rig.exit_without_teardown()


if __name__ == "__main__":
    main()
