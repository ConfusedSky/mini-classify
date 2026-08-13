"""Does the numpy STL parser change pose decisions on the labelled set?

A binary STL is a triangle soup and `read_triangle_mesh` welds a handful of its
vertices where the numpy parser does not, so the two produce different vertex
normals and therefore different pixels — measured at 4.5% of pixels above 2/255
against a 0.004% noise floor (docs/masa/renderer_alternatives.md). That is well
above jitter, so it has to be scored against labels before it ships.

This runs the production pose path twice per model, changing *only* the loader:

    geometry scores -> 6x4 candidate tiles -> SigLIP -> combine_up

and reports both parsers against gold. `eval/tile_and_vlm.py` is the wrong gate
for this change — it sweeps tile resolution and the ollama VLM tier, neither of
which the parser touches, and it needs a prior harness's results.json.

One OffscreenRenderer per process, so pass the size rather than sweeping it:

    .venv/bin/python eval/parser_gate.py [render_px]   # default 384, production
"""
import json
import sys

import numpy as np
import open3d as o3d
import torch

from common import AX, OUT, load_labels   # puts REPO on sys.path

import classify_stls as C
import pose

PX = int(sys.argv[1]) if len(sys.argv) > 1 else 384
MID = "google/siglip2-so400m-patch14-384"


def old_reader(path):
    """What load_mesh did before the swap."""
    m = o3d.io.read_triangle_mesh(str(path))
    m.compute_vertex_normals()
    return m


def resolve(mesh, renderer, model, proc, dev, upT, dnT):
    """resolve_up's ensemble, without the arbiter — (geo_idx, ens_idx, margin)."""
    geo = pose.up_axis_scores(mesh)
    grid = C.render_up_candidate_grid(renderer, mesh)
    flat = [im for row in grid for im in row]
    sc = pose.upright_scores(
        C.embed_images(model, proc, flat, dev).float().cpu().numpy(), upT, dnT)
    sig = np.asarray(sc).reshape(len(grid), -1).mean(axis=1)
    idx, margin = pose.combine_up(geo, sig)
    return pose.rank_up_scores(geo)[0], idx, margin


def main():
    from transformers import AutoModel, AutoProcessor

    labels = load_labels()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(MID, torch_dtype=torch.float16).to(dev).eval()
    proc = AutoProcessor.from_pretrained(MID)
    upT = C.embed_raw(model, proc, pose.UPRIGHT_PROMPTS, dev).float().cpu().numpy()
    dnT = C.embed_raw(model, proc, pose.TOPPLED_PROMPTS, dev).float().cpu().numpy()
    renderer = C.make_renderer(PX)

    print(f"{len(labels)} labelled models | {PX}px tiles | {MID}\n")
    hdr = f"{'model':30} {'set':8} {'gold':>5} {'old':>6} {'new':>6} {'margin':>14}"
    print(hdr + "\n" + "-" * len(hdr))

    rows = []
    for l in sorted(labels, key=lambda x: x["stem"]):
        try:
            g_old, e_old, m_old = resolve(old_reader(l["path"]), renderer,
                                          model, proc, dev, upT, dnT)
            g_new, e_new, m_new = resolve(C.load_mesh(l["path"]), renderer,
                                          model, proc, dev, upT, dnT)
        except Exception as e:
            print(f"{l['stem'][:30]:30} {l['set']:8} ERROR {e}")
            continue
        gold = l["gold"]
        mark = lambda p: AX[p] + ("" if p == gold else "*")
        flag = "" if e_old == e_new else "   <-- DIFFERS"
        print(f"{l['stem'][:30]:30} {l['set']:8} {AX[gold]:>5} {mark(e_old):>6} "
              f"{mark(e_new):>6} {m_old:6.3f}->{m_new:6.3f}{flag}", flush=True)
        rows.append(dict(stem=l["stem"], set=l["set"], gold=gold, geo_old=g_old,
                         geo_new=g_new, old=e_old, new=e_new,
                         m_old=float(m_old), m_new=float(m_new)))

    print()
    for name, sel in (("all", rows), ("orig", [r for r in rows if r["set"] == "orig"]),
                      ("holdout", [r for r in rows if r["set"] == "holdout"]),
                      ("hard", [r for r in rows if r["set"] == "hard"])):
        if not sel:
            continue
        n = len(sel)
        print(f"{name:9} n={n:<3} ensemble old {sum(r['old'] == r['gold'] for r in sel)}/{n}"
              f"   new {sum(r['new'] == r['gold'] for r in sel)}/{n}"
              f"   geometry old {sum(r['geo_old'] == r['gold'] for r in sel)}/{n}"
              f"   new {sum(r['geo_new'] == r['gold'] for r in sel)}/{n}")

    moved = [r for r in rows if r["old"] != r["new"]]
    geo_moved = [r for r in rows if r["geo_old"] != r["geo_new"]]
    dm = np.abs([r["m_new"] - r["m_old"] for r in rows])
    print(f"\nensemble pick changed on {len(moved)}/{len(rows)}"
          f"   geometry pick changed on {len(geo_moved)}/{len(rows)}")
    print(f"margin shift: mean {dm.mean():.4f}  max {dm.max():.4f}")
    esc = lambda k: sum(pose.needs_arbiter_margin(r[k]) for r in rows)
    print(f"would escalate to the arbiter: old {esc('m_old')}  new {esc('m_new')}")

    for r in moved:
        print(f"  pick moved: {r['stem']} gold {AX[r['gold']]} "
              f"{AX[r['old']]} -> {AX[r['new']]}")
    # models the margin gate stops (or starts) sending to the arbiter: invisible
    # in ensemble accuracy, but it changes what the VLM ever gets to see
    for r in rows:
        was, now = (pose.needs_arbiter_margin(r[k]) for k in ("m_old", "m_new"))
        if was != now:
            print(f"  escalation {'lost' if was else 'gained'}: {r['stem']} "
                  f"margin {r['m_old']:.3f} -> {r['m_new']:.3f}"
                  f"{'' if r['new'] == r['gold'] else '  (ensemble is WRONG here)'}")

    out = OUT / f"parser_gate_{PX}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out, "w"), indent=1, default=int)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
