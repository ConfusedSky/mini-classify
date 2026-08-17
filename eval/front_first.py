"""Does finding the front first make the up axis easier?

  python front_first.py                 # all labels
  python front_first.py --set hard

The pipeline today finds up, then names the front among the azimuths it already
rendered. This tests the reverse: score the *front* first, let that constrain
up, and see whether up accuracy improves.

Front and up are not independent — a (front, up) pair with front ⊥ up fixes the
orientation completely, and there are exactly 6 × 4 = 24 such pairs. Handily,
for a fixed up the four perpendicular fronts are just four azimuths after
`rotation_to_z_up`, so all 24 cost the same six geometry uploads as today's six
tiles (upload dominates: 3.3 s against ~0.1 s of pixels).

That makes the honest comparison a three-way one, because "front first" changes
two things at once — it adds front probes *and* it quadruples the views. The
`up-only, 4 views` rows are the control that separates them.
"""
import argparse, json

import numpy as np

from common import AX, OUT, build_tiles, load_labels  # puts REPO on sys.path

from src import pose

BACKBONE = "google/siglip2-so400m-patch14-384"
RENDER_PX = 384          # measured as good as 2048 for this, and ~3x faster to embed
N_AZ = 4                 # the four fronts perpendicular to each up
ELEV = 20.0


def front_axis_map():
    """(up_idx, az_idx) -> index of the mesh axis facing the camera."""
    from classify_stls import rotation_to_z_up
    m = np.zeros((6, N_AZ), dtype=int)
    for u, up in enumerate(pose.UP_CANDIDATES):
        R = rotation_to_z_up(up)
        for k in range(N_AZ):
            az = 2 * np.pi * k / N_AZ
            d = R.T @ np.array([np.cos(az), np.sin(az), 0.0])
            m[u, k] = int(np.argmax([d @ c for c in pose.UP_CANDIDATES]))
    return m


def build_orbit_tiles(labels, render_px):
    """24 tiles per model — 6 up candidates x 4 azimuths — cached on disk."""
    import classify_stls as C
    d = OUT / f"orbit{render_px}x{N_AZ}"
    d.mkdir(parents=True, exist_ok=True)
    paths = {l["stem"]: [[d / f"{l['stem']}_u{u}a{k}.png" for k in range(N_AZ)]
                         for u in range(6)] for l in labels}
    todo = [l for l in labels
            if any(not p.exists() for row in paths[l["stem"]] for p in row)]
    if todo:
        print(f"rendering {len(todo)} models x {6 * N_AZ} orientations at {render_px}px")
        renderer = C.make_renderer(render_px)
        import open3d as o3d
        for n, l in enumerate(todo, 1):
            mesh = C.load_mesh(l["path"])
            for u, up in enumerate(pose.UP_CANDIDATES):
                m = o3d.geometry.TriangleMesh(mesh)
                m.rotate(C.rotation_to_z_up(up), center=(0, 0, 0))
                imgs = C.render_views(renderer, m, C.view_angles(N_AZ, [ELEV]))
                for k, im in enumerate(imgs):
                    im.save(paths[l["stem"]][u][k])
            print(f"  [{n}/{len(todo)}] {l['stem']}", flush=True)
    return paths


def score_models(labels, paths):
    """{stem: {"up": (6,4), "front": (6,4), "geo": (6,)}} of raw probe scores."""
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    import classify_stls as C

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(BACKBONE, torch_dtype=torch.float16).to(dev).eval()
    proc = AutoProcessor.from_pretrained(BACKBONE)
    upT = C.embed_raw(model, proc, pose.UPRIGHT_PROMPTS, dev).float().cpu().numpy()
    dnT = C.embed_raw(model, proc, pose.TOPPLED_PROMPTS, dev).float().cpu().numpy()
    frT = C.embed_raw(model, proc, pose.FRONT_PROMPTS, dev).float().cpu().numpy()
    bkT = C.embed_raw(model, proc, pose.BACK_PROMPTS, dev).float().cpu().numpy()

    geo_src = build_tiles(labels, 2048)      # geometry only; tiles already cached
    out = {}
    for l in labels:
        flat = [p for row in paths[l["stem"]] for p in row]
        imgs = [Image.open(p).convert("RGB") for p in flat]
        emb = C.embed_images(model, proc, imgs, dev).float().cpu().numpy()
        up_s = pose.upright_scores(emb, upT, dnT).reshape(6, N_AZ)
        fr_s = ((emb @ frT.T).mean(1) - (emb @ bkT.T).mean(1)).reshape(6, N_AZ)
        out[l["stem"]] = {"up": up_s, "front": fr_s,
                          "geo": np.asarray(geo_src[l["stem"]]["geo"])}
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


def methods(sc, fmap):
    """{name: (up_scores(6), valid_mask(6), chosen_front_axis or None)}.

    The mask matters: fixing the front rules out the two ups parallel to it, and
    those have no score at all rather than a bad one. Leaving them as -inf and
    min-maxing produces NaN, which silently poisons the ensemble.
    """
    up_s, fr_s = sc["up"], sc["front"]
    all_valid = np.ones(6, dtype=bool)
    out = {}
    out["up-only, 1 view (today)"] = (up_s[:, 0], all_valid, None)
    out["up-only, 4 views mean"] = (up_s.mean(1), all_valid, None)
    out["up-only, 4 views max"] = (up_s.max(1), all_valid, None)

    # front first: score each mesh axis as "the front", marginalising over the
    # four ups it is compatible with, then let that column decide up
    front_axis = np.zeros(6)
    for u in range(6):
        for k in range(N_AZ):
            front_axis[fmap[u, k]] += fr_s[u, k]
    f_star = int(np.argmax(front_axis))
    col, valid = np.zeros(6), np.zeros(6, dtype=bool)
    for u in range(6):
        for k in range(N_AZ):
            if fmap[u, k] == f_star:
                col[u], valid[u] = up_s[u, k], True
    out["front first -> up"] = (col, valid, f_star)

    # joint: one argmax over all 24 orientations, both probe sets weighted equally
    joint = pose._unit(up_s.ravel()) + pose._unit(fr_s.ravel())
    out["joint (up + front)"] = (joint.reshape(6, N_AZ).max(1), all_valid, None)
    return out


def pick(scores, valid, geo=None):
    """Argmax over the valid ups, ensembled with geometry when given. Both
    vectors are min-maxed over the valid subset so a masked-out candidate
    cannot skew the normalisation it is excluded from."""
    idx = np.flatnonzero(valid)
    v = pose._unit(scores[idx])
    if geo is not None:
        v = v + pose._unit(np.asarray(geo)[idx])
    return int(idx[int(np.argmax(v))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="all")
    ap.add_argument("--render-px", type=int, default=RENDER_PX)
    args = ap.parse_args()

    labels = load_labels(None if args.set == "all" else args.set)
    fmap = front_axis_map()
    paths = build_orbit_tiles(labels, args.render_px)
    print("\nSigLIP phase")
    sc = score_models(labels, paths)

    gold = {l["stem"]: l["gold"] for l in labels}
    sets = {}
    for l in labels:
        sets.setdefault(l["set"], []).append(l["stem"])
    sets["orig+hold"] = sets.get("orig", []) + sets.get("holdout", [])

    per_model = {s: methods(sc[s], fmap) for s in gold}
    names = list(next(iter(per_model.values())))

    order = [n for n in ("orig", "holdout", "orig+hold", "hard") if sets.get(n)]
    hdr = f"{'method':28} " + " ".join(f"{n:>11}" for n in order)
    print(f"\nSigLIP alone — up-axis accuracy\n{hdr}\n" + "-" * len(hdr))
    for name in names:
        cells = []
        for grp in order:
            stems = sets[grp]
            ok = sum(pick(*per_model[s][name][:2]) == gold[s] for s in stems)
            cells.append(f"{ok}/{len(stems)}")
        print(f"{name:28} " + " ".join(f"{c:>11}" for c in cells))

    print(f"\nensembled with geometry\n{hdr}\n" + "-" * len(hdr))
    for name in names:
        cells = []
        for grp in order:
            stems = sets[grp]
            ok = sum(pick(*per_model[s][name][:2], geo=sc[s]["geo"]) == gold[s]
                     for s in stems)
            cells.append(f"{ok}/{len(stems)}")
        print(f"{name:28} " + " ".join(f"{c:>11}" for c in cells))

    # Front has no ground truth, so the most we can say is whether the front it
    # picks is at least consistent with the true up (front must be perpendicular).
    ff = "front first -> up"
    perp = sum(abs(float(np.dot(pose.UP_CANDIDATES[per_model[s][ff][2]],
                                pose.UP_CANDIDATES[gold[s]]))) < 1e-9 for s in gold)
    print(f"\nchosen front is perpendicular to the true up on {perp}/{len(gold)} models"
          "\n(there is no front ground truth — this is a consistency check, not accuracy)")

    json.dump({s: {"gold": AX[gold[s]],
                   **{n: AX[pick(v[0], v[1], geo=sc[s]["geo"])]
                      for n, v in per_model[s].items()},
                   "front_pick": AX[per_model[s][ff][2]]} for s in gold},
              open(OUT / "front_first.json", "w"), indent=1)
    print(f"\nwrote {OUT}/front_first.json")


if __name__ == "__main__":
    main()
