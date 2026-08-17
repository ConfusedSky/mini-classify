"""Can SigLIP pick the upright orientation from the 6 up-candidate tiles?

For each sampled mesh: render the 6 candidate-up tiles (the same ones the VLM
arbiter is shown), embed them, score each against several upright/toppled probe
sets, and record the argmax alongside the geometry answer and the cached pose.
Writes results.json + one contact sheet per model for hand-labelling.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import OUT, AX, IDX, collection_root, load_labels, mark, score  # puts REPO on sys.path

import classify_stls as C
from src import pose

OUT = OUT / "siglip_up"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
RENDER = 384
AXIS = ["+Z", "-Z", "+Y", "-Y", "+X", "-X"]

PROBES = {
    "upright_toppled": (
        ["a miniature figurine standing upright on its base, the way it sits on a table",
         "a miniature standing upright, head at the top and feet at the bottom"],
        ["a miniature figurine lying on its side, toppled over",
         "a miniature figurine upside down, head at the bottom"],
    ),
    "plain_orientation": (
        ["an upright 3D render of a miniature"],
        ["a sideways 3D render of a miniature",
         "an upside-down 3D render of a miniature"],
    ),
    "anatomical": (
        ["a figure with its head at the top of the image and its feet at the bottom"],
        ["a figure with its head at the bottom of the image",
         "a figure lying horizontally across the image"],
    ),
    "upright_only": (
        ["a miniature figurine standing upright on its base, the way it sits on a table"],
        [],
    ),
    # the collection is ~half terrain (walls, floors, crates), where "head at the
    # top" is meaningless — these avoid assuming the subject has anatomy
    "object_generic": (
        ["a 3D printed model sitting the right way up on a table"],
        ["a 3D printed model tipped onto its side",
         "a 3D printed model turned upside down"],
    ),
    "gravity": (
        ["an object resting stably on the ground the right way up"],
        ["an object lying on its side", "an object upside down"],
    ),
}


def geometry_scores(mesh, draws=3, n_samples=40000):
    """Per-candidate flat-base scores, averaged over draws — the sampler is
    unseeded and 4000 points leave the winner resting on ~30, so the scores need
    the noise beaten down before they can be compared or ensembled."""
    acc = np.zeros(len(pose.UP_CANDIDATES))
    for _ in range(draws):
        pcd = mesh.sample_points_uniformly(n_samples, use_triangle_normal=True)
        pts, nrm = np.asarray(pcd.points), np.asarray(pcd.normals)
        for i, up in enumerate(pose.UP_CANDIDATES):
            h = pts @ up
            ext = h.max() - h.min()
            if ext <= 0:
                continue
            acc[i] += np.mean((h < h.min() + 0.02 * ext) & (nrm @ up < -0.9))
    return acc / draws


def robust_geometry(mesh, draws=3, n_samples=40000):
    s = geometry_scores(mesh, draws, n_samples)
    order = np.argsort(s)[::-1]
    best, runner = s[order[0]], s[order[1]]
    return int(order[0]), float(runner / best if best > 0 else 1.0), float(best)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModel, AutoProcessor

    root = Path("/home/masa/Documents/tests/mini-classify")
    walk = json.load(open(root / "embed-cache/walk-c6c430c8f5b1b94cf063e538c33e63d8f4f7fd01.json"))
    pc = root / "embed-cache/pose-cache.json"
    pose_cache = json.load(open(pc)) if pc.exists() else {}
    files = [Path(p) for p in walk["files"]]
    stl_root = collection_root()   # cache keys are relative to this
    random.seed(11)
    sample = random.sample(files, min(N, len(files)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/siglip2-so400m-patch14-384"
    print(f"loading {model_id} on {device}")
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
    proc = AutoProcessor.from_pretrained(model_id)

    text = {k: (C.embed_raw(model, proc, pos, device).float().cpu().numpy(),
                C.embed_raw(model, proc, neg, device).float().cpu().numpy() if neg else None)
            for k, (pos, neg) in PROBES.items()}

    renderer = C.make_renderer(RENDER)
    results = []
    for i, f in enumerate(sample):
        try:
            mesh = C.load_mesh(f)
        except Exception as e:
            print(f"[{i+1}/{len(sample)}] {f.stem}: load failed ({e})")
            continue
        gs = geometry_scores(mesh)
        gorder = np.argsort(gs)[::-1]
        gi = int(gorder[0])
        best, runner = float(gs[gorder[0]]), float(gs[gorder[1]])
        ratio = runner / best if best > 0 else 1.0
        tiles = C.render_up_candidate_tiles(renderer, mesh)
        emb = C.embed_images(model, proc, tiles, device).float().cpu().numpy()

        picks = {}
        for k, (pos, neg) in text.items():
            score = (emb @ pos.T).mean(1)
            if neg is not None:
                score = score - (emb @ neg.T).mean(1)
            picks[k] = {"pick": int(np.argmax(score)),
                        "scores": [round(float(v), 4) for v in score]}

        cached = pose_cache.get(pose.file_identity(f, stl_root))
        cached_i = None
        if cached:
            cached_i = next((j for j, u in enumerate(pose.UP_CANDIDATES)
                             if np.allclose(u, cached["up"])), None)
        results.append({
            "file": str(f), "stem": f.stem,
            "geo": {"pick": gi, "ratio": round(ratio, 4), "best": round(best, 5),
                    "scores": [round(float(v), 6) for v in gs]},
            "cached": {"pick": cached_i, "source": cached["source"]} if cached else None,
            "siglip": picks,
        })
        pose.make_contact_sheet(tiles).save(OUT / f"{i:03d}_{f.stem}.png")
        print(f"[{i+1}/{len(sample)}] {f.stem[:44]:44} geo={AXIS[gi]} "
              f"(best {best:.3f}) cached={AXIS[cached_i] if cached_i is not None else '--':>2}"
              f"/{cached['source'][:4] if cached else '----':4} "
              + " ".join(f"{k[:4]}={AXIS[v['pick']]}" for k, v in picks.items()))

    json.dump(results, open(OUT / "results.json", "w"), indent=1)
    print(f"\nwrote {OUT}/results.json  ({len(results)} models)")


if __name__ == "__main__":   # importable without re-running the sweep
    main()
