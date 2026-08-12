"""Run every up-detection method on specific meshes and dump per-tile scores."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import classify_stls as C
import pose
from siglip_up import PROBES, robust_geometry, geometry_scores

DUMP = {}

OUT = OUT / "siglip_up"
AXIS = ["+Z", "-Z", "+Y", "-Y", "+X", "-X"]
TARGETS = [
    "/run/media/masa/Files and S/STL/Loot Studios/A Light In the Shadow/Pit Fiend/32mm/No Supports/32mm_PitFiend.stl",
    "/run/media/masa/Files and S/STL/Loot Studios/A Light In the Shadow/Pit Fiend/Bust/No Supports/PitFiend_Bust.stl",
]

from transformers import AutoModel, AutoProcessor

from common import OUT, AX, IDX, load_labels, mark, score

device = "cuda" if torch.cuda.is_available() else "cpu"
mid = "google/siglip2-so400m-patch14-384"
model = AutoModel.from_pretrained(mid, torch_dtype=torch.float16).to(device).eval()
proc = AutoProcessor.from_pretrained(mid)
text = {k: (C.embed_raw(model, proc, p, device).float().cpu().numpy(),
            C.embed_raw(model, proc, n, device).float().cpu().numpy() if n else None)
        for k, (p, n) in PROBES.items()}

renderer = C.make_renderer(384)
for stl in TARGETS:
    f = Path(stl)
    mesh = C.load_mesh(f)
    gi, ratio, best = robust_geometry(mesh)

    # raw per-candidate geometry scores too, for the writeup
    pcd = mesh.sample_points_uniformly(40000, use_triangle_normal=True)
    pts, nrm = np.asarray(pcd.points), np.asarray(pcd.normals)
    gscores = []
    for up in pose.UP_CANDIDATES:
        h = pts @ up
        ext = h.max() - h.min()
        gscores.append(float(np.mean((h < h.min() + 0.02 * ext) & (nrm @ up < -0.9))) if ext > 0 else 0.0)

    tiles = C.render_up_candidate_tiles(renderer, mesh)
    emb = C.embed_images(model, proc, tiles, device).float().cpu().numpy()
    pose.make_contact_sheet(tiles).save(OUT / f"TARGET_{f.stem}.png")

    print(f"\n=== {f.stem}   (geometry: {AXIS[gi]}, best={best:.4f}, ratio={ratio:.3f}, "
          f"needs_arbiter={pose.needs_arbiter(ratio, best, 0.6)})")
    print(f"{'method':18} " + " ".join(f"{a:>8}" for a in AXIS) + "   pick")
    print(f"{'geometry':18} " + " ".join(f"{s:8.4f}" for s in gscores) + f"   {AXIS[int(np.argmax(gscores))]}")
    entry = {"stem": f.stem, "geo": {"scores": [float(v) for v in gscores],
                                     "best": best, "ratio": ratio}, "siglip": {}}
    for k, (p, n) in text.items():
        sc = (emb @ p.T).mean(1)
        if n is not None:
            sc = sc - (emb @ n.T).mean(1)
        entry["siglip"][k] = {"scores": [float(v) for v in sc]}
        print(f"{k:18} " + " ".join(f"{v:8.4f}" for v in sc) + f"   {AXIS[int(np.argmax(sc))]}")
    DUMP[f.stem] = entry

json.dump(DUMP, open(OUT / "targets.json", "w"), indent=1)
print(f"\nwrote {OUT}/targets.json")
