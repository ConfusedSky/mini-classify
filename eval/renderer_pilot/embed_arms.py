"""Embed each render arm with the production Embedder -> arm_<name>.npy
(n_models, n_views, dim). RGBA three.js renders are composited over the app's
tile background (zinc-900, #18181b) — the shipped look."""
import json, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path.home() / "Documents/tests/mini-classify"
sys.path.insert(0, str(REPO))
from src.embedder import Embedder

HERE = Path(__file__).parent
S = json.load(open(HERE / "sample.json"))
ARMS = sys.argv[1:] or ["f3d", "open3d_tight", "three_ao", "three_noao", "three_white"]
BG = (0x18, 0x18, 0x1b)


def load(p):
    im = Image.open(p)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, BG)
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    return np.asarray(im.convert("RGB"))


def main():
    emb = Embedder(categories=["placeholder"], model_name=S.get("model", "google/siglip2-so400m-patch16-512"))
    nv = len(S["angles"])
    for arm in ARMS:
        d = HERE / "renders" / arm
        rows, missing, t0 = [], 0, time.time()
        for m in S["sample"]:
            paths = [d / f"{m['key']}_v{i}.png" for i in range(nv)]
            if not all(p.exists() for p in paths):
                missing += 1
                rows.append(np.zeros((nv, 1152), np.float32)); continue
            rows.append(emb.embed_images([load(p) for p in paths]).float().cpu().numpy())
        arr = np.stack(rows)
        np.save(HERE / f"arm_{arm}.npy", arr)
        print(f"{arm}: {arr.shape}, {missing} models missing renders, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
