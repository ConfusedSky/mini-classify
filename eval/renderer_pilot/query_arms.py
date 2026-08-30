"""Query the renderer-pilot arms side by side with your own text.

  .venv/bin/python eval/renderer_pilot/query_arms.py            # REPL
  .venv/bin/python eval/renderer_pilot/query_arms.py "dragon" "treasure chest"

Each query prints one column per arm: the top-N of the 301-model sample under
that arm's embeddings, scored exactly as production does (`src.query`,
templated text). Names are the render keys — the matching pictures are
`renders/<arm>/<key>_v<i>.png` beside this script (v0..7 at +20°, v8..15 at
-20°; the control arm's renders are the cache's own, under
embed-cache512/renders/). REPL commands: `:n 15` rows, `:pool mean|max|softmax`,
`:raw` toggles templated vs verbatim text, `:arms f3d,three_ao` to narrow.
"""
import json, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src import query as Q
from src.embedder import Embedder

HERE = Path(__file__).parent
S = json.load(open(HERE / "sample.json"))
NAMES = [m["key"] for m in S["sample"]]
ALL_ARMS = ["open3d", "open3d_tight", "f3d", "three_ao", "three_noao", "three_white"]
TEMPLATES = ["a 3D render of a {} miniature", "a photo of a {} figurine", "a tabletop miniature of a {}"]


class Arms:
    def __init__(self):
        self.M = {a: np.load(HERE / f"arm_{a}.npy") for a in ALL_ARMS if (HERE / f"arm_{a}.npy").exists()}
        self.emb = Embedder(categories=["placeholder"], model_name="google/siglip2-so400m-patch16-512")
        self.n, self.pool, self.raw, self.arms = 10, "softmax", False, list(self.M)

    def text(self, q):
        texts = [q] if self.raw else [t.format(q) for t in TEMPLATES]
        f = self.emb._embed_raw(texts).float().cpu().numpy().mean(0)
        return (f / np.linalg.norm(f))[:, None]

    def run(self, q):
        t = self.text(q)
        cols = {}
        for a in self.arms:
            M = self.M[a]
            sims = Q.score(M, t, self.pool)[:, 0]
            sims[np.abs(M).sum(axis=(1, 2)) == 0] = -1          # unrendered (f3d could not load)
            r = Q.rank(sims, top=self.n)
            cols[a] = [(NAMES[i], float(r.z[i])) for i in r.order]
        w = 28
        print(f"\n{q!r}  pool={self.pool} {'raw' if self.raw else 'templated'}")
        print("     " + "".join(f"{a:<{w}}" for a in self.arms))
        for k in range(self.n):
            print(f"{k+1:>3}  " + "".join(f"{c[k][0][:w-6]:<{w-6}}{c[k][1]:>5.1f} " if k < len(c) else " " * w
                                          for c in cols.values()))
        top = [set(n for n, _ in c) for c in cols.values()]
        common = set.intersection(*top) if top else set()
        print(f"     in every arm's top-{self.n}: {len(common)}; z is robust_z against the 301-model sample")

    def repl(self):
        print(f"arms: {', '.join(self.arms)} — {len(NAMES)} models. Type a query; :q to quit.")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line or line == ":q":
                break
            if line.startswith(":n "):
                self.n = int(line[3:])
            elif line.startswith(":pool "):
                self.pool = line[6:].strip()
            elif line == ":raw":
                self.raw = not self.raw; print("raw" if self.raw else "templated")
            elif line.startswith(":arms "):
                self.arms = [a for a in line[6:].replace(",", " ").split() if a in self.M] or list(self.M)
            else:
                self.run(line)


if __name__ == "__main__":
    arms = Arms()
    if len(sys.argv) > 1:
        for q in sys.argv[1:]:
            arms.run(q)
    else:
        arms.repl()
