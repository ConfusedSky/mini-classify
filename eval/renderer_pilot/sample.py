"""Pick the pilot sample from embed-cache512 and pull the control arm's cached
embeddings for it. 12 queries; per query up to 12 filename-keyword matches
(renderer-independent seeding so each query has candidates), then random
fill to N_TOTAL. Writes sample.json + control.npy.
"""
import argparse, json, random, sys
from pathlib import Path

import numpy as np

REPO = Path.home() / "Documents/tests/mini-classify"
sys.path.insert(0, str(REPO))
from src import pose
from src.cachedir import add_cache_args, apply_run_params, cache_root, load_file_list
from src.embed_store import load_embedding_matrix
from src.identity import render_key, rel_path

HERE = Path(__file__).parent
N_TOTAL, PER_QUERY, SEED = 300, 12, 20260829
QUERIES = {                     # query text -> filename keywords used only for seeding
    "dragon": ["dragon"],
    "wizard casting a spell": ["mage", "wizard", "sorcer"],
    "barbarian warrior with a sword": ["barbarian"],
    "troll": ["troll"],
    "bust of a character": ["bust"],
    "rider mounted on a horse or beast": ["rider", "horse", "mount"],
    "demon": ["demon"],
    "stone golem": ["golem"],
    "goblin": ["goblin"],
    "knight in plate armor": ["knight", "paladin"],
    "robot or mech": ["robot", "mech"],
    "treasure chest": ["chest"],
}


def main():
    sys.argv = [sys.argv[0], "--cache-dir", str(REPO / "embed-cache512")]
    parser = argparse.ArgumentParser()
    add_cache_args(parser, "")
    args = apply_run_params(parser)
    root = cache_root(Path(args.input), args.cache_dir, confirm=False)
    files = load_file_list(Path(args.input), args.cache_dir, False)
    matrix, files, missing = load_embedding_matrix(files, args, root)
    poses = pose.load_pose_cache(args.cache_dir)
    print(f"{len(files)} embedded, {missing} missing, matrix {matrix.shape}")

    def up_of(f):
        return poses.get(pose.file_identity(f, root), {}).get("up")

    ok = [i for i, f in enumerate(files) if up_of(f) is not None]
    rng = random.Random(SEED)
    chosen, seeded = [], {}
    for q, kws in QUERIES.items():
        hits = [i for i in ok if any(k in files[i].name.lower() for k in kws) and i not in chosen]
        rng.shuffle(hits)
        seeded[q] = hits[:PER_QUERY]
        chosen += seeded[q]
    rest = [i for i in ok if i not in set(chosen)]
    rng.shuffle(rest)
    chosen += rest[:N_TOTAL - len(chosen)]
    chosen = sorted(set(chosen))
    print(f"sample {len(chosen)} (seeded {sum(len(v) for v in seeded.values())})")

    sample = []
    for n, i in enumerate(chosen):
        f = files[i]
        sample.append({"n": n, "row": i, "path": str(f), "rel": rel_path(f, root),
                       "key": render_key(f, root), "up": up_of(f)})
    json.dump({"root": str(root), "views": args.views, "elevations": list(args.elevations),
               "angles": [list(map(float, a)) for a in pose.view_angles(args.views, list(args.elevations))],
               "queries": {q: [files[i].name for i in v] for q, v in seeded.items()},
               "sample": sample}, open(HERE / "sample.json", "w"), indent=1)
    np.save(HERE / "arm_open3d.npy", matrix[chosen].astype(np.float32))
    print("wrote sample.json, arm_open3d.npy", matrix[chosen].shape)


if __name__ == "__main__":
    main()
