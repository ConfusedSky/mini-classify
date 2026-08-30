"""Append one model (by path substring) to sample.json + arm_open3d.npy."""
import argparse, json, sys
from pathlib import Path
import numpy as np
REPO = Path.home() / "Documents/tests/mini-classify"; sys.path.insert(0, str(REPO))
from src import pose
from src.cachedir import add_cache_args, apply_run_params, cache_root, load_file_list
from src.embed_store import load_embedding_matrix
from src.identity import render_key, rel_path
HERE = Path(__file__).parent
needle = sys.argv[1]
sys.argv = [sys.argv[0], "--cache-dir", str(REPO / "embed-cache512")]
p = argparse.ArgumentParser(); add_cache_args(p, ""); args = apply_run_params(p)
root = cache_root(Path(args.input), args.cache_dir, confirm=False)
files = load_file_list(Path(args.input), args.cache_dir, False)
matrix, files, _ = load_embedding_matrix(files, args, root)
poses = pose.load_pose_cache(args.cache_dir)
S = json.load(open(HERE / "sample.json")); A = np.load(HERE / "arm_open3d.npy")
hits = [i for i, f in enumerate(files) if needle in str(f)]
assert len(hits) == 1, hits
i = hits[0]; f = files[i]
if any(m["row"] == i for m in S["sample"]): sys.exit("already in sample")
S["sample"].append({"n": len(S["sample"]), "row": i, "path": str(f), "rel": rel_path(f, root),
                    "key": render_key(f, root), "up": poses[pose.file_identity(f, root)]["up"]})
json.dump(S, open(HERE / "sample.json", "w"), indent=1)
np.save(HERE / "arm_open3d.npy", np.concatenate([A, matrix[i:i+1]]).astype(np.float32))
print("added", S["sample"][-1]["key"], "n=", S["sample"][-1]["n"], "up", S["sample"][-1]["up"])
