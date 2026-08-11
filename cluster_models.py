"""Cluster the collection by cached embeddings — discover structure without categories.

Usage:
  python cluster_models.py /path/to/stls --k 10 --renders-dir my_renders

Groups models by visual similarity (k-means over the classifier's cached
SigLIP embeddings — no text, no model download). For each cluster prints the
members closest to the centroid, writes assignments to a CSV, and (if a
renders directory is given) saves a contact-sheet image per cluster so you can
eyeball and name the groups.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

import pose
from classify_stls import add_cache_args, apply_run_params, load_file_list
from test_categories import load_embedding_matrix


def contact_sheet(members, renders_dir, out_path, thumb=160, cols=6, max_tiles=36,
                  poses=None):
    tiles, no_front, no_render = [], 0, 0
    for f in members[:max_tiles]:
        front = (poses or {}).get(pose.file_identity(f), {}).get("front_view", 0)
        img_path = renders_dir / f"{f.stem}_view{front}.png"
        if not img_path.exists():
            no_front += 1
            alts = sorted(renders_dir.glob(f"{f.stem}_view*.png"))
            img_path = alts[0] if alts else None
        if img_path is None:
            no_render += 1
        else:
            im = Image.open(img_path)
            im.thumbnail((thumb, thumb))
            tiles.append(im)
    if no_front:
        print(f"  {out_path.name}: front render missing for {no_front} of "
              f"{len(members[:max_tiles])} tiles"
              + (f", {no_render} skipped entirely" if no_render else "")
              + " — rerun classify_stls.py --save-renders")
    if not tiles:
        return False
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
    sheet.save(out_path)
    return True


def main():
    parser = argparse.ArgumentParser()
    # cache-identity params default to the last classify_stls.py run
    add_cache_args(parser, "STL directory (defaults to the last classify_stls.py run)")
    parser.add_argument("--k", type=int, default=10, help="number of clusters")
    parser.add_argument("--out", default="clusters.csv")
    parser.add_argument("--renders-dir", help="renders saved by classify_stls.py --save-renders; "
                                              "enables per-cluster contact sheets")
    parser.add_argument("--sheets-dir", default="cluster-sheets")
    args = apply_run_params(parser)
    if not args.input:
        raise SystemExit("no input given, and no directory recorded by classify_stls.py — "
                         "pass the STL directory explicitly")

    files = load_file_list(Path(args.input), args.cache_dir)
    matrix, files, missing = load_embedding_matrix(files, args)
    poses = pose.load_pose_cache(args.cache_dir)
    matrix = matrix.mean(axis=1)  # pool the per-view embeddings
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    print(f"clustering {len(files)} models into {args.k} groups"
          + (f" ({missing} not in cache — run classify_stls.py to add them)" if missing else ""))

    km = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit(matrix)
    # distance to own centroid = how typical a member is of its cluster
    dist = np.linalg.norm(matrix - km.cluster_centers_[km.labels_], axis=1)

    sheets_dir = Path(args.sheets_dir)
    renders_dir = Path(args.renders_dir) if args.renders_dir else None
    if renders_dir:
        sheets_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    order = np.argsort(-np.bincount(km.labels_, minlength=args.k))  # biggest first
    for c in order:
        members = np.where(km.labels_ == c)[0]
        members = members[np.argsort(dist[members])]  # most central first
        names = [files[i].stem for i in members]
        print(f"\ncluster {c} ({len(members)} models): " + ", ".join(names[:6])
              + (" ..." if len(names) > 6 else ""))
        if renders_dir:
            if contact_sheet([files[i] for i in members], renders_dir,
                             sheets_dir / f"cluster_{c:02d}.png", poses=poses):
                print(f"  sheet: {sheets_dir}/cluster_{c:02d}.png")
        for i in members:
            rows.append({"file": str(files[i]), "cluster": int(c),
                         "centroid_dist": round(float(dist[i]), 4)})

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "cluster", "centroid_dist"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
