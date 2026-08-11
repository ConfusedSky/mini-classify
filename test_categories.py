"""Interactive category tester over the classifier's cached embeddings.

Usage: python test_categories.py /path/to/stls

Loads all cached embeddings once, keeps SigLIP warm, then loops:
  <enter>        reload categories.txt, classify everything, show distribution
                 and what changed since the previous iteration
  any text       one-off query: top matches across the collection
  q / quit       exit

Requires classify_stls.py to have been run first (it builds the caches);
files without cached embeddings are skipped with a warning.
"""
import argparse
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch

import pose
from classify_stls import (add_cache_args, apply_run_params, as_tensor, cache_key,
                           embed_texts, load_file_list, pool_sims, total_views)


def link(f, text):
    """OSC 8 terminal hyperlink to the file; plain text when piped."""
    if not sys.stdout.isatty():
        return text
    return f"\033]8;;file://{quote(str(f))}\033\\{text}\033]8;;\033\\"


def load_embedding_matrix(files, args):
    cache_dir = Path(args.cache_dir)
    poses = pose.load_pose_cache(args.cache_dir)
    vecs, kept, missing = [], [], 0
    for f in files:
        token = pose.embed_cache_token(poses.get(pose.file_identity(f)), args.up_axis)
        p = cache_dir / f"{cache_key(f, args, token)}.npy"
        if p.exists():
            vecs.append(np.load(p))
            kept.append(f)
        else:
            missing += 1
    if not vecs:
        raise SystemExit("no cached embeddings found — run classify_stls.py first")
    return np.stack(vecs).astype(np.float32), kept, missing  # (n_files, n_views, dim)


def show_classification(sims, categories, names, prev):
    assign = sims.argmax(1)
    print(f"\n{'category':<40} {'count':>5}   best match")
    for c in np.argsort(-np.bincount(assign, minlength=len(categories))):
        members = np.where(assign == c)[0]
        if len(members) == 0:
            print(f"{categories[c]:<40} {0:>5}   —")
            continue
        best = members[sims[members, c].argmax()]
        print(f"{categories[c]:<40} {len(members):>5}   {names[best]} ({sims[best, c]:.3f})")
    if prev is not None:
        changed = np.where(assign != prev["assign"])[0]
        print(f"\n{len(changed)} of {len(names)} models changed assignment")
        for i in changed[:15]:
            print(f"  {names[i]}: {prev['categories'][prev['assign'][i]]} -> {categories[assign[i]]}")
        if len(changed) > 15:
            print(f"  ... and {len(changed) - 15} more")
    return {"assign": assign, "categories": categories}


def show_query(sims_1d, names, top=10, min_score=None):
    # z-score: how far a model stands out from the whole collection for this
    # query. Cosine scores are only comparable within a query; z is comparable
    # across queries, so it detects "nothing here actually matches".
    # robust z (median/MAD): unlike mean/std it isn't skewed when many models
    # genuinely match, so broad queries don't get falsely flagged as weak
    med = np.median(sims_1d)
    mad = np.median(np.abs(sims_1d - med)) * 1.4826 + 1e-9
    z = (sims_1d - med) / mad
    order = np.argsort(-sims_1d)
    # 2.0 catches only unambiguous noise: measured on real queries, correct
    # matches ran z 2.4+ while semantic near-misses ran up to 3.7 — a higher
    # cutoff would suppress good results without stopping near-misses. Raw
    # scores < min_score (default 0.1) filter the middle ground instead.
    if z[order[0]] < 2.0:
        print(f"  WEAK QUERY (best z {z[order[0]]:.1f}) — nothing stands out; "
              f"probably not represented in the collection")
        return
    if min_score is not None:
        order = order[sims_1d[order] >= min_score]
        if len(order) == 0:
            print(f"  nothing scores >= {min_score}")
            return
        print(f"  {len(order)} models >= {min_score}:")
    else:
        order = order[:top]
    for i in order:
        print(f"  {sims_1d[i]:.3f} (z {z[i]:4.1f})  {names[i]}")


def main():
    parser = argparse.ArgumentParser()
    # cache-identity params default to the last classify_stls.py run
    add_cache_args(parser, "STL directory (defaults to the last classify_stls.py run)")
    parser.add_argument("--categories", default="categories.txt")
    parser.add_argument("--pool", choices=["mean", "max", "softmax"], default="mean")
    parser.add_argument("--min-score", type=float, default=0.1,
                        help="queries list every model scoring at least this instead of a "
                             "top-10 (default 0.1; pass -1 or use :min off for top-10 mode)")
    parser.add_argument("--renders-dir", default="my_renders",
                        help="saved renders; display names link to the render image when it exists")
    args = apply_run_params(parser)
    if not args.input:
        sys.exit("no input given, and no directory recorded by classify_stls.py — "
                 "pass the STL directory explicitly")

    root = Path(args.input)
    files = load_file_list(root, args.cache_dir)
    matrix, files, missing = load_embedding_matrix(files, args)
    renders_dir = Path(args.renders_dir)
    poses = pose.load_pose_cache(args.cache_dir)

    # display name: path relative to the input root, minus filler dirs.
    # Links open the front ("hero") render when it exists, any other saved
    # view otherwise, and only fall back to the STL when no render is saved.
    no_front = 0
    names = []
    for f in files:
        rel = str(f.relative_to(root)) if f.is_relative_to(root) else str(f)
        front = poses.get(pose.file_identity(f), {}).get("front_view", 0)
        candidates = [renders_dir / f"{f.stem}_view{i}.png"
                      for i in [front] + [v for v in range(total_views(args)) if v != front]]
        target = next((p.resolve() for p in candidates if p.exists()), f)
        no_front += not candidates[0].exists()
        names.append(link(target, rel.replace("/No Supports", "").removesuffix(".stl")))
    print(f"{len(files)} models with cached embeddings"
          + (f" ({missing} not in cache — run classify_stls.py to add them)" if missing else ""))
    if no_front:
        print(f"front render missing for {no_front} of {len(files)} models — links use "
              f"another view or the STL; run classify_stls.py --save-renders {renders_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModel, AutoProcessor
    print(f"loading {args.model} on {device} ...")
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    def text_matrix(texts):
        emb = embed_texts(model, processor, texts, device)
        return emb.float().cpu().numpy().T  # (dim, n_texts)

    pool = args.pool
    min_score = args.min_score if args.min_score >= 0 else None

    def score(texts):  # (n_files, n_texts), pooled over views
        view_sims = matrix @ text_matrix(texts)  # (n_files, n_views, n_texts)
        return pool_sims(view_sims, pool)

    prev = None
    print(f"\nenter = classify with categories.txt | text = query | :find <text> = "
          f"locate files | :pool mean|max|softmax (now {pool}) | "
          f":min <score>/off = threshold instead of top-10 | q = quit")
    while True:
        try:
            line = input(f"\ncategory-test[{pool}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line.lower() in ("q", "quit", "exit"):
            break
        if line.startswith(":min"):
            val = line[4:].strip()
            if val in ("off", ""):
                min_score = None
                print("showing top 10 per query")
            else:
                try:
                    min_score = float(val)
                    print(f"showing all results >= {min_score}")
                except ValueError:
                    print("usage: :min 0.1  or  :min off")
        elif line.startswith(":find"):
            needle = line[5:].strip().lower()
            matches = [f for f in files if needle in str(f).lower()]
            for f in matches[:20]:
                print(f"  {link(f, str(f))}")
            print(f"  ({len(matches)} matches)" if matches else "  no matches")
        elif line.startswith(":pool"):
            choice = line.split()[-1]
            if choice in ("mean", "max", "softmax"):
                pool = choice
                print(f"pooling set to {pool}")
            else:
                print("usage: :pool mean|max|softmax")
        elif line:
            show_query(score([line]).ravel(), names, min_score=min_score)
        else:
            categories = [l.strip() for l in open(args.categories) if l.strip()]
            prev = show_classification(score(categories), categories, names, prev)


if __name__ == "__main__":
    main()
