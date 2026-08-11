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
from pathlib import Path

import numpy as np
import torch

from classify_stls import as_tensor, cache_key, embed_texts, load_file_list, pool_sims


def load_embedding_matrix(files, args):
    cache_dir = Path(args.cache_dir)
    vecs, kept, missing = [], [], 0
    for f in files:
        p = cache_dir / f"{cache_key(f, args)}.npy"
        if p.exists():
            vecs.append(np.load(p))
            kept.append(f)
        else:
            missing += 1
    if not vecs:
        raise SystemExit("no cached embeddings found — run classify_stls.py first")
    return np.stack(vecs).astype(np.float32), kept, missing  # (n_files, n_views, dim)


def show_classification(sims, categories, files, prev):
    assign = sims.argmax(1)
    print(f"\n{'category':<40} {'count':>5}   best match")
    for c in np.argsort(-np.bincount(assign, minlength=len(categories))):
        members = np.where(assign == c)[0]
        if len(members) == 0:
            print(f"{categories[c]:<40} {0:>5}   —")
            continue
        best = members[sims[members, c].argmax()]
        print(f"{categories[c]:<40} {len(members):>5}   {files[best].stem} ({sims[best, c]:.3f})")
    if prev is not None:
        changed = np.where(assign != prev["assign"])[0]
        print(f"\n{len(changed)} of {len(files)} models changed assignment")
        for i in changed[:15]:
            print(f"  {files[i].stem}: {prev['categories'][prev['assign'][i]]} -> {categories[assign[i]]}")
        if len(changed) > 15:
            print(f"  ... and {len(changed) - 15} more")
    return {"assign": assign, "categories": categories}


def show_query(sims_1d, files, top=10):
    for i in np.argsort(-sims_1d)[:top]:
        print(f"  {sims_1d[i]:.3f}  {files[i].stem}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="STL directory (same as passed to classify_stls.py)")
    parser.add_argument("--categories", default="categories.txt")
    # cache-identity params: must match the classify_stls.py run that built the cache
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--model", default="google/siglip2-so400m-patch14-384")
    parser.add_argument("--up-axis", choices=["auto", "z", "y"], default="auto")
    parser.add_argument("--cache-dir", default="embed-cache")
    parser.add_argument("--pool", choices=["mean", "max", "softmax"], default="mean")
    args = parser.parse_args()

    files = load_file_list(Path(args.input), args.cache_dir)
    matrix, files, missing = load_embedding_matrix(files, args)
    print(f"{len(files)} models with cached embeddings"
          + (f" ({missing} not in cache — run classify_stls.py to add them)" if missing else ""))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModel, AutoProcessor
    print(f"loading {args.model} on {device} ...")
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    def text_matrix(texts):
        emb = embed_texts(model, processor, texts, device)
        return emb.float().cpu().numpy().T  # (dim, n_texts)

    pool = args.pool

    def score(texts):  # (n_files, n_texts), pooled over views
        view_sims = matrix @ text_matrix(texts)  # (n_files, n_views, n_texts)
        return pool_sims(view_sims, pool)

    prev = None
    print(f"\nenter = classify with categories.txt | text = query | "
          f":pool mean|max|softmax (now {pool}) | q = quit")
    while True:
        try:
            line = input(f"\ncategory-test[{pool}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line.lower() in ("q", "quit", "exit"):
            break
        if line.startswith(":pool"):
            choice = line.split()[-1]
            if choice in ("mean", "max", "softmax"):
                pool = choice
                print(f"pooling set to {pool}")
            else:
                print("usage: :pool mean|max|softmax")
        elif line:
            show_query(score([line]).ravel(), files)
        else:
            categories = [l.strip() for l in open(args.categories) if l.strip()]
            prev = show_classification(score(categories), categories, files, prev)


if __name__ == "__main__":
    main()
