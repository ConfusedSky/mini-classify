"""Re-key poses for files whose bytes were rewritten in place at the same length.

Usage:
  python migrate_pose_mtimes.py --cache-dir embed-cache-test \
      --rewrites ~/Documents/tests/test-models/metadata/rewound.json [--apply]

The corpus project re-winds inside-out meshes by writing the facet block over
itself (`scripts/fix_normals.py` there). Binary STL is `84 + 50*facets` and a
winding flip changes no count, so the file keeps its size and its inode — but
not its mtime, and mtime is a third of the identity every cache here is keyed on
(identity.py). Every rewritten model therefore misses, and a miss re-resolves
the pose, which bills a VLM call.

The pose survives the rewrite, and that is the whole argument for this tool: a
winding flip moves no vertex. The up-axis of the mesh is the same fact about the
same geometry. Renders and embeddings do *not* survive, and are deliberately not
migrated — the mesh now shades the other way round, which is the entire point of
having fixed it, so the cached views are of something that no longer exists.
Those are GPU-minutes; the pose is money.

What identifies the moved key is the pair (relative path, size) plus the
rewriter's own record of which files it touched. Size alone is not evidence —
`84 + 50*tris` makes an equal size guaranteed for any re-export at the same
triangle count — so the `--rewrites` list is the gate, and nothing outside it is
considered.

Dry run by default. Re-runnable: a file already filed under its current identity
is left alone.
"""
import argparse
import json
import sys
from pathlib import Path

from src import pose
from src.cachedir import write_atomic
from src.identity import collection_root


def rewritten_paths(manifest: Path):
    """{collection-root-relative path: set of sizes it was rewritten at}.

    The paths carry a leading tree name (`original/`, `clustered-hq/`) because
    the rewriter walks several sibling trees of one corpus, while a cache is
    keyed under whichever single tree it was built against. Stripping that
    component is what lets a cache built on `deduplicated/` claim a rewrite
    recorded against `original/` — the normal case, since those two are
    hardlinks to one inode and a rewrite lands on both.

    It also means a rewrite recorded against `clustered-hq/` would otherwise
    license re-keying the `deduplicated/` file of the same name, which is a
    different file that may never have been touched. Carrying the size the
    rewriter saw is what closes that: a re-key has to match a size some tree
    was actually rewritten at, not merely the size the file happens to have now.
    """
    sizes = {}
    for e in json.loads(manifest.read_text()):
        rel = str(Path(*Path(e["path"]).parts[1:]))
        sizes.setdefault(rel, set()).add(e["size"])
    return sizes


def plan(cache, root, rewrites):
    """(moves, already, absent, unclaimed) for the named rewrites.

    `moves` maps an old key to the current identity of the same file. A file is
    matched on its relative path and its byte count, never on mtime: the old
    mtime is precisely what was lost. The byte count has to agree three ways —
    the cache key, the file on disk, and the size the rewriter recorded — or the
    entry is left for the next run to resolve honestly.
    """
    by_rel = {}
    for key in cache:
        rel, _, size = key.rsplit("|", 2)
        by_rel.setdefault(rel, []).append((key, size))

    moves, already, absent, unclaimed = {}, [], [], []
    for rel, rewritten_sizes in sorted(rewrites.items()):
        f = root / rel
        if not f.is_file():
            absent.append(rel)
            continue
        new = pose.file_identity(f, root)
        if new in cache:
            already.append(rel)
            continue
        size = f.stat().st_size
        if size not in rewritten_sizes:
            # The name was rewritten in some tree, but not at this file's size,
            # so this is not that file.
            unclaimed.append(rel)
            continue
        candidates = [k for k, s in by_rel.get(rel, []) if s == str(size)]
        if not candidates:
            unclaimed.append(rel)
            continue
        # A file rewritten twice leaves two stale keys. The later mtime is the
        # later resolution, so it is the one worth keeping.
        moves[max(candidates, key=lambda k: int(k.rsplit("|", 2)[1]))] = new
    return moves, already, absent, unclaimed


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-dir", default="embed-cache",
                    help="cache directory holding pose-cache.json")
    ap.add_argument("--rewrites", type=Path, required=True,
                    help="fix_normals.py's manifest of in-place rewrites")
    ap.add_argument("--root", type=Path,
                    help="collection root; defaults to the one run-params.json "
                         "recorded for this cache")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    params_path = cache_dir / "run-params.json"
    root = args.root
    if root is None:
        if not params_path.exists():
            sys.exit(f"{params_path} does not exist; pass --root")
        recorded = json.loads(params_path.read_text()).get("collection_root")
        if not recorded:
            sys.exit(f"{params_path} records no collection_root; pass --root")
        root = Path(recorded)
    root = collection_root(root)

    path = cache_dir / "pose-cache.json"
    cache = json.loads(path.read_text())
    rewrites = rewritten_paths(args.rewrites)
    moves, already, absent, unclaimed = plan(cache, root, rewrites)

    print(f"collection root {root}")
    print(f"rewrites   {len(rewrites)} files named by the manifest")
    print(f"poses      {len(moves)} to re-key of {len(cache)} cached"
          + (f", {len(already)} already current" if already else ""))
    paid = [k for k in moves if cache[k].get("source") == "vlm"]
    if paid:
        print(f"           {len(paid)} of them are VLM-sourced — that is the "
              f"paid arbiter call this run is saving")
    if absent:
        print(f"           {len(absent)} named files are not under this root "
              f"(a different tree of the corpus); ignored")
    if unclaimed:
        # Either the model was never posed, or it was posed and then rewritten
        # at a *different* size, which this tool must not guess about.
        print(f"           {len(unclaimed)} have no cached pose at a size the "
              f"rewriter recorded; they will be resolved on the next run")
    print("renders and embeddings are left to miss on purpose: the mesh shades "
          "the other way round now, so the cached views are stale.")

    if not args.apply:
        print("\ndry run — nothing changed; pass --apply to do it")
        return

    for old, new in moves.items():
        cache[new] = cache.pop(old)
    # write_atomic, not pose.save_pose_cache: a torn write here loses the one
    # file in the cache that costs money to rebuild.
    write_atomic(path, json.dumps(cache))
    print(f"\n{len(moves)} poses re-keyed in {path}")


if __name__ == "__main__":
    main()
