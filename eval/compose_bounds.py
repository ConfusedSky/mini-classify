"""What the composing bounds cost, and what the composition order can be seen in.

Two measurements, one script, because they are the two halves of the same
claim (LEARNINGS, "the default that only became load-bearing later"):

* `--floor-set` (needs a cache and SigLIP): scores one phrase against a real
  cache and reports what each ranking arrangement returns — the floor set with
  no count, and the same request under the ten-row default `QueryRequest.top`
  carried before 2026-08-27. This is the 875 -> 10 the composition change was
  argued from, and the number `src/query.py`'s `rank` docstring cites.

* `--orders` (pure numpy, no cache, seconds): fuzzes floor-then-count against
  count-then-floor. They select the **same rows** for every input — the floor
  tests the same key the sort ordered, so the floor set is always a prefix of
  the descending order — and differ only in what can be *reported*. This is
  why the order is load-bearing for `matched` and for nothing else, and why a
  contract test asserting rows cannot see it.

Both run by default. CPU is fine and is what the recorded numbers were taken
on: these are counts, not timings, and a resident `serve_api.py` holds the
4060 often enough that contending for it buys nothing (CLAUDE.md).

Recorded run, 2026-08-27 — embed-cache512, 3380 models on
/run/media/masa/STLLibrary, `fantasy character` templated, pool softmax,
floor 0.1: 875 rows with no count, 10 under the old default, floor set
0.1463 down to 0.1000 with the 10th at 0.1426. Orders, at this script's
defaults: **0** row divergences in 20000 cases, `matched` differing in
**11560**.

The zero is the finding and is parameter-free — it follows from the prefix
argument, so any fuzz gets it. The 11560 is not: it is a function of this
generator's spread, floor and count draws, and a model-browser session
fuzzing the same claim independently got 8926 of 20000 under its own
parameters. Quote the zero; quote the other only with the script and seed
that produced it.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from src import query                                        # noqa: E402
from src.cachedir import add_cache_args, apply_run_params    # noqa: E402
from src.collection import Collection                        # noqa: E402

OLD_TOP = 10            # QueryRequest.top's default before the composition


def floor_set(args):
    """The 875 -> 10, through the production path."""
    c = Collection.load(args)
    print(f"cache {args.cache_dir} — {len(c.files)} models, root {c.root}")

    from src.embedder import embed_raw, embed_texts, load_siglip
    print(f"loading {args.model} on cpu ...")
    model, processor = load_siglip(args.model, "cpu")
    fn = embed_raw if args.raw else embed_texts
    text_T = fn(model, processor, [args.text], "cpu").float().cpu().numpy().T

    sims = query.score(c.matrix, text_T, args.pool).ravel()
    print(f"scored {len(sims)} on {args.text!r} "
          f"(pool={args.pool}, {'raw' if args.raw else 'templated'})")

    floor = query.rank(sims, min_score=args.floor)
    old = query.rank(sims, top=OLD_TOP, min_score=args.floor)
    count = query.rank(sims, top=OLD_TOP)

    print(f"\nfloor {args.floor}, no count       : {len(floor.order):5d} rows "
          f"(matched {floor.matched})")
    print(f"floor {args.floor}, top={OLD_TOP} (old default): {len(old.order):5d} rows "
          f"(matched {old.matched})")
    print(f"top={OLD_TOP}, no floor           : {len(count.order):5d} rows "
          f"(matched {count.matched})")

    above = sims[floor.order]
    if len(above):
        print(f"\nfloor set runs {above[0]:.4f} down to {above[-1]:.4f}; "
              f"{OLD_TOP}th is {above[min(OLD_TOP, len(above)) - 1]:.4f}")
    print(f"weak={floor.weak} best_z={floor.z[floor.best]:.2f}; "
          f"the 500 cap would truncate it: {len(floor.order) > 500}")


def orders(trials=20000, seed=0):
    """Floor-then-count vs count-then-floor: same rows, different `matched`."""
    rng = np.random.default_rng(seed)
    rows_differ = matched_differ = 0
    for _ in range(trials):
        n = int(rng.integers(1, 40))
        # rounded on purpose: continuous normals essentially never tie, and a
        # fuzz that cannot see ties cannot see the prefix argument's edge
        sims = np.round(rng.normal(size=n), 1).astype(np.float32)
        t = int(rng.integers(1, 12))
        f = float(np.round(rng.normal(), 1))
        o = np.argsort(-sims, kind="stable")

        kept = o[sims[o] >= f]
        floor_then = kept[:t]                       # what `rank` does
        cut = o[:t]
        then_floor = cut[sims[cut] >= f]            # the other order

        rows_differ += floor_then.tolist() != then_floor.tolist()
        matched_differ += len(kept) != len(cut[sims[cut] >= f])

    print(f"\n{trials} fuzzed cases (ties forced): rows differ in "
          f"{rows_differ}, `matched` differs in {matched_differ}")
    print("rows: identical by construction — the floor tests the sort key, so "
          "the floor set is a prefix of the order")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_cache_args(parser, "STL directory (defaults to the last classify run)")
    parser.add_argument("--text", default="fantasy character")
    parser.add_argument("--floor", type=float, default=0.1)
    parser.add_argument("--pool", choices=["mean", "max", "softmax"],
                        default="softmax")
    parser.add_argument("--raw", action="store_true",
                        help="verbatim query text instead of the templates")
    parser.add_argument("--floor-set", action="store_true",
                        help="only the cache-backed half")
    parser.add_argument("--orders", action="store_true",
                        help="only the numpy fuzz (no cache, no model)")
    args = apply_run_params(parser)

    both = not (args.floor_set or args.orders)
    if both or args.floor_set:
        if not args.input:
            sys.exit("no input given, and none recorded by classify_stls.py")
        floor_set(args)
    if both or args.orders:
        orders()


main()
