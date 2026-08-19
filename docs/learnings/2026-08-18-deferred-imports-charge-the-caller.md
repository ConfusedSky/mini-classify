# A deferred import charges the caller (2026-08-18)

Three modules in `src/` carried a docstring asserting they were light. Two of
them were wrong, and both had been made wrong by the same move: taking a heavy
import out of module scope and putting it inside the function that used it.
Deferring an import does not delete its cost. It relocates it onto whoever
calls the function — which is a fix only when that set is empty.

All counts below are what an import **adds** over a bare interpreter's 48
modules. Quoting totals instead is how the same measurement got reported as
both 186 and 138, and as both 1687 and 1639, by two sessions on the same day.

## What was measured

`add_cache_args` — the argparse block every tool shares — imported `--model`'s
default from `src.embedder` inside the function body, on the reasoning that
deferring it kept torch out of `cachedir`'s module scope. It did. It also
handed torch to every tool that built a parser:

| | before | after |
|---|---|---|
| `add_cache_args(parser)` | **+836 modules, ~0.92 s**, torch resident | **+0**, 0.0 s, no torch |

Measure the *call* with the parser already constructed, which is what every
caller does — that is the +0. Widen the region and the number moves without
the finding changing: **+9** if the first `ArgumentParser()` is built inside it
(gettext machinery, once per process), **+40** if `import src.cachedir` is
inside it too (its own +31, which already includes argparse, plus that 9).
None of the three contains torch. Say which region you measured: this one
number was quoted as +0, +29 and +31 by two sessions inside an hour, each time
correctly for a region neither had stated.

`cluster_models.py` and `migrate_cache_keys.py` paid that to name a string.
Neither touches torch otherwise. `DEFAULT_MODEL` now lives in `src/identity.py`
beside `DEFAULT_ELEVATIONS` — the model name is part of every embedding key, so
identity was its home anyway — and `src/embedder.py` re-exports it, so there is
still exactly one copy (`1a54713`).

`src/embed_store.py` claimed to be "deliberately numpy-only: loading cached
vectors is not a reason to load SigLIP". The torch half was true. The rest was
not:

| | before | after |
|---|---|---|
| `import src.embed_store` | **+2602**, open3d and sklearn resident | **+201** |
| `import src.pose` | +2599 | **+196** |
| `import cluster_models` | +2649 | **+1639**, open3d gone |

The `pose` "before" is reconstructed after the fix — `import open3d, src.pose`
in one interpreter — since the module-scope version is in history (`2791cd1^`);
the other rows were measured on the code as it stood.

A tool that reads `.npy` files off disk was loading a rendering library. It
arrived through `from src import pose` at module scope, and `embed_store` uses
exactly three names from `pose` — `load_pose_cache`, `embed_cache_token`,
`file_identity` — all of which are pure: stdlib and json, no `np.`, no `o3d`.

## The fix that looked obvious was the wrong one

Moving those three pure functions into `src/identity.py` was the natural
inference and would have been a large diff for a fraction of the win.
Decomposing `pose`'s three heavy imports first is what changed the answer:

| import in `src/pose.py` | modules |
|---|---|
| open3d | **2596** |
| numpy | 130 |
| PIL | 3 |

open3d was 2596 of the 2602, and it is used in exactly **one** function,
`up_axis_scores`. The whole cost was one import serving one caller. Deferring
it there (`2791cd1`) took `pose` to +196 without moving a single name across
the 37 Python files that import it. The sklearn in the earlier measurement
turned out to be arriving through open3d; what `cluster_models` still loads is
its own,
which is correct for a clustering tool.

## The rule, and why it is not "never defer"

Deferring open3d into `up_axis_scores` is the same mechanical move that made
`add_cache_args` wrong. The property that separates them:

> A deferred import removes cost only when the function's own **signature
> already implies the dependency**. Then the set of newly-charged callers is
> empty, because none of them could have called it without the dependency in
> hand.

`up_axis_scores(mesh)` qualifies — you cannot hold an Open3D mesh without
having imported open3d. `DEFAULT_MODEL` did not: it handed back a *string*, so
every caller was by construction a caller that did not already have torch.

This is a review-time test on the argument types, not a property of a module.
`pose` passes it today because `up_axis_scores` is its only open3d user; a
future function there taking a *path* and reaching for open3d would fail the
rule while the deferred import sat there looking compliant. `interfaces.md`'s
import table states the rule in that form, and names `up_axis_scores` as the
only function in `pose` that satisfies it, so the next one lands at review.

## The counter-case worth keeping

`src/driver.py`'s `spawn_render_child` defers `from src.render_child import
run_child` with the comment "so the parent does not pull
`open3d.visualization.rendering` in just to spawn". That claim is false: the
import runs *in* the parent, in the only function that spawns, on every run —
and `classify_stls.main()` has already imported `src.renderer` for
`RENDER_FORMATS` a hundred lines earlier. The deferral has a real beneficiary,
just not the stated one: `tests/test_driver.py` and anything importing
`src.driver` without spawning. Import kept, comment corrected.

So the failure mode is not only "deferral that relocates cost". It is also
"deferral whose rationale names a beneficiary that pays anyway" — and the two
look identical until someone measures.

## How it was found

By measuring, not by reading. All three docstrings asserted lightness, in
prose, next to code that contradicted them; the assertions were what made the
modules look reviewed. `python -c "import sys; b=len(sys.modules); import X;
print(len(sys.modules)-b)"` is the whole method, and it is worth running on any
module whose docstring makes a claim about weight — including after a refactor
that was *about* import discipline, which is when these two were introduced.
