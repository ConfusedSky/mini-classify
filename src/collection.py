"""The in-memory index a query answers against: embeddings, poses, and the
scope resolver (docs/api/implementation.md phase 1).

This is the object an API handler asks questions of, and it is deliberately
*not* the API: no HTTP, no pydantic, no torch. It owns the load preamble
`test_categories.py` and `cluster_models.py` already open with — `cache_root`,
`load_file_list`, `load_embedding_matrix`, `load_pose_cache`, `view_config` —
so the server does not become a third copy of it.

**Nothing here touches the filesystem after `load`.** That is the load-bearing
property, not an optimisation: the collection lives on spinning exfat over USB
where model-browser measured a cold tree walk at ~32 s, and two processes
walking one platter contend for the head (docs/api/surface.md §scope). So the
walk is the *classify run's*, read from its cached file list; every per-file
identity and display name is computed once at load; and `resolve` answers from
precomputed path tuples. The one exception is the existence check in `resolve`,
which is a single `stat` of the scope path — the I/O budget for a 404.

`reload` returns a **new instance** rather than mutating this one, which is
what lets the server rebind a name and never lock a reader (implementation.md
phase 2: bind once at handler entry).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import identity, pose
from src.cachedir import (cache_root, load_file_list, total_views, view_config)
from src.embed_store import load_embedding_matrix

# What `classify_stls.py` walks (`cachedir.find_stls`), and therefore the only
# thing this index can ever hold. Published in every scope block because
# model-browser lists .3mf and .obj too: without it a folder of .3mf reports
# nothing scanned and nothing missing, which reads as "fully covered" when it
# means "not searchable at all" (surface.md §scope).
COVERS = ["stl"]


class ScopeError(Exception):
    """A scope that cannot be answered. Subclasses carry *why*, because the
    three reasons are three different HTTP answers and collapsing them is what
    makes a UI say "nothing matched" when it should say "not indexed"."""


class VirtualPath(ScopeError):
    """A zip virtual path (`foo.zip!/entry`). Not addressable here at all —
    the classifier walks real files (surface.md §path space). 422."""


class OutsideCollection(ScopeError):
    """A real path, but not under `collection_root`. 400."""


class NoSuchPath(ScopeError):
    """Nothing on disk at that path. 404 — and distinct from a real directory
    with nothing classified in it, which is a 200."""


@dataclass(frozen=True)
class Scope:
    """What a path filter matched, and what it could not.

    `rows` indexes `Collection.matrix`; slicing before scoring is what narrows
    the robust z along with the results (`src/query.py`'s `score`).

    `status` is the tri-state the UI needs: `indexed` (everything the last
    classify run saw here is embedded), `partial` (some is not — run
    classify_stls.py), `unindexed` (a real directory with nothing embedded,
    which is an answer and not an error)."""
    path: str | None                # as given; None for the whole collection
    rows: np.ndarray
    n_indexed: int
    n_scanned: int
    covers: list[str]

    @property
    def status(self) -> str:
        if self.n_indexed == 0:
            return "unindexed"
        return "indexed" if self.n_indexed >= self.n_scanned else "partial"

    def as_dict(self) -> dict:
        return {"path": self.path, "status": self.status,
                "n_indexed": self.n_indexed, "n_scanned": self.n_scanned,
                "covers": list(self.covers)}


class Collection:
    """One loaded cache: the matrix, the files behind its rows, and the poses.

    Construct with `Collection.load(args)`; `args` is the shared cache-identity
    block (`cachedir.add_cache_args`), so this agrees with the classifier about
    which cache it is reading by construction."""

    def __init__(self, args, root, files, scanned, matrix, poses, missing):
        self.args = args
        self.root = root                    # collection_root: anchor and display base
        self.files = files                  # aligned with matrix rows
        self.scanned = scanned              # the classify run's cached walk
        self.matrix = matrix                # (n_files, n_views, dim) float32
        self.poses = poses
        self.missing = missing              # walked, not embedded
        self.view_cfg = view_config(args)   # keys front_view entries
        self.n_views = total_views(args)
        self._angles = pose.view_angles(args.views, list(args.elevations))
        # Everything below is why a request needs no filesystem: paths as
        # tuples for prefix matching, identities for the pose lookup (one stat
        # each, here rather than per hit), and the display name.
        self._rel = [self._parts(f) for f in files]
        self._scanned_rel = [self._parts(f) for f in scanned]
        self._ident = [pose.file_identity(f, root) for f in files]
        self._names = [self._display_name(f) for f in files]

    # --- loading ------------------------------------------------------------

    @classmethod
    def load(cls, args) -> "Collection":
        """The preamble, once. Prints the file-list note `load_file_list`
        always prints — a server logs it at startup and on reload, nowhere
        else."""
        inp = Path(args.input)
        root = cache_root(inp, args.cache_dir, confirm=False)
        scanned = load_file_list(inp, args.cache_dir, args.rescan)
        matrix, files, missing = load_embedding_matrix(scanned, args, root)
        poses = pose.load_pose_cache(args.cache_dir)
        return cls(args, root, files, scanned, matrix, poses, missing)

    def reload(self, rescan: bool = False) -> "Collection":
        """A *new* Collection from the same args. The caller rebinds; readers
        holding this one keep a consistent view for the rest of their request
        (implementation.md phase 2)."""
        args = _with(self.args, rescan=rescan)
        return Collection.load(args)

    # --- scoping ------------------------------------------------------------

    def resolve(self, path: str | None) -> Scope:
        """A path to the rows under it, plus what the caller must be told.

        Accepts absolute or root-relative, directory or file. `None` is the
        whole collection. Raises `VirtualPath`, `OutsideCollection` or
        `NoSuchPath` — see each for why they are not one error."""
        if path is None:
            return Scope(None, np.arange(len(self.files)), len(self.files),
                         len(self.scanned), COVERS)
        if "!/" in str(path):
            raise VirtualPath(f"zip entries are not indexed: {path}")

        p = Path(path)
        p = p if p.is_absolute() else self.root / p
        # realpath both sides, never a string prefix: the library is on
        # removable media and remounts under a different name
        # (/run/media/.../STLLibrary vs ...STLLibrary1) for the same tree,
        # where prefix equality silently stops matching (surface.md §status).
        try:
            real = p.resolve()
            root = self.root.resolve()
        except OSError as e:                       # broken symlink loop, etc.
            raise NoSuchPath(f"cannot resolve {path}: {e}") from e
        if real != root and not real.is_relative_to(root):
            raise OutsideCollection(f"{path} is not under {self.root}")
        if not real.exists():                      # the one stat of a request
            raise NoSuchPath(f"no such path: {path}")

        want = real.relative_to(root).parts
        n = len(want)
        rows = np.array([i for i, rel in enumerate(self._rel) if rel[:n] == want],
                        dtype=np.intp)
        n_scanned = sum(1 for rel in self._scanned_rel if rel[:n] == want)
        return Scope(str(path), rows, len(rows), n_scanned, COVERS)

    # --- per-model detail ---------------------------------------------------

    def pose_of(self, i: int) -> dict | None:
        """The pose block for row `i`, or None when nothing is resolved.

        `azimuth_zero` is the model-space direction azimuth 0 is measured from,
        so a viewer that cannot rotate a mesh derives its own azimuth offset
        from data rather than reimplementing `rotation_to_z_up`
        (surface.md §pose). `front` is null whenever the pose cache holds no
        `front_view` for *this* view config — a real state, since an index
        cached at 8 views means nothing at 4."""
        entry = self.poses.get(self._ident[i])
        if not entry:
            return None
        up = [float(x) for x in entry["up"]]
        R = pose.rotation_to_z_up(np.array(up))
        block = {
            "up": up,
            "azimuth_zero": [float(x) for x in R.T @ np.array([1.0, 0.0, 0.0])],
            "source": entry.get("source"),
            "confidence": float(entry.get("confidence", 0.0)),
            "front": None,
        }
        view = pose.front_view(entry, self.view_cfg)
        if view is not None and 0 <= view < len(self._angles):
            az, el = self._angles[view]
            block["front"] = {"view": int(view),
                              "azimuth_deg": float(np.rad2deg(az)),
                              "elevation_deg": float(np.rad2deg(el))}
        return block

    def hit(self, i: int, score: float, z: float) -> dict:
        """One result row in the API's `hit` shape (surface.md §hit).

        `rel_path` is the join key a consumer uses against its own tree, and
        no path is validated here: two independent caches of one tree drift by
        design, and checking would mean a stat per hit on the volume this
        module refuses to walk."""
        f = self.files[i]
        return {"id": identity.render_key(f, self.root),
                "path": str(f),
                "rel_path": "/".join(self._rel[i]),
                "name": self._names[i],
                "score": float(score),
                "z": float(z),
                "pose": self.pose_of(i)}

    # --- helpers ------------------------------------------------------------

    def _parts(self, f: Path) -> tuple:
        """Path relative to the collection root, as parts. Files outside the
        root keep their absolute parts, so they simply match no scope rather
        than raising during a load."""
        return f.relative_to(self.root).parts if f.is_relative_to(self.root) else f.parts

    def _display_name(self, f: Path) -> str:
        """The REPL's rule, one copy: root-relative, filler directory dropped,
        extension dropped (`test_categories.py`)."""
        rel = str(f.relative_to(self.root)) if f.is_relative_to(self.root) else str(f)
        return rel.replace("/No Supports", "").removesuffix(".stl")


def _with(args, **over):
    """A shallow copy of an argparse.Namespace with fields replaced — the
    Namespace is the cache identity and must not be mutated under a reader."""
    import argparse
    return argparse.Namespace(**{**vars(args), **over})
