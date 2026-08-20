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
identity, render key and display name is computed once at load; and `resolve`
answers from precomputed path tuples.

To be exact about the budget, since an earlier version of this docstring
claimed "a single stat" and was wrong by an order of magnitude (review,
2026-08-19): **`pose_of` and `hit` do no I/O at all**, and `resolve` costs one
`Path.resolve()` plus one `exists()` on the scope path — which is a handful of
`lstat`s, proportional to that path's *depth* and never to the size of the
collection. `tests/test_collection.py` asserts both as budgets, because the
first version of that guard patched four walk functions and missed `hit`
resolving a path nine times per result.

`reload` returns a **new instance** rather than mutating this one, which is
what lets the server rebind a name and never lock a reader (implementation.md
phase 2: bind once at handler entry).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import identity, pose
from src.cachedir import (cache_root, load_file_list, require_cache_version,
                          total_views, view_config)
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


class CacheUnusable(Exception):
    """The cache is present but cannot answer queries.

    Exists to keep `SystemExit` out of a request handler. `embed_store` and
    `cachedir.require_cache_version` both raise it — correct for a CLI, where
    it prints and exits — but `SystemExit` is a `BaseException`, so it walks
    straight through Starlette's `except Exception` and never becomes a 500.
    `POST /reload` runs this code inside a handler (implementation.md phase 2).

    It is also the same inconsistency `VolumeUnavailable` was created to fix:
    an empty *scope* is a 200 with `status: "unindexed"`, so an empty
    *collection* should not be a process exit.

    `hint` carries the actionable line where there is one — a version mismatch
    wants `migrate_cache_keys.py`, not `classify_stls.py`."""

    def __init__(self, message, hint=None):
        self.message, self.hint = str(message), hint
        super().__init__(f"{message}" + (f"\n  {hint}" if hint else ""))

    def as_dict(self) -> dict:
        return {"usable": False, "reason": self.message, "hint": self.hint}


class VolumeUnavailable(Exception):
    """The collection's storage is not there — typically the library's USB
    volume is unmounted.

    Raised instead of letting the load fall through, because falling through
    is *misleading*: `load_file_list` drops every entry whose file is missing
    and `load_embedding_matrix` then reports "no cached embeddings found — run
    classify_stls.py first", which is the wrong advice twice over. The
    embeddings are intact and local; nothing needs re-running; the drive needs
    plugging in.

    Deliberately not a degraded load. The file identity is `rel|mtime|size`,
    so serving without the volume would mean trusting the pose cache's keys
    over the filesystem, and this project would rather refuse than answer from
    a snapshot it cannot check.

    Carries `as_dict()` in the same shape as `Collection.volume`, so a server
    reports the absence through the same field it reports the presence
    (docs/api/surface.md §`GET /status`) rather than inventing a second."""

    def __init__(self, root, missing, what="collection volume"):
        self.root = Path(root)
        self.missing = Path(missing)
        super().__init__(f"{what} is not available: {missing}")

    def as_dict(self) -> dict:
        return {"present": False, "root": str(self.root),
                "missing": str(self.missing)}


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
        # The root, resolved once. `resolve()` compared realpaths by calling
        # `Path.resolve()` on *both* sides per request, which lstats every
        # component of each — and the root never changes (review, 2026-08-19).
        self._real_root = _real(root)
        # Files come from walking `args.input`, which may be the root, a
        # subdirectory of it, or a symlink to either. Relating the two once is
        # what lets every path below be pure tuple arithmetic — and what stops
        # a symlinked input from producing absolute `_rel` tuples, where every
        # scope silently matched nothing and `rel_path` came out with a doubled
        # leading slash (review, 2026-08-19).
        self._inp = Path(args.input)
        try:
            self._prefix = _real(self._inp).relative_to(self._real_root).parts
        except ValueError:
            self._prefix = ()
        # Everything below is why a request needs no filesystem: paths as
        # tuples for prefix matching, identities for the pose lookup, and the
        # render key and display name — all one stat each here rather than per
        # hit. `render_key` is the expensive one: it resolves the path, so
        # leaving it in `hit` cost 9 syscalls per result.
        self._rel = [self._parts(f) for f in files]
        self._scanned_rel = [self._parts(f) for f in scanned]
        self._ident = [pose.file_identity(f, root) for f in files]
        self._keys = [identity.render_key(f, root) for f in files]
        self._names = [self._display_name(f) for f in files]

    # --- loading ------------------------------------------------------------

    @classmethod
    def load(cls, args) -> "Collection":
        """The preamble, once. Prints the file-list note `load_file_list`
        always prints — a server logs it at startup and on reload, nowhere
        else.

        Raises `VolumeUnavailable` when the collection's storage is missing,
        before the walk rather than after it: the walk's own failure mode is a
        silent zero-file result that reads as an empty cache."""
        inp = Path(args.input)
        root = cache_root(inp, args.cache_dir, confirm=False)
        cls._require_volume(root, inp, args.cache_dir)
        # Both of these exit the process on failure, which is right for a CLI
        # and wrong inside a request handler — see `CacheUnusable`. The version
        # guard is the one every other cache consumer calls
        # (classify_stls.py, test_categories.py): without it a cache written
        # under an older key scheme misses on every lookup and reports
        # "run classify_stls.py first", when the fix is migrate_cache_keys.
        try:
            require_cache_version(args.cache_dir)
            scanned = load_file_list(inp, args.cache_dir, args.rescan)
            matrix, files, missing = load_embedding_matrix(scanned, args, root)
        except SystemExit as e:
            text = str(e)
            hint = next((ln.strip() for ln in text.splitlines()
                         if ln.strip().startswith("run:")), None)
            raise CacheUnusable(text.split("\n")[0], hint) from e
        poses = pose.load_pose_cache(args.cache_dir)
        return cls(args, root, files, scanned, matrix, poses, missing)

    @classmethod
    def load_with(cls, args, **over) -> "Collection":
        """`load` with fields replaced, without needing an instance first.

        The server's retry path: a load that failed leaves nothing to call
        `reload` on, and a process that told the user to mount the drive and
        try again has to have somewhere for them to try (review, 2026-08-19)."""
        return cls.load(_with(args, **over))

    def reload(self, rescan: bool = False) -> "Collection":
        """A *new* Collection from the same args. The caller rebinds; readers
        holding this one keep a consistent view for the rest of their request
        (implementation.md phase 2)."""
        args = _with(self.args, rescan=rescan)
        return Collection.load(args)

    @staticmethod
    def _require_volume(root: Path, inp: Path, cache_dir) -> None:
        """Two stats, and the console line that explains a failure nobody
        should have to diagnose from "run classify_stls.py first".

        The root is checked before the input because they fail for opposite
        reasons: the root missing means the storage is gone, while the root
        present and the input missing means the library is mounted and the
        directory this run is scoped to has been moved or deleted."""
        for path, what in ((root, "collection volume"), (inp, "input path")):
            if path.exists():
                continue
            print(f"\n{what} is not available: {path}")
            if what == "collection volume":
                print(f"  the caches in {cache_dir} are intact and local — nothing "
                      f"needs re-running.\n"
                      f"  mount the volume and retry. File identities are "
                      f"rel|mtime|size, so the\n"
                      f"  files have to be present to key what is already cached.")
            else:
                print(f"  the volume at {root} is mounted, but this run's input "
                      f"directory is gone.\n"
                      f"  check the path, or point at the collection root.")
            raise VolumeUnavailable(root, path, what)

    @property
    def volume(self) -> dict:
        """What `/status` reports about the storage. Present by construction —
        a Collection cannot be loaded without it — so the interesting case is
        the exception's `as_dict()`, which carries the same keys."""
        return {"present": True, "root": str(self.root), "missing": None}

    # --- scoping ------------------------------------------------------------

    def resolve(self, path: str | None) -> Scope:
        """A path to the rows under it, plus what the caller must be told.

        Accepts absolute or root-relative, directory or file. `None` is the
        whole collection. Raises `VirtualPath`, `OutsideCollection` or
        `NoSuchPath` — see each for why they are not one error."""
        if path is None:
            return Scope(None, np.arange(len(self.files)), len(self.files),
                         len(self.scanned), list(COVERS))
        if "!/" in str(path):
            raise VirtualPath(f"zip entries are not indexed: {path}")

        # realpath, never a string prefix: the library is on removable media
        # and remounts under a different name (/run/media/.../STLLibrary vs
        # ...STLLibrary1) for the same tree, where prefix equality silently
        # stops matching (surface.md §status). The root half is precomputed.
        #
        # Everything here runs on a request-body string, so the catch is wide
        # on purpose. `Path.resolve()` raises RuntimeError on a symlink loop
        # (not OSError — it swallows that internally), ValueError on an
        # embedded null, OSError on an over-long name, and TypeError on a
        # non-string; an earlier version caught only OSError and named symlink
        # loops in the comment, which was the one case it could not catch
        # (review, 2026-08-19). All of them mean the same thing to a caller:
        # that is not a usable path.
        try:
            p = Path(path)
            p = p if p.is_absolute() else self.root / p
            real = _real(p)
            exists = real.exists()
        except (OSError, ValueError, RuntimeError, TypeError) as e:
            raise NoSuchPath(f"unusable path {path!r}: {e}") from e
        if not real.is_relative_to(self._real_root):
            raise OutsideCollection(f"{path} is not under {self.root}")
        if not exists:                             # the one stat of a request
            raise NoSuchPath(f"no such path: {path}")

        want = real.relative_to(self._real_root).parts
        n = len(want)
        rows = np.array([i for i, rel in enumerate(self._rel) if rel[:n] == want],
                        dtype=np.intp)
        n_scanned = sum(1 for rel in self._scanned_rel if rel[:n] == want)
        return Scope(str(path), rows, len(rows), n_scanned, list(COVERS))

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
        # A malformed entry degrades to "no pose", never to a raised
        # exception: `load_pose_cache` filters on `v` and validates no shape,
        # pose-cache.json is hand-editable, and one bad entry must not fail a
        # whole query response when this module already treats a null pose as
        # a real state (review, 2026-08-19). `confidence: null` and a missing
        # `up` are the two that occur.
        up = pose.entry_up(entry)
        if up is None:
            return None
        up = list(up)
        R = pose.rotation_to_z_up(np.array(up))
        try:
            confidence = float(entry.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        block = {
            "up": up,
            "azimuth_zero": [float(x) for x in R.T @ np.array([1.0, 0.0, 0.0])],
            "source": entry.get("source"),
            "confidence": confidence,
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
        return {"id": self._keys[i],            # precomputed: render_key resolves
                "path": str(f),
                "rel_path": "/".join(self._rel[i]),
                "name": self._names[i],
                "score": float(score),
                "z": float(z),
                "pose": self.pose_of(i)}

    # --- helpers ------------------------------------------------------------

    def _parts(self, f: Path) -> tuple:
        """Path relative to the collection root, as parts.

        Routed through the *input* rather than the root, because the walk that
        produced `f` started at the input: when the two differ only by a
        symlink, `f.relative_to(root)` fails and the old fallback kept the
        absolute parts — which matched no scope and emitted a `rel_path` with
        a doubled leading slash, the documented join key silently unusable
        (review, 2026-08-19). `_prefix` is the input's offset from the root, so
        this stays exact for a run scoped to a subdirectory too."""
        try:
            return self._prefix + f.relative_to(self._inp).parts
        except ValueError:
            pass
        return f.relative_to(self.root).parts if f.is_relative_to(self.root) else f.parts

    def _display_name(self, f: Path) -> str:
        """The REPL's rule, one copy: root-relative, filler directory dropped,
        extension dropped (`test_categories.py`)."""
        rel = "/".join(self._parts(f))
        return rel.replace("/No Supports", "").removesuffix(".stl")


def _real(p: Path) -> Path:
    """`Path.resolve()`, named so the cost is visible at the call site: it
    lstats every component, which is why the root's is computed once at load
    and never per request."""
    return Path(p).resolve()


def _with(args, **over):
    """A copy of an argparse.Namespace with fields replaced — the Namespace is
    the cache identity and must not be mutated under a reader.

    `elevations` is copied rather than aliased: it is the one mutable field,
    and two Collections sharing a list is the kind of aliasing that only shows
    up once something sorts it in place."""
    import argparse
    d = {**vars(args), **over}
    if isinstance(d.get("elevations"), list):
        d["elevations"] = list(d["elevations"])
    return argparse.Namespace(**d)
