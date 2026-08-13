import os

from restore_mtimes import compare, manifest, plan

ORIGINAL = 1_634_572_888_000_000_000     # what the source carries
COPIED = 1_786_656_855_794_968_843       # what a plain cp stamped on


def tree(root, files=("Kit/model.stl", "Kit/other.stl"), mtime=ORIGINAL):
    for rel in files:
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"solid")
        os.utime(f, ns=(mtime, mtime))
    for d in sorted(root.rglob("*"), key=lambda p: -len(p.parts)):
        if d.is_dir():
            os.utime(d, ns=(mtime, mtime))
    return root


def test_a_copy_without_dash_a_needs_every_file_restored(tmp_path):
    src = tree(tmp_path / "src")
    dst = tree(tmp_path / "dst", mtime=COPIED)
    files, dirs = plan(manifest(src), manifest(dst))
    assert sorted(files) == ["Kit/model.stl", "Kit/other.stl"] and dirs == ["Kit"]


def test_directories_come_deepest_first(tmp_path):
    # writing a file updates its parent's mtime, so a shallow directory
    # restored first would be undone by the children restored after it
    src = tree(tmp_path / "src", ("a/b/c/deep.stl", "a/shallow.stl"))
    dst = tree(tmp_path / "dst", ("a/b/c/deep.stl", "a/shallow.stl"), mtime=COPIED)
    _, dirs = plan(manifest(src), manifest(dst))
    assert dirs == ["a/b/c", "a/b", "a"]


def test_an_already_good_copy_needs_nothing(tmp_path):
    src = tree(tmp_path / "src")
    dst = tree(tmp_path / "dst")
    assert plan(manifest(src), manifest(dst)) == ([], [])


def test_restoring_makes_the_manifests_agree(tmp_path):
    src = tree(tmp_path / "src")
    dst = tree(tmp_path / "dst", mtime=COPIED)
    ms, md = manifest(src), manifest(dst)
    files, dirs = plan(ms, md)
    for rel in files + dirs:
        os.utime(dst / rel, ns=(ms[rel][2], ms[rel][3]))
    assert plan(manifest(src), manifest(dst)) == ([], [])


# --- the checks that stop it running on the wrong tree -----------------------

def test_a_missing_file_is_reported(tmp_path):
    src = tree(tmp_path / "src")
    dst = tree(tmp_path / "dst", ("Kit/model.stl",), mtime=COPIED)
    missing, extra, size_bad = compare(manifest(src), manifest(dst))
    assert missing == ["Kit/other.stl"] and not extra and not size_bad


def test_a_truncated_file_is_reported(tmp_path):
    # the case that matters: same path, fewer bytes. Restoring the timestamp
    # would make a half-copied file look untouched since the original.
    src = tree(tmp_path / "src")
    dst = tree(tmp_path / "dst", mtime=COPIED)
    (dst / "Kit/model.stl").write_bytes(b"sol")
    _, _, size_bad = compare(manifest(src), manifest(dst))
    assert size_bad == ["Kit/model.stl"]


def test_an_extra_file_is_reported(tmp_path):
    src = tree(tmp_path / "src")
    dst = tree(tmp_path / "dst", ("Kit/model.stl", "Kit/other.stl", "Kit/spare.stl"),
               mtime=COPIED)
    _, extra, _ = compare(manifest(src), manifest(dst))
    assert extra == ["Kit/spare.stl"]


def test_the_filesystems_own_directory_is_not_an_extra_file(tmp_path):
    # ext4 makes lost+found at the root of every volume; counting it as extra
    # would stop every drive-to-drive restore before it started
    src = tree(tmp_path / "src")
    dst = tree(tmp_path / "dst", mtime=COPIED)
    (dst / "lost+found").mkdir()
    missing, extra, size_bad = compare(manifest(src), manifest(dst))
    assert not missing and not extra and not size_bad


def test_a_lost_found_deeper_in_the_tree_is_still_an_extra_file(tmp_path):
    # only the volume root gets the exemption; a directory a user made is theirs
    src = tree(tmp_path / "src")
    dst = tree(tmp_path / "dst", mtime=COPIED)
    (dst / "Kit" / "lost+found").mkdir()
    _, extra, _ = compare(manifest(src), manifest(dst))
    assert extra == ["Kit/lost+found"]
