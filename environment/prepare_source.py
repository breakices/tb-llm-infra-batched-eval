"""Materialize the offline solver workspace from the pinned upstream commit."""

import argparse
import errno
import filecmp
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream" / "lm-evaluation-harness"
COMMIT = "ddd67220430a2470529f25fd5c05a576ca1057a0"
TREE = "b2c54aacf87ea3fe55e790cc9ad00cdba925833e"


def same_tree(left, right):
    """Compare contents, file types, and executable bits, including dotfiles."""
    names = set(os.listdir(left))
    if names != set(os.listdir(right)):
        return False
    for name in names:
        a, b = left / name, right / name
        amode, bmode = a.lstat().st_mode, b.lstat().st_mode
        if stat.S_IFMT(amode) != stat.S_IFMT(bmode):
            return False
        if stat.S_ISDIR(amode):
            if not same_tree(a, b):
                return False
        elif stat.S_ISLNK(amode):
            if os.readlink(a) != os.readlink(b):
                return False
        elif not stat.S_ISREG(amode) or bool(amode & 0o111) != bool(bmode & 0o111):
            return False
        elif not filecmp.cmp(a, b, shallow=False):
            return False
    return True


def prepare(source, dest):
    # A submodule's working tree can contain local edits. Export its immutable
    # object instead, so such edits cannot enter a published solver workspace.
    if not source.is_dir():
        raise ValueError("initialize environment/upstream/lm-evaluation-harness with git submodule update --init")
    git = ["git", "-C", str(source), "--no-replace-objects"]
    tree = subprocess.check_output([*git, "rev-parse", "--verify", f"{COMMIT}^{{tree}}"], text=True).strip()
    if tree != TREE:
        raise ValueError(f"unexpected source tree {tree} for {COMMIT}")
    patch = HERE / "starter.patch"
    if not patch.is_file():
        raise ValueError(f"missing starter patch: {patch}")
    if dest.is_symlink() or (dest.exists() and not dest.is_dir()):
        raise ValueError(f"destination must be a directory: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".codesense-build-", dir=dest.parent) as tmp:
        work = Path(tmp)
        staged = work / "repo"
        staged.mkdir()
        archive = work / "source.tar"
        subprocess.run([*git, "archive", "--format=tar", "-o", str(archive), COMMIT], check=True)
        subprocess.run(["tar", "-xf", str(archive), "-C", str(staged)], check=True)
        # Do not let git interpret the patch relative to the author repository.
        env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(work)}
        subprocess.run(["git", "apply", "--no-index", str(patch)], cwd=staged, env=env, check=True)
        try:
            staged.rename(dest)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            if dest.is_symlink() or not same_tree(staged, dest):
                raise ValueError(f"{dest} contains other files or changes; choose a fresh destination") from exc
    print(f"Ready: {dest} (upstream {COMMIT})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=UPSTREAM,
                        help="Git repository containing the pinned upstream object")
    parser.add_argument("--dest", type=Path, default=Path("repo"),
                        help="new, empty, or already identical solver workspace (default: repo)")
    args = parser.parse_args()
    try:
        prepare(args.source.resolve(), args.dest.absolute())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Cannot prepare workspace: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
