#!/usr/bin/env python3
"""Build a complete Code Sense batch archive and a separate reviewer archive."""
import argparse
import csv
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
TASK = "CS-XX-003"
UPSTREAM = "environment/upstream/lm-evaluation-harness"


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="directory for the two archives")
    parser.add_argument("--rev", default="HEAD", help="author repository commit or tag")
    args = parser.parse_args()
    sha = git("rev-parse", "--verify", args.rev + "^{commit}").decode().strip()
    meta = tomllib.loads(git("show", f"{sha}:task.toml").decode())["metadata"]
    if meta["id"] != TASK or meta["prompt"] != git("show", f"{sha}:instruction.md").decode():
        parser.error("metadata.id or metadata.prompt differs from the selected task")
    manifest = list(csv.DictReader(io.StringIO(git("show", f"{sha}:manifest.csv").decode())))
    if len(manifest) != 1 or (manifest[0]["id"], manifest[0]["version"]) != (TASK, meta["version"]):
        parser.error("manifest.csv differs from task.toml")
    pin = git("rev-parse", f"{sha}:{UPSTREAM}").decode().strip()
    if not (ROOT / UPSTREAM / ".git").exists():
        parser.error("initialize environment/upstream/lm-evaluation-harness with git submodule update --init")
    date = git("show", "-s", "--format=%cs", sha).decode().strip().replace("-", "")
    prefix = f"CS-XX-pilot-{date}"
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = [args.output / f"{prefix}{suffix}.tar.gz" for suffix in ("", "-external")]
    if any(output.exists() for output in outputs):
        parser.error("archive already exists; choose a new output directory")

    with tempfile.TemporaryDirectory(dir=args.output) as tmp:
        tmp = Path(tmp)
        payload = tmp / prefix / TASK
        payload.mkdir(parents=True)
        external = TASK + "-external"
        entries = ["instruction.md", "task.toml", "environment", "solution", "tests", external, "manifest.csv"]
        # git archive takes files only from the selected commit, independent of
        # any local scores, solver edits, or scratch checkouts.
        with tempfile.TemporaryFile() as frozen:
            subprocess.run(["git", "-C", str(ROOT), "archive", sha, *entries],
                           check=True, stdout=frozen)
            frozen.seek(0)
            with tarfile.open(fileobj=frozen) as archive:
                archive.extractall(payload, filter="data")
        (payload / "manifest.csv").rename(payload.parent / "manifest.csv")
        review = tmp / (prefix + "-external")
        review.mkdir()
        (payload / external).rename(review / external)
        env = payload / "environment"
        source_head = subprocess.check_output(
            ["git", "-C", str(ROOT / UPSTREAM), "rev-parse", "HEAD"], text=True
        ).strip()
        if source_head != pin:
            parser.error(f"source checkout differs from the selected gitlink {pin}")
        subprocess.run([sys.executable, str(env / "prepare_source.py"),
                        "--source", str(ROOT / UPSTREAM), "--dest", str(env / "workspace/repo")],
                       check=True)
        # The handoff carries materialized source and notices. Exclude author
        # bootstrap scripts and patches from the build context.
        (env / "starter.patch").unlink()
        (env / "prepare_source.py").unlink()
        shutil.rmtree(env / "upstream")
        for tree, filename in [(payload.parent, "batch.tar.gz"), (review, "review.tar.gz")]:
            with tarfile.open(tmp / filename, "w:gz", compresslevel=6) as archive:
                archive.add(tree, arcname=tree.name)
        for filename, output in zip(("batch.tar.gz", "review.tar.gz"), outputs):
            (tmp / filename).rename(output)
    print(f"commit {sha}")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
