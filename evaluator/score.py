"""Run the local private suite against a candidate runtime.

The scorer is intentionally a thin subprocess wrapper.  It accepts a repo
directory so the same private tests can be run against the NOP checkout, the
golden checkout, and a model-modified checkout without mutating any of them.
No network access or external service is used.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


SUMMARY_RE = re.compile(r"(?:(\d+) passed)?(?:, )?(?:(\d+) failed)?")


def _collect_count(python: str, root: Path, test_file: Path, env: dict[str, str]) -> int:
    proc = subprocess.run(
        [python, "-m", "pytest", "--collect-only", "-q", str(test_file)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    # pytest emits one node id per line and a final ``N tests collected`` line.
    match = re.search(r"(\d+)\s+tests?\s+collected", proc.stdout)
    if match:
        return int(match.group(1))
    return sum("::test_" in line for line in proc.stdout.splitlines())


def run(repo: Path, test_file: Path, *, python: str = sys.executable) -> dict[str, Any]:
    repo = repo.resolve()
    test_file = test_file.resolve()
    root = test_file.parent.parent
    env = os.environ.copy()
    old_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repo) + (os.pathsep + old_path if old_path else "")
    total = _collect_count(python, root, test_file, env)
    try:
        proc = subprocess.run(
            [python, "-m", "pytest", "-q", str(test_file), "--disable-warnings"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {
            "valid": False,
            "score": 0,
            "ut_total": total,
            "ut_passed": 0,
            "pytest_exit_code": None,
            "error": str(exc),
        }
    passed = 0
    failed = 0
    summary = SUMMARY_RE.search(proc.stdout.splitlines()[-1] if proc.stdout else "")
    if summary:
        passed = int(summary.group(1) or 0)
        failed = int(summary.group(2) or 0)
    valid = proc.returncode in {0, 1} and total > 0
    return {
        "valid": valid,
        "score": int(valid and proc.returncode == 0),
        "ut_total": total,
        "ut_passed": passed,
        "ut_failed": failed,
        "pytest_exit_code": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-3000:],
        "repo": str(repo),
        "tests": str(test_file),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument(
        "--tests",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests" / "test_batched_eval_private.py",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = run(args.repo, args.tests, python=args.python)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.out:
        args.out.write_text(text)
    return 0 if result.get("valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
