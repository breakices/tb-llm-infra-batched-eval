"""Offline score entrypoint for the local batched-eval candidate.

Usage is ``python tests/score.py <repo> [result.json]``.  The verifier runs
only local pytest code and the dependency-free oracle shipped beside it.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import re


def _collected_count(selected, *, env, cwd):
    """Count parametrized test cases from pytest's collection output.

    Counting only ``test_*`` function definitions undercounts a matrix-heavy
    private suite.  Collection runs against the fixed verifier files and
    therefore reports the actual number of rows executed without trusting
    candidate code or a hand-maintained constant.
    """

    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *(str(path) for path in selected)],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    match = re.search(r"(\d+)\s+tests?\s+collected", collected.stdout)
    return int(match.group(1)) if match else 0


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: score.py <repo> [result.json]")
    repo = Path(sys.argv[1]).resolve()
    tests = Path(__file__).resolve().parent
    selected = [
        tests / "test_llm_runtime.py",
        tests / "test_batched_eval_private.py",
        tests / "test_tiny_evaluator_private.py",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        # Keep the evaluator and oracle from the task checkout ahead of a
        # candidate repo that happens to ship a module with the same name.
        str(path) for path in (tests.parent, tests, repo) if path.exists()
    )
    selfcheck = subprocess.run(
        [sys.executable, "-m", "evaluator.selfcheck"],
        cwd=tests.parent,
        env=env,
        capture_output=True,
        text=True,
    )
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *(str(path) for path in selected)],
        env=env,
        cwd=tests.parent,
        capture_output=True,
        text=True,
    )
    total = _collected_count(selected, env=env, cwd=tests.parent)
    passed = 0
    # Pytest's summary is intentionally not used as the source of truth for
    # validity. A non-zero exit code or a missing test file is always invalid.
    if proc.returncode == 0:
        passed = total
    valid = proc.returncode in (0, 1) and selfcheck.returncode == 0 and total > 0
    result = {
        "valid": valid,
        "score": passed / total if valid else 0.0,
        "ut_passed": passed,
        "ut_total": total,
        "pytest_exit_code": proc.returncode,
        "selfcheck_exit_code": selfcheck.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-2000:],
        "selfcheck_tail": selfcheck.stdout[-2000:],
    }
    print(json.dumps(result, indent=2))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

