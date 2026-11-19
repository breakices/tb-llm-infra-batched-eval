"""Compatibility entry point for the complete candidate verifier.

    python -m evaluator.score --repo /workspace/repo --out result.json

Diagnostic pressure runs of the evaluator fixture live in evaluator.selfcheck.
The candidate verifier always includes its own measured 192-row checks.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "tests" / "score.py"),
                   run_name="__main__")
