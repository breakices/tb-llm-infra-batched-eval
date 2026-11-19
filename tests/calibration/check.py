"""Replay author controls in fresh, offline containers using the shipped verifier."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


PACKAGE = Path(__file__).resolve().parents[2]
PROFILES = {
    "nop": (None, 0),
    "golden": ("solution/golden/golden.patch", 1),
    "full_score": ("tests/calibration/full_score.patch", 1),
    "equivalent": ("tests/calibration/equivalent.patch", 1),
    "variant_a": ("solution/variants/variant_a.patch", 1),
    "variant_b": ("solution/variants/variant_b.patch", 1),
    "guessed": ("tests/calibration/guessed.patch", 0),
    "defective": ("tests/calibration/defective.patch", 0),
}
TARGET_FAILURES = {
    "nop": {"U23", "U42", "U45"},
    "guessed": {"U23", "U45"},
    "defective": {"U5", "U6", "U46"},
}


def bind(source, destination, writable=False):
    mode = "rw" if writable else "ro"
    return ["--volume", f"{source}:{destination}:{mode}"]


def replay(job):
    name, iteration, image, output = job
    patch, expected = PROFILES[name]
    testcase = f"{name}-{iteration}"
    with tempfile.TemporaryDirectory(prefix="codesense-control-") as tmp:
        logs = Path(tmp) / "logs"
        logs.mkdir()
        # Pytest-rewritten bytecode can retain filenames from the author's
        # machine. Mount verifier sources just as a clean checkout would.
        verifier = Path(tmp) / "tests"
        shutil.copytree(PACKAGE / "tests", verifier,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "assets", "calibration"))
        command = ["docker", "run", "--rm", "--platform", "linux/amd64",
                   "--network", "none", "--user", "root"]
        command += bind(verifier, "/tests")
        command += bind(logs, "/logs/verifier", writable=True)
        command += bind(PACKAGE / "solution" / "counterexamples", "/author-probes")
        steps = [
            "set -eu", "cd /workspace/repo", "export PYTHONDONTWRITEBYTECODE=1",
            "python --version", "uname -sm",
            "git config --global --add safe.directory /workspace/repo",
            "git rev-parse HEAD^{tree}", "test -z \"$(git status --porcelain)\"",
        ]
        if name == "golden":
            command += bind(PACKAGE / "solution", "/solution")
            steps.append("sh /solution/solve.sh")
        elif patch:
            command += bind(PACKAGE / patch, "/control.patch")
            steps.append("git apply --check /control.patch")
            steps.append("git apply /control.patch")
        steps.append("git diff --stat")
        steps.append("sh /tests/test.sh")
        behavior = "correct" if expected else "defect"
        steps.append(
            f"python -I -B /author-probes/cache_observations.py . --require {behavior}"
            " > /logs/verifier/observations.json"
        )
        command += [image, "sh", "-c", "\n".join(steps)]
        process = subprocess.run(command, capture_output=True)
        if process.returncode:
            error_log = logs / "score.stderr"
            detail = process.stderr.decode(errors="replace") + (
                error_log.read_text() if error_log.is_file() else ""
            )
            result_log = logs / "result.json"
            if result_log.is_file():
                failure = json.loads(result_log.read_text())
                detail += str(failure.get("problems", [])) + failure.get("diagnostic", "")
            raise RuntimeError(f"{testcase}: verifier exited {process.returncode}: {detail[-1600:]}")
        result = json.loads((logs / "result.json").read_bytes())
        reward = float((logs / "reward.txt").read_text().strip())
        failures = {entry["id"] for entry in result["results"] if entry["status"] == "failed"}
        if (not result["valid"] or reward != expected or result["score"] != expected
                or not TARGET_FAILURES.get(name, set()).issubset(failures)):
            raise RuntimeError(f"{testcase}: wanted valid reward {expected}, got {reward}, "
                               f"failures={sorted(failures)}, problems={result.get('problems')}")
        files = {str(p.relative_to(logs)): p.read_text()
                 for p in sorted(logs.rglob("*")) if p.is_file()}
        files["entry.stdout"] = process.stdout.decode()
        files["entry.stderr"] = process.stderr.decode()
        evidence = json.dumps(files, ensure_ascii=False, sort_keys=True).encode()
        (output / f"{testcase}.json.gz").write_bytes(gzip.compress(evidence, mtime=0))
        methods = [(entry["id"], entry["status"]) for entry in result["results"]]
        cases = [(row["nodeid"], row["status"]) for entry in result["results"] for row in entry["cases"]]
        return name, methods, cases, json.loads(files["observations.json"]), reward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="task image built from this checkout")
    parser.add_argument("--out", type=Path, required=True, help="directory for the per-run records")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.repetitions < 1 or args.jobs < 1:
        parser.error("repetitions and jobs must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", args.image], text=True
    ).strip()
    jobs = [(name, n, image_id, args.out) for name in PROFILES
            for n in range(1, args.repetitions + 1)]
    profiles = {}
    signatures = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for name, outcomes, cases, observations, reward in pool.map(replay, jobs):
            patch = PROFILES[name][0]
            summary = {
                "reward": reward,
                "ut_total": len(outcomes),
                "ut_passed": sum(status == "passed" for _, status in outcomes),
                "case_total": len(cases),
                "case_passed": sum(status == "passed" for _, status in cases),
                "failures": [test for test, status in outcomes if status != "passed"],
                "patch_sha256": hashlib.sha256((PACKAGE / patch).read_bytes()).hexdigest()
                if patch else None,
            }
            if profiles.setdefault(name, summary) != summary:
                raise RuntimeError(f"{name}: inconsistent results across fresh containers")
            if signatures.setdefault(name, (cases, observations)) != (cases, observations):
                raise RuntimeError(f"{name}: case outcomes or observations differ across runs")
    report = {"image": image_id, "repetitions": args.repetitions, "profiles": profiles}
    (args.out / "matrix.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, row in profiles.items():
        print(f"{name}: {row['ut_passed']}/{row['ut_total']}, reward {row['reward']:.0f}")


if __name__ == "__main__":
    main()
