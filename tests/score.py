"""Score a candidate repository with the frozen verifier methods in ut_index.json.

    python tests/score.py --repo /workspace/repo --out /logs/verifier/result.json

UT counts refer to independent methods. Parameter cases are recorded separately.
An incomplete, skipped or broken verifier run is invalid, with no model reward.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

TESTS = Path(__file__).resolve().parent

def read_index():
    entries = json.loads((TESTS / "ut_index.json").read_text(encoding="utf-8"))
    for key in ("id", "ut_name"):
        values = [entry[key] for entry in entries]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"empty or duplicate {key} in ut_index.json")
    for entry in entries:
        cases = entry["cases"]
        if not cases or len(set(cases)) != len(cases):
            raise ValueError(f"empty or duplicate parameter case for {entry['ut_name']}")
    declared = {entry["file"] for entry in entries}
    actual = {path.name for path in TESTS.glob("test_*.py")}
    if declared != actual:
        raise ValueError(f"verifier files differ from ut_index.json: {sorted(declared ^ actual)}")
    return entries


def parse_report(report, attributed):
    """Retain each pytest outcome, including parameters and assertion evidence."""
    results = {}
    problems = []
    for testcase in ET.parse(report).getroot().iter("testcase"):
        # The sidecar names the verifier module blocked by an import-time
        # SyntaxError in submitted code. It is not itself a collected UT.
        if not testcase.get("classname") and testcase.get("file") in attributed:
            continue
        name = testcase.attrib["name"]
        nodeid = f"{testcase.attrib['classname']}.py::{name}"
        if nodeid in results:
            problems.append(f"duplicate test report: {nodeid}")
        status, evidence = "passed", "all assertions held"
        for failure in testcase:
            if failure.tag not in {"failure", "skipped", "error"}:
                continue
            status = {"failure": "failed", "skipped": "skipped", "error": "error"}[failure.tag]
            evidence = re.sub(r" at 0x[0-9a-fA-F]+", "", failure.get("message", "")).strip()
            if not evidence:
                evidence = (failure.text or "").strip()
            break
        results[nodeid] = {"nodeid": nodeid, "status": status, "evidence": evidence}
    return results, problems


def score(repo, python=sys.executable, timeout=300, artifacts_dir=None):
    entries = read_index()
    selected = sorted({entry["file"] for entry in entries})
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_CURRENT_TEST"):
        env.pop(name, None)
    cases, problems, attributed = {}, [], {}
    returncode = None
    stdout, stderr = "", ""
    with tempfile.TemporaryDirectory(prefix="batch-verifier-") as tmp:
        report = Path(tmp) / "junit.xml"
        attribution_file = Path(tmp) / "candidate-syntax.json"
        cmd = [
            python, "-I", "-B", str(TESTS / "_run_pytest.py"), str(TESTS),
            str(repo.resolve()), str(attribution_file), "--continue-on-collection-errors",
            "-c", os.devnull, "--rootdir", str(TESTS), "--confcutdir", str(TESTS),
            "-p", "no:cacheprovider", "--color=no", "--tb=short", "-q",
            "-o", "junit_family=xunit1", "--junitxml", str(report),
            *(str(TESTS / name) for name in selected),
        ]
        if not repo.is_dir():
            problems.append(f"candidate directory does not exist: {repo}")
        else:
            try:
                proc = subprocess.run(
                    cmd, env=env, cwd=tmp, capture_output=True, text=True,
                    errors="replace", timeout=timeout,
                )
                returncode = proc.returncode
                stdout, stderr = proc.stdout, proc.stderr
                if attribution_file.is_file():
                    attributed = json.loads(attribution_file.read_text(encoding="utf-8"))
                cases, problems = parse_report(report, attributed)
            except subprocess.TimeoutExpired as error:
                stdout = (error.stdout or b"").decode("utf-8", errors="replace")
                stderr = (error.stderr or b"").decode("utf-8", errors="replace")
                problems.append(f"pytest exceeded the verifier time limit of {timeout}s")
            except (OSError, ET.ParseError, KeyError) as error:
                problems.append(f"could not complete verifier or read pytest report: {error}")
        if artifacts_dir:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "stdout.log").write_text(stdout, encoding="utf-8")
            (artifacts_dir / "stderr.log").write_text(stderr, encoding="utf-8")
            saved = artifacts_dir / "junit.xml"
            if report.is_file():
                shutil.copyfile(report, saved)
            else:
                saved.unlink(missing_ok=True)
    if returncode not in (0, 1):
        problems.append(f"pytest did not complete normally (exit {returncode})")

    expected = set()
    methods = []
    for entry in entries:
        rows = []
        attributed_error = attributed.get(entry["file"])
        for name in entry["cases"]:
            nodeid = f"{Path(entry['file']).stem}.py::{name}"
            expected.add(nodeid)
            row = cases.get(nodeid, {
                "nodeid": nodeid, "status": "not_run",
                "evidence": attributed_error or "pytest produced no outcome for this case",
            })
            rows.append(row)
        issues = [row for row in rows if row["status"] not in ("passed", "failed")]
        failures = [row for row in rows if row["status"] == "failed"]
        problems.extend(f"{row['nodeid']}: {row['status']}" for row in issues
                        if row["status"] != "not_run" or not attributed_error)
        first = (issues or failures or [None])[0]
        methods.append({
            "id": entry["id"], "ut_name": entry["ut_name"], "dimension": entry["dimension"],
            "targets_badpattern": entry["targets_badpattern"],
            "status": "passed" if first is None else first["status"],
            "evidence": "all assertions held" if first is None else f"{first['nodeid']}: {first['evidence']}",
            "cases": rows,
        })
    for nodeid in sorted(set(cases) - expected):
        problems.append(f"undeclared pytest outcome: {nodeid}")
    passed = sum(row["status"] == "passed" for row in methods)
    failed = sum(row["status"] == "failed" for row in methods)
    if returncode == 1 and passed == len(methods):
        problems.append("pytest reported failure without a corresponding failed method")
    valid = not problems
    doc = {
        "score": int(valid and passed == len(methods)), "valid": valid, "ut_total": len(methods),
        "ut_passed": passed, "ut_failed": failed,
        "ut_not_run": sum(row["status"] == "not_run" for row in methods),
        "ut_pass_rate": passed / len(methods),
        "case_total": len(expected), "case_passed": sum(row["status"] == "passed" for row in cases.values()),
        "case_failed": sum(row["status"] == "failed" for row in cases.values()),
        "case_not_run": len(expected - set(cases)),
        "candidate_syntax_errors": attributed,
        "pytest_exit_code": returncode, "problems": problems, "results": methods,
    }
    if not valid:
        doc["diagnostic"] = (stdout + stderr)[-6000:]
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", nargs="?", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--artifacts-dir", type=Path, help="save complete pytest stdout, stderr and JUnit XML")
    parser.add_argument("--timeout", type=int, default=300, help="total pytest time limit, in seconds")
    args = parser.parse_args(argv)
    repo, out = args.repo or args.candidate, args.out or args.output
    if repo is None or (args.repo and args.candidate) or (args.out and args.output):
        parser.error("specify one candidate directory and at most one output file")
    document = score(repo, args.python, args.timeout, args.artifacts_dir)
    data = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if out:
        out.write_text(data, encoding="utf-8")
    sys.stdout.write(data)
    return 0 if document["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
