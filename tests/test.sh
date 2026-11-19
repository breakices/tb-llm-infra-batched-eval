#!/bin/sh
# Harbor invokes this entry after the solver exits and mounts the private tests.
set -eu
export PYTHONDONTWRITEBYTECODE=1
tests=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
logs=/logs/verifier
mkdir -p "$logs"
rm -f "$logs/reward.txt" "$logs/result.json"
python "$tests/score.py" --repo /workspace/repo --out "$logs/result.json" \
    --artifacts-dir "$logs/pytest" >"$logs/score.stdout" 2>"$logs/score.stderr"
python -I - "$logs" <<'PY'
import json
from pathlib import Path
import sys

logs = Path(sys.argv[1])
report = json.loads((logs / "result.json").read_text())
if report["valid"] is not True or report["score"] not in (0, 1):
    sys.exit("Verifier could not produce a valid binary reward; see result.json")
(logs / "reward.txt").write_text(f"{int(report['score'])}\n")
PY
