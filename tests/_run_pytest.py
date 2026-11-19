"""Run pytest in a clean interpreter and identify candidate syntax errors."""

import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

import pytest


class CandidateSyntaxErrors:
    def __init__(self, repo):
        self.repo = repo
        self.modules = {}

    def pytest_exception_interact(self, node, call, report):
        if report.when != "collect":
            return
        # Pytest wraps import-time SyntaxError in Collector.CollectError.
        exc = call.excinfo.value
        while exc is not None:
            if isinstance(exc, SyntaxError) and exc.filename:
                source = Path(exc.filename).resolve()
                if source.is_relative_to(self.repo):
                    self.modules[node.path.name] = (
                        f"{source.relative_to(self.repo)}:{exc.lineno}: {exc.msg}"
                    )
                return
            exc = exc.__cause__ or exc.__context__


def main():
    tests, repo, attribution_file, *args = sys.argv[1:]
    # These dependencies are also used by lazy imports in upstream Collator and
    # CachingLM. A broken worker environment must not count as a model failure.
    for dependency in ("jinja2", "numpy", "requests", "sqlitedict", "tqdm", "typing_extensions", "yaml"):
        importlib.import_module(dependency)
    importlib.metadata.version("lm_eval")
    sys.path[:0] = [tests, repo]
    for module in ("llm_runtime", "lm_eval"):
        spec = importlib.util.find_spec(module)
        if spec is None or not spec.origin or not Path(spec.origin).resolve().is_relative_to(repo):
            sys.exit(f"Candidate workspace must supply {module} inside {repo}")
    errors = CandidateSyntaxErrors(Path(repo))
    code = pytest.main(args, plugins=[errors])
    Path(attribution_file).write_text(json.dumps(errors.modules), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
