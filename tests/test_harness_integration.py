"""Exercise the native lm-eval registry, request objects and reopened SQLite caches."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from lm_eval.api.instance import Instance
from lm_eval.api.model import CachingLM
from lm_eval.api.registry import get_model
from sqlitedict import SqliteDict

import llm_runtime
from llm_runtime import BatchEngine
from tiny_oracle import RecordingModel, oracle_tokens


ALPHABET = "abcdefghijklmnopqrstuvwxyz01234"


def row(context, length, **options):
    # Several distinct documents may legitimately use the same idx and metadata.
    return Instance(
        request_type="generate_until",
        doc={"text": context},
        arguments=(context, {"max_gen_toks": length, **options}),
        idx=0,
        metadata=("generation", 0, 1),
    )


def rendered(context, length, model="m1", adapter="", temperature=0.0):
    tokens = oracle_tokens(model, adapter, context.encode("utf-8"), temperature, length)
    return "".join(ALPHABET[token % len(ALPHABET)] for token in tokens)


@contextmanager
def reopened(path, *, model="m1", tenant="local", adapter="", revision="1", mode="packed"):
    backend = RecordingModel(model)
    engine = BatchEngine(models={model: backend})
    lm = get_model("codesense-batch")(
        model=model, tenant=tenant, adapter=adapter, revision=revision, mode=mode, engine=engine
    )
    cached = CachingLM(lm, str(path))
    try:
        yield cached, backend
    finally:
        cached.dbdict.close()


def test_harness_registry_renders_instances_in_order():
    """A registered backend accepts native Instance rows and returns UTF-8-based text in row order."""
    backend_class = get_model("codesense-batch")
    rows = [row("猫", 4), row("mixed α", 8, temperature=0.25), row("", 0), row("猫", 4)]
    for mode in ("padded", "packed"):
        lm = backend_class.create_from_arg_string(f"model=m1,adapter=base,mode={mode}")
        assert lm.generate_until(rows) == [
            rendered("猫", 4, adapter="base"),
            rendered("mixed α", 8, adapter="base", temperature=0.25),
            "",
            rendered("猫", 4, adapter="base"),
        ]


def test_harness_stops_at_first_text_match():
    """Stop at the first textual occurrence, excluding the marker for scalar and list forms."""
    lm = get_model("codesense-batch")(model="m1", mode="packed")
    text = rendered("offset café", 12)
    later = text[7:10]
    earlier = text[2:5]
    assert text.index(later) == 7 and text.index(earlier) == 2  # fixture has distinct markers
    rows = [row("offset café", 12, until=[later, earlier]),
            row("offset café", 12, until=later), row("offset café", 12)]
    assert lm.generate_until(rows) == [text[:2], text[:7], text]


def _run_cache_worker(database, mode, requests):
    with reopened(database, mode=mode) as (cached, backend):
        outputs = cached.generate_until([row(text, length) for text, length in requests])
        prompts = [list(prompt) for _, batch in backend.calls for prompt in batch]
    return {"outputs": outputs, "model_inputs": prompts}


def test_harness_response_cache_reopens_and_scatters_hits(tmp_path):
    """New Python processes with different hash seeds share only persisted hits in input order."""
    # Start clean Python rather than fork: a module-global cache would otherwise
    # masquerade as durability. -P and -s remove cwd and user packages; -I would
    # also disable the deliberately distinct PYTHONHASHSEEDs.
    bootstrap = """
import json, sys
from pathlib import Path
tests, repo, database, mode, requests, report = sys.argv[1:]
sys.path[:0] = [tests, repo]
from test_harness_integration import _run_cache_worker
Path(report).write_text(json.dumps(_run_cache_worker(database, mode, json.loads(requests))))
"""
    warm, cold = ("first", 4), ("other", 7)
    for seed, mode, requests, expected_inputs in [
        (385, "packed", [warm], [list(b"first")]),
        (729, "padded", [cold, warm, warm], [list(b"other")]),
    ]:
        report = tmp_path / f"worker-{seed}.json"
        env = dict(os.environ, PYTHONHASHSEED=str(seed))
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        proc = subprocess.run(
            [sys.executable, "-P", "-s", "-B", "-c", bootstrap, str(Path(__file__).parent),
             str(Path(llm_runtime.__file__).resolve().parent), str(tmp_path / "saved.sqlite"),
             mode, json.dumps(requests), str(report)],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        observed = json.loads(report.read_text())
        assert observed["outputs"] == [rendered(text, length) for text, length in requests]
        assert observed["model_inputs"] == expected_inputs


def test_harness_response_cache_isolates_serving_scopes(tmp_path):
    """Tenant, model, adapter, revision and independently replaced model lifetimes isolate saved text."""
    database = tmp_path / "shared.sqlite"
    request = row("same input", 4)
    scopes = [{}, {"tenant": "another"}, {"model": "m2"},
              {"adapter": "tuned"}, {"revision": "2"}]
    for scope in scopes:
        with reopened(database, **scope) as (cached, backend):
            assert cached.generate_until([request]) == [
                rendered("same input", 4, model=scope.get("model", "m1"),
                         adapter=scope.get("adapter", ""))
            ]
            assert backend.calls, f"fresh scope {scope!r} reused a different namespace"
    for scope in scopes:
        with reopened(database, **scope) as (cached, backend):
            assert cached.generate_until([request]) == [
                rendered("same input", 4, model=scope.get("model", "m1"),
                         adapter=scope.get("adapter", ""))
            ]
            assert not backend.calls, f"stored response missing from scope {scope!r}"

    # Both workers start on the same deployment, then hot-replace that model
    # once. Their local replacement counters now agree despite serving two
    # different implementations. A restart cannot use that counter as a
    # persistent identity for either replacement.
    raw = oracle_tokens("m1", "", b"same input", 0.0, 4)
    for offset in (1, 2):
        replacement = RecordingModel(model_id="m1", revision=offset)
        expected = "".join(ALPHABET[((token + offset) % 101) % len(ALPHABET)] for token in raw)
        with reopened(database) as (cached, _):
            cached.lm.engine.register(replacement)
            result = cached.generate_until([request])
            assert result == [expected], f"replacement offset {offset} got stale text {result}"
            assert replacement.calls, f"replacement offset {offset} reused another worker's response"
            replacement.calls.clear()
            assert cached.generate_until([request]) == [expected]
            assert not replacement.calls, "replay recomputed within the same replacement lifetime"


def test_harness_partial_checkpoint_uses_response_cache_scope(tmp_path):
    """An interrupted batch checkpoint survives reopening and is visible only to its writer's scope."""
    database = tmp_path / "partial.sqlite"
    request = row("batch", 8)
    response = rendered("batch", 8)
    with reopened(database, tenant="A") as (cached, _):
        cached.lm.cache_hook.add_partial("generate_until", request.args, response)
    with reopened(database, tenant="B") as (cached, backend):
        assert cached.generate_until([request]) == [response]
        assert backend.calls, "a partial result leaked across tenants"
    with reopened(database, tenant="A") as (cached, backend):
        assert cached.generate_until([request]) == [response]
        assert not backend.calls, "a partial result was not visible to the originating tenant"


def test_harness_legacy_response_keys_remain_readable(tmp_path):
    """An unscoped upstream provider can reuse a SQLite response written by the previous release."""
    database = tmp_path / "legacy.sqlite"
    request = row("unchanged", 3)
    # This is the on-disk v0.4.13 format, independent of the candidate's hash function.
    key = hashlib.sha256(json.dumps(["generate_until", *request.args]).encode("utf-8")).hexdigest()
    with SqliteDict(str(database), autocommit=True) as records:
        records[key] = "saved upstream response"
    with reopened(database) as (scoped, backend):
        assert scoped.generate_until([request]) == [rendered("unchanged", 3)]
        assert backend.calls, "a scoped backend reused an unscoped legacy response"
    cache = CachingLM(get_model("dummy")(), str(database))
    try:
        assert cache.generate_until([request]) == ["saved upstream response"]
    finally:
        cache.dbdict.close()
