"""Deterministic private-gate helpers for the local tiny evaluator.

The gate deliberately keeps expected values in :mod:`evaluator.oracle`, which
never imports a candidate runtime.  It provides a small, dependency-free
harness for generating a hidden-style corpus, selecting reproducible shards,
checking padded/packed correctness, and running a performance check only after
correctness has passed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import random
import time
from typing import Any, Iterable, Mapping, Sequence

from .oracle import OracleRequest, generate as oracle_generate
from .tiny_evaluator import EvalRequest, TinyBatchRuntime, TinyEvaluator


DEFAULT_CASE_COUNT = 192
DEFAULT_SEED = 0x5EED03


@dataclass(frozen=True)
class GateConfig:
    """Configuration shared by local and private gate invocations."""

    seed: int = DEFAULT_SEED
    case_count: int = DEFAULT_CASE_COUNT
    shard_index: int = 0
    shard_count: int = 1
    min_correctness: float = 1.0
    max_model_calls_per_case: float = 1.0
    performance_repetitions: int = 3

    def __post_init__(self) -> None:
        if int(self.case_count) < 0:
            raise ValueError("case_count must be non-negative")
        if int(self.shard_count) <= 0:
            raise ValueError("shard_count must be positive")
        if not 0 <= int(self.shard_index) < int(self.shard_count):
            raise ValueError("shard_index must be within shard_count")
        if not 0.0 <= float(self.min_correctness) <= 1.0:
            raise ValueError("min_correctness must be between zero and one")
        if float(self.max_model_calls_per_case) < 0 or not math.isfinite(float(self.max_model_calls_per_case)):
            raise ValueError("max_model_calls_per_case must be finite and non-negative")
        if int(self.performance_repetitions) <= 0:
            raise ValueError("performance_repetitions must be positive")


@dataclass
class GateResult:
    """Machine-readable result with an explicit correctness/performance split."""

    ok: bool
    correctness_ok: bool
    performance_ok: bool | None
    score: float
    num_cases: int
    failures: list[dict[str, Any]] = field(default_factory=list)
    performance: dict[str, float] = field(default_factory=dict)
    skipped_performance: bool = False
    seed: int = DEFAULT_SEED
    shard_index: int = 0
    shard_count: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "correctness_ok": bool(self.correctness_ok),
            "performance_ok": self.performance_ok,
            "score": float(self.score),
            "num_cases": int(self.num_cases),
            "failures": [dict(item) for item in self.failures],
            "performance": dict(self.performance),
            "skipped_performance": bool(self.skipped_performance),
            "seed": int(self.seed),
            "shard_index": int(self.shard_index),
            "shard_count": int(self.shard_count),
        }


def _request_from_rng(index: int, rng: random.Random) -> EvalRequest:
    """Construct one request with intentionally awkward edge values."""

    prompts: tuple[tuple[Any, ...], ...] = (
        (),
        (0,),
        (1, True),
        (-1, 2, 3),
        (31, 30, 29),
        (7, 7, 7),
        (ord("雪"), ord("é")),
        (2, 4, 6, 8, 10, 12),
    )
    prompt = prompts[index % len(prompts)]
    # Add a deterministic Unicode token context without allowing unbounded
    # values into the tiny vocabulary arithmetic.
    if index % 11 == 0:
        prompt = prompt + (rng.choice((3, 5, 9)),)
    max_new = (index * 7 + rng.randrange(0, 5)) % 9
    min_new = 0 if max_new == 0 else (index * 3) % (max_new + 1)
    eos: int | None = None if index % 4 == 0 else ((index * 5) % 31)
    temperature = (rng.randrange(-4, 9) / 10.0)
    weight = (0.25 if index % 13 == 0 else float((index % 5) + 1))
    # Context labels include non-ASCII text and collision-prone values to make
    # accidental string/identity-only cache keys visible.
    model = "tiny-雪" if index % 9 == 0 else ("tiny" if index % 3 else "tiny-alt")
    adapter = "base" if index % 5 else "适配器"
    tenant = "租户-β" if index % 7 == 0 else ("default" if index % 2 else "tenant-a")
    return EvalRequest(
        request_id="dup" if index % 17 in (0, 1) else f"case-{index}",
        prompt=prompt,
        model=model,
        adapter=adapter,
        tenant=tenant,
        temperature=temperature,
        max_new_tokens=max_new,
        min_new_tokens=min_new,
        eos_token_id=eos,
        weight=weight,
    )


def make_cases(seed: int = DEFAULT_SEED, count: int = DEFAULT_CASE_COUNT) -> list[EvalRequest]:
    """Return a reproducible corpus of at least 150 edge-rich requests."""

    if int(count) < 0:
        raise ValueError("count must be non-negative")
    rng = random.Random(int(seed))
    cases = [_request_from_rng(i, rng) for i in range(int(count))]
    # Repeat full semantic rows at a regular cadence.  The row ids and
    # weights remain metadata, while generation should be served by one cache
    # entry.  This also gives each hidden shard duplicate-sample coverage.
    for index in range(23, len(cases), 29):
        source = cases[index - 1]
        cases[index] = replace(source, request_id=f"dup-row-{index}", weight=0.5)
    return cases


def select_shard(cases: Sequence[EvalRequest], shard_index: int = 0, shard_count: int = 1) -> list[EvalRequest]:
    """Select a deterministic, disjoint shard while preserving corpus order."""

    if int(shard_count) <= 0:
        raise ValueError("shard_count must be positive")
    if not 0 <= int(shard_index) < int(shard_count):
        raise ValueError("shard_index must be within shard_count")
    return [case for index, case in enumerate(cases) if index % int(shard_count) == int(shard_index)]


def _semantic_key(request: EvalRequest) -> tuple[Any, ...]:
    # request_id and weight do not affect generation work.
    return (
        request.model,
        request.adapter,
        request.tenant,
        type(request.temperature).__name__,
        float(request.temperature),
        tuple((type(t).__name__, t) for t in request.prompt),
        int(request.max_new_tokens),
        int(request.min_new_tokens),
        request.eos_token_id,
        request.context_length,
        request.seed,
        request.shard,
        tuple(request.stop_sequences),
        tuple((hint.name, tuple((type(token).__name__, token) for token in hint.tokens)) for hint in request.prefix_hints),
    )


def correctness_gate(
    requests: Sequence[EvalRequest],
    *,
    runtime: TinyBatchRuntime | None = None,
) -> tuple[bool, float, list[dict[str, Any]], TinyBatchRuntime]:
    """Compare both layouts with independent oracle outputs, row by row."""

    runtime = runtime or TinyBatchRuntime()
    failures: list[dict[str, Any]] = []
    packed = runtime.generate_batch(requests, mode="packed")
    padded = runtime.generate_batch(requests, mode="padded")
    for index, request in enumerate(requests):
        expected = list(oracle_generate(request.as_oracle_request()))
        for mode, outputs in (("packed", packed), ("padded", padded)):
            actual = list(outputs[index])
            if actual != expected:
                failures.append({"index": index, "mode": mode, "request_id": request.request_id, "expected": expected, "actual": actual})
        if packed[index] != padded[index]:
            failures.append({"index": index, "mode": "parity", "request_id": request.request_id, "expected": packed[index], "actual": padded[index]})
    score = 1.0 - (len(failures) / max(1, 2 * len(requests)))
    return not failures, score, failures, runtime


def performance_gate(
    requests: Sequence[EvalRequest],
    runtime: TinyBatchRuntime,
    *,
    repetitions: int = 3,
    max_model_calls_per_case: float = 1.0,
) -> tuple[bool, dict[str, float]]:
    """Run pressure only after correctness and report stable cache counters."""

    if int(repetitions) <= 0:
        raise ValueError("repetitions must be positive")
    runtime.clear()
    runtime.reset_stats()
    started = time.perf_counter()
    for _ in range(int(repetitions)):
        runtime.generate_batch(requests, mode="packed")
    elapsed = time.perf_counter() - started
    unique = len({_semantic_key(request) for request in requests if request.max_new_tokens > 0})
    calls = float(runtime.stats.model_calls)
    # First execution may need at most one model call per semantic row. Every
    # subsequent repetition should be served by the sequence cache.
    limit = float(max_model_calls_per_case) * max(1, unique)
    pressure = {
        "elapsed_seconds": float(elapsed),
        "model_calls": calls,
        "unique_cases": float(unique),
        "model_calls_per_unique_case": calls / max(1, unique),
        "repetitions": float(repetitions),
    }
    return calls <= limit, pressure


def run_gate(config: GateConfig = GateConfig(), *, runtime: TinyBatchRuntime | None = None) -> GateResult:
    """Execute correctness first, then conditionally run the performance gate."""

    all_cases = make_cases(config.seed, config.case_count)
    requests = select_shard(all_cases, config.shard_index, config.shard_count)
    try:
        correctness_ok, score, failures, runtime = correctness_gate(requests, runtime=runtime)
    except Exception as exc:
        correctness_ok, score, failures = False, 0.0, [{"phase": "correctness", "error": repr(exc)}]
        runtime = runtime or TinyBatchRuntime()
    meets_score = correctness_ok and score >= float(config.min_correctness)
    if meets_score:
        performance_ok, performance = performance_gate(
            requests,
            runtime,
            repetitions=config.performance_repetitions,
            max_model_calls_per_case=config.max_model_calls_per_case,
        )
        skipped = False
    else:
        performance_ok, performance, skipped = None, {}, True
    return GateResult(
        ok=bool(meets_score and performance_ok is True),
        correctness_ok=bool(meets_score),
        performance_ok=performance_ok,
        score=float(score),
        num_cases=len(requests),
        failures=failures,
        performance=performance,
        skipped_performance=skipped,
        seed=int(config.seed),
        shard_index=int(config.shard_index),
        shard_count=int(config.shard_count),
    )


def selfcheck() -> dict[str, Any]:
    """Validate corpus determinism, shard coverage, oracle independence and gate order."""

    cases_a = make_cases(DEFAULT_SEED, DEFAULT_CASE_COUNT)
    cases_b = make_cases(DEFAULT_SEED, DEFAULT_CASE_COUNT)
    if cases_a != cases_b:
        raise AssertionError("case generation is not deterministic")
    shards = [select_shard(cases_a, i, 4) for i in range(4)]
    if sum((len(shard) for shard in shards), 0) != len(cases_a):
        raise AssertionError("shards do not cover the corpus")
    golden = run_gate(GateConfig(case_count=DEFAULT_CASE_COUNT, performance_repetitions=1))
    if not golden.ok:
        raise AssertionError(f"default gate failed: {golden.as_dict()}")
    # A deliberately wrong oracle/runtime must fail correctness and skip
    # pressure. This guards against accidentally treating performance as the
    # primary score.
    class WrongRuntime(TinyBatchRuntime):
        def generate_batch(self, requests: Iterable[EvalRequest], *, mode: str = "packed") -> list[list[int]]:
            values = super().generate_batch(requests, mode=mode)
            return [[999] if values else value for value in values]

    wrong = run_gate(GateConfig(case_count=8), runtime=WrongRuntime())
    if wrong.ok or not wrong.skipped_performance or wrong.performance_ok is not None:
        raise AssertionError("correctness must gate performance")
    return {"ok": True, "cases": len(cases_a), "golden": golden.as_dict(), "wrong": wrong.as_dict()}


__all__ = [
    "DEFAULT_CASE_COUNT",
    "DEFAULT_SEED",
    "GateConfig",
    "GateResult",
    "correctness_gate",
    "make_cases",
    "performance_gate",
    "run_gate",
    "select_shard",
    "selfcheck",
]
