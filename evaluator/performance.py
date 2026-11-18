"""Deterministic correctness-first pressure checks for the tiny evaluator.

The pressure helpers intentionally avoid wall-clock thresholds.  CI hosts vary
widely, while model-forward accounting is stable and makes a useful gate for
batching regressions.  A gate first checks exact oracle parity and only then
checks the amount of model work performed on a seeded, sharded workload.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Iterable, Sequence

try:
    from .oracle import generate as oracle_generate
    from .tiny_evaluator import EvalRequest, TinyBatchRuntime
except ImportError:
    from oracle import generate as oracle_generate
    from tiny_evaluator import EvalRequest, TinyBatchRuntime


@dataclass(frozen=True)
class PressureGateResult:
    """Summary of a correctness and model-work pressure run."""

    ok: bool
    correctness_ok: bool
    performance_ok: bool
    num_requests: int
    seed: int
    shard: int
    shards: int
    model_calls: int
    max_model_calls: int
    prefix_cache_hits: int
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "correctness_ok": self.correctness_ok,
            "performance_ok": self.performance_ok,
            "num_requests": self.num_requests,
            "seed": self.seed,
            "shard": self.shard,
            "shards": self.shards,
            "model_calls": self.model_calls,
            "max_model_calls": self.max_model_calls,
            "prefix_cache_hits": self.prefix_cache_hits,
            "reason": self.reason,
        }


def seeded_pressure_requests(
    *,
    count: int = 128,
    seed: int = 0,
    shard: int = 0,
    shards: int = 1,
) -> list[EvalRequest]:
    """Create a reproducible workload and select one hidden-test shard.

    Rows are generated before sharding, so changing ``shard`` never changes
    the values in another shard.  Duplicate prompts and context variants are
    deliberate and exercise both sequence and shared-prefix caches.
    """

    if count < 0:
        raise ValueError("count must be non-negative")
    if shards <= 0:
        raise ValueError("shards must be positive")
    if shard < 0 or shard >= shards:
        raise ValueError("shard must be in [0, shards)")
    rng = random.Random(int(seed))
    all_rows: list[EvalRequest] = []
    for index in range(count):
        stem = rng.randrange(0, 32)
        # Repeated stems produce shared prefixes while the suffix preserves
        # enough distinct rows to keep the batch representative.
        prompt = (stem, (stem + index % 5) % 32, index % 3)
        all_rows.append(
            EvalRequest(
                request_id=f"pressure-{seed}-{index}",
                prompt=prompt,
                model="tiny" if index % 7 else "tiny-alt",
                adapter="base" if index % 5 else "adapter-alt",
                tenant="tenant-a" if index % 3 else "tenant-b",
                temperature=0.0 if index % 4 else 0.25,
                max_new_tokens=4 + index % 4,
                min_new_tokens=index % 2,
                eos_token_id=None if index % 6 else 0,
            )
        )
    return [row for index, row in enumerate(all_rows) if index % shards == shard]


def run_pressure_gate(
    requests: Iterable[EvalRequest] | None = None,
    *,
    runtime_factory: Callable[[], TinyBatchRuntime] = TinyBatchRuntime,
    count: int = 128,
    seed: int = 0,
    shard: int = 0,
    shards: int = 1,
    max_model_calls: int | None = None,
) -> PressureGateResult:
    """Run oracle correctness first, followed by a deterministic work gate.

    ``max_model_calls`` defaults to the number of unique semantic rows.  This
    catches accidental per-row duplicate work while accepting legitimate
    context and prompt differences.
    """

    rows = list(requests) if requests is not None else seeded_pressure_requests(
        count=count, seed=seed, shard=shard, shards=shards
    )
    runtime = runtime_factory()
    correctness_ok = True
    reason = ""
    try:
        packed = runtime.generate_batch(rows, mode="packed")
        padded = runtime.generate_batch(rows, mode="padded")
        correctness_ok = packed == padded and all(
            output == list(oracle_generate(row.as_oracle_request()))
            for row, output in zip(rows, packed)
        )
        if not correctness_ok:
            reason = "oracle or padded/packed parity failed"
    except Exception as exc:  # pragma: no cover - exercised by candidate failures
        correctness_ok = False
        reason = f"correctness raised {type(exc).__name__}: {exc}"
        packed = []

    unique_semantics = {
        (
            row.model,
            row.adapter,
            row.tenant,
            type(row.temperature).__name__,
            float(row.temperature),
            tuple((type(token).__name__, token) for token in row.prompt),
            row.max_new_tokens,
            row.min_new_tokens,
            row.eos_token_id,
        )
        for row in rows
    }
    limit = len(unique_semantics) if max_model_calls is None else int(max_model_calls)
    calls = int(runtime.stats.model_calls)
    performance_ok = calls <= limit
    if not performance_ok and not reason:
        reason = f"model_calls={calls} exceeds max_model_calls={limit}"
    return PressureGateResult(
        ok=bool(correctness_ok and performance_ok),
        correctness_ok=correctness_ok,
        performance_ok=performance_ok,
        num_requests=len(rows),
        seed=int(seed),
        shard=int(shard),
        shards=int(shards),
        model_calls=calls,
        max_model_calls=limit,
        prefix_cache_hits=int(runtime.stats.prefix_cache_hits),
        reason=reason,
    )


__all__ = ["PressureGateResult", "run_pressure_gate", "seeded_pressure_requests"]
