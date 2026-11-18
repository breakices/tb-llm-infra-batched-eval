"""Dependency-free tiny batched evaluator.

The production-shaped part of the local task lives here instead of in a test
fixture.  ``TinyBatchRuntime`` exercises padded and packed batch paths, keeps
prefix state scoped by model context, and returns one result per input
position.  ``TinyEvaluator`` then computes weighted metrics against the
independent oracle in :mod:`evaluator.oracle`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from .oracle import (
    DEFAULT_EOS_TOKEN_ID,
    DEFAULT_VOCAB_SIZE,
    OracleRequest,
    generate as oracle_generate,
)


@dataclass(frozen=True)
class EvalRequest:
    """One evaluation item.

    ``request_id`` is metadata only.  It is deliberately excluded from cache
    keys, so duplicate IDs and repeated semantic requests remain independent
    input positions while still allowing work reuse.
    """

    request_id: str
    prompt: tuple[int, ...]
    model: str = "tiny"
    adapter: str = "base"
    tenant: str = "default"
    temperature: float = 0.0
    max_new_tokens: int = 4
    min_new_tokens: int = 0
    eos_token_id: Optional[int] = DEFAULT_EOS_TOKEN_ID
    weight: float = 1.0
    reference: Optional[tuple[int, ...]] = None

    def as_oracle_request(self) -> OracleRequest:
        return OracleRequest(
            request_id=self.request_id,
            prompt=tuple(self.prompt),
            model=self.model,
            adapter=self.adapter,
            tenant=self.tenant,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
            eos_token_id=self.eos_token_id,
            weight=self.weight,
            reference=self.reference,
        )


@dataclass
class RuntimeStats:
    """Counters exposed to private tests without inspecting implementation state."""

    mode: str = "packed"
    batch_calls: int = 0
    model_calls: int = 0
    sequence_cache_hits: int = 0
    prefix_cache_hits: int = 0
    prefix_cache_misses: int = 0
    padded_tokens: int = 0
    packed_tokens: int = 0

    def copy(self) -> "RuntimeStats":
        return RuntimeStats(**self.__dict__)


@dataclass
class EvaluationResult:
    """Outputs, metrics, and runtime counters for one evaluation run."""

    outputs: list[list[int]]
    metrics: dict[str, float]
    stats: RuntimeStats
    records: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "outputs": [list(tokens) for tokens in self.outputs],
            "metrics": dict(self.metrics),
            "stats": dict(self.stats.__dict__),
            "records": [dict(record) for record in self.records],
        }


def _stable_text(value: object) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(str(value))) % 1009


def _validate_request(request: EvalRequest, vocab_size: int) -> None:
    if request.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if request.min_new_tokens < 0:
        raise ValueError("min_new_tokens must be non-negative")
    if request.min_new_tokens > request.max_new_tokens:
        raise ValueError("min_new_tokens cannot exceed max_new_tokens")
    if not math.isfinite(float(request.temperature)):
        raise ValueError("temperature must be finite")
    if request.eos_token_id is not None and not 0 <= int(request.eos_token_id) < vocab_size:
        raise ValueError("eos_token_id is outside the vocabulary")
    if not math.isfinite(float(request.weight)) or float(request.weight) < 0:
        raise ValueError("weight must be a finite non-negative number")


def _context_key(request: EvalRequest) -> tuple[object, ...]:
    temperature = float(request.temperature)
    return (
        request.model,
        request.adapter,
        request.tenant,
        type(request.temperature).__name__,
        temperature,
        repr(temperature),
    )


def _token_key(token: object) -> tuple[str, str, object]:
    """Return a hashable cache component that preserves token identity.

    Python considers ``True == 1`` and gives them the same hash.  Prompt
    arithmetic intentionally treats both values as numeric token ids, while
    cache entries must keep their distinct input identities.
    """

    return (type(token).__module__, type(token).__qualname__, token)


def pad_batch(requests: Sequence[EvalRequest], pad_token_id: int = -1) -> tuple[list[list[int]], list[int]]:
    """Return a left-aligned padded prompt matrix and original lengths."""

    lengths = [len(request.prompt) for request in requests]
    width = max(lengths, default=0)
    rows = [list(request.prompt) + [pad_token_id] * (width - len(request.prompt)) for request in requests]
    return rows, lengths


def pack_batch(requests: Sequence[EvalRequest]) -> tuple[list[int], list[int]]:
    """Return flattened prompts and exclusive-end offsets."""

    tokens: list[int] = []
    offsets = [0]
    for request in requests:
        # Keep the original token object in the packed layout. Arithmetic
        # generation can coerce it later, while cache identity must keep bool
        # and int prompt values distinct.
        tokens.extend(request.prompt)
        offsets.append(len(tokens))
    return tokens, offsets


class TinyBatchRuntime:
    """Small deterministic runtime used as the candidate implementation.

    The state update is intentionally duplicated from the oracle rather than
    importing its private implementation.  This lets the private verifier
    detect accidental oracle coupling while still checking exact parity.
    """

    def __init__(
        self,
        *,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        pad_token_id: int = -1,
        shared_prefix_cache: bool = True,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.vocab_size = int(vocab_size)
        self.pad_token_id = int(pad_token_id)
        self.shared_prefix_cache = bool(shared_prefix_cache)
        self._prefix_cache: dict[tuple[object, ...], int] = {}
        self._sequence_cache: dict[tuple[object, ...], tuple[int, ...]] = {}
        self.stats = RuntimeStats()

    def clear(self) -> None:
        self._prefix_cache.clear()
        self._sequence_cache.clear()

    def reset_stats(self) -> None:
        self.stats = RuntimeStats()

    @property
    def prefix_cache_size(self) -> int:
        return len(self._prefix_cache)

    @property
    def sequence_cache_size(self) -> int:
        return len(self._sequence_cache)

    def _initial_state(self, request: EvalRequest) -> int:
        context = (
            _stable_text(request.model)
            + 3 * _stable_text(request.adapter)
            + 5 * _stable_text(request.tenant)
            + int(round(float(request.temperature) * 1000))
        ) % 1009
        return (context + 97) % 1009

    def _prefix_state(self, request: EvalRequest) -> int:
        context = _context_key(request)
        state = self._initial_state(request)
        prefix: tuple[object, ...] = ()
        for index, token in enumerate(request.prompt):
            prefix = prefix + (token,)
            key = (context, tuple(_token_key(value) for value in prefix))
            if self.shared_prefix_cache and key in self._prefix_cache:
                state = self._prefix_cache[key]
                self.stats.prefix_cache_hits += 1
                continue
            state = (state * 31 + (index + 1) * int(token) + 17) % 1009
            if self.shared_prefix_cache:
                self._prefix_cache[key] = state
            self.stats.prefix_cache_misses += 1
        if not request.prompt and self.shared_prefix_cache:
            key = (context, ())
            if key in self._prefix_cache:
                self.stats.prefix_cache_hits += 1
                state = self._prefix_cache[key]
            else:
                self._prefix_cache[key] = state
                self.stats.prefix_cache_misses += 1
        return state

    def _generate_one(self, request: EvalRequest) -> tuple[int, ...]:
        _validate_request(request, self.vocab_size)
        state = self._prefix_state(request)
        output: list[int] = []
        for step in range(request.max_new_tokens):
            token = (state + 13 * step + 3) % self.vocab_size
            output.append(token)
            state = (state * 37 + token + step + 11) % 1009
            if (
                request.eos_token_id is not None
                and token == int(request.eos_token_id)
                and len(output) >= request.min_new_tokens
            ):
                break
        return tuple(output)

    def _sequence_key(self, request: EvalRequest) -> tuple[object, ...]:
        return (
            _context_key(request),
            tuple(_token_key(token) for token in request.prompt),
            int(request.max_new_tokens),
            int(request.min_new_tokens),
            None if request.eos_token_id is None else _token_key(request.eos_token_id),
        )

    def generate(self, request: EvalRequest, *, mode: str = "packed") -> list[int]:
        return self.generate_batch([request], mode=mode)[0]

    def generate_batch(self, requests: Iterable[EvalRequest], *, mode: str = "packed") -> list[list[int]]:
        requests = list(requests)
        if mode not in {"padded", "packed"}:
            raise ValueError("mode must be 'padded' or 'packed'")
        self.stats.mode = mode
        self.stats.batch_calls += 1
        if mode == "padded":
            rows, lengths = pad_batch(requests, self.pad_token_id)
            self.stats.padded_tokens += sum(len(row) for row in rows)
            # Consume only the original lengths.  This explicit read makes it
            # difficult for a future padded implementation to treat pad IDs as
            # prompt tokens accidentally.
            for request, row, length in zip(requests, rows, lengths):
                if tuple(row[:length]) != tuple(request.prompt):
                    raise AssertionError("padded representation changed a prompt")
        else:
            tokens, offsets = pack_batch(requests)
            self.stats.packed_tokens += len(tokens)
            for index, request in enumerate(requests):
                if tuple(tokens[offsets[index] : offsets[index + 1]]) != tuple(request.prompt):
                    raise AssertionError("packed representation changed a prompt")

        outputs: list[Optional[tuple[int, ...]]] = [None] * len(requests)
        for index, request in enumerate(requests):
            key = self._sequence_key(request)
            cached = self._sequence_cache.get(key)
            if cached is not None:
                self.stats.sequence_cache_hits += 1
                outputs[index] = cached
                continue
            self.stats.model_calls += 1
            value = self._generate_one(request)
            self._sequence_cache[key] = value
            outputs[index] = value
        return [list(value or ()) for value in outputs]


def _token_accuracy(actual: Sequence[int], expected: Sequence[int]) -> float:
    width = max(len(actual), len(expected))
    if width == 0:
        return 1.0
    matches = sum(
        left == right
        for left, right in zip(actual, expected)
    )
    return matches / width


def weighted_metrics(
    requests: Sequence[EvalRequest],
    outputs: Sequence[Sequence[int]],
    *,
    oracle: Callable[[OracleRequest], Sequence[int]] = oracle_generate,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Compute exact-match and token metrics with request weights."""

    if len(requests) != len(outputs):
        raise ValueError("one output is required for every request position")
    if not requests:
        return (
            {
                "weighted_exact_match": 0.0,
                "weighted_token_accuracy": 0.0,
                "exact_match": 1.0,
                "token_accuracy": 1.0,
                "total_weight": 0.0,
                "num_requests": 0.0,
                "num_exact": 0.0,
            },
            [],
        )
    total_weight = sum(float(request.weight) for request in requests)
    if total_weight <= 0:
        raise ValueError("at least one request must have positive weight")
    records: list[dict[str, Any]] = []
    exact_weight = 0.0
    token_weight = 0.0
    exact_count = 0
    token_sum = 0.0
    for request, output in zip(requests, outputs):
        _validate_request(request, DEFAULT_VOCAB_SIZE)
        expected = (
            tuple(request.reference)
            if request.reference is not None
            else tuple(oracle(request.as_oracle_request()))
        )
        actual = tuple(int(token) for token in output)
        exact = float(actual == expected)
        token_score = _token_accuracy(actual, expected)
        weight = float(request.weight)
        exact_weight += weight * exact
        token_weight += weight * token_score
        exact_count += int(exact)
        token_sum += token_score
        records.append(
            {
                "request_id": request.request_id,
                "expected": list(expected),
                "actual": list(actual),
                "exact_match": bool(exact),
                "token_accuracy": token_score,
                "weight": weight,
            }
        )
    count = len(requests)
    metrics = {
        "weighted_exact_match": exact_weight / total_weight,
        "weighted_token_accuracy": token_weight / total_weight,
        "exact_match": exact_count / count if count else 1.0,
        "token_accuracy": token_sum / count if count else 1.0,
        "total_weight": total_weight,
        "num_requests": float(count),
        "num_exact": float(exact_count),
    }
    return metrics, records


class TinyEvaluator:
    """Run a batch and compare it with the independent oracle."""

    def __init__(
        self,
        runtime: Optional[TinyBatchRuntime] = None,
        *,
        oracle: Callable[[OracleRequest], Sequence[int]] = oracle_generate,
    ) -> None:
        self.runtime = runtime or TinyBatchRuntime()
        self.oracle = oracle

    def run(self, requests: Iterable[EvalRequest], *, mode: str = "packed") -> EvaluationResult:
        requests = list(requests)
        outputs = self.runtime.generate_batch(requests, mode=mode)
        metrics, records = weighted_metrics(requests, outputs, oracle=self.oracle)
        return EvaluationResult(
            outputs=outputs,
            metrics=metrics,
            stats=self.runtime.stats.copy(),
            records=records,
        )

    def score(
        self,
        requests: Iterable[EvalRequest],
        *,
        mode: str = "packed",
    ) -> float:
        """Return the weighted exact-match score for a batch.

        This convenience method keeps the local evaluator pleasant to use in a
        small private test.  Callers that need per-row diagnostics should use
        :meth:`run` and inspect ``EvaluationResult.records``.
        """

        return self.run(requests, mode=mode).metrics["weighted_exact_match"]


__all__ = [
    "EvalRequest",
    "EvaluationResult",
    "RuntimeStats",
    "TinyBatchRuntime",
    "TinyEvaluator",
    "pack_batch",
    "pad_batch",
    "weighted_metrics",
]
