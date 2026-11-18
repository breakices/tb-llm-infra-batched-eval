"""Dependency-free tiny batched evaluator.

The production-shaped part of the local task lives here instead of in a test
fixture.  ``TinyBatchRuntime`` exercises padded and packed batch paths, keeps
prefix state scoped by model context, and returns one result per input
position.  ``TinyEvaluator`` then computes weighted metrics against the
independent oracle in :mod:`evaluator.oracle`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import random
from numbers import Integral, Real
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from .oracle import (
    DEFAULT_EOS_TOKEN_ID,
    DEFAULT_VOCAB_SIZE,
    OracleRequest,
    generate as oracle_generate,
    prompt_state as oracle_prompt_state,
)


@dataclass(frozen=True)
class SupportRow:
    """A support/example row used by contrastive scoring.

    Support rows are deliberately data-only.  They can be supplied as this
    class, a mapping, or a ``(prompt, continuation)`` pair and are normalized
    by :func:`normalize_support_rows`.  Keeping them separate from the main
    request makes empty support and duplicate support rows unambiguous.
    """

    prompt: tuple[Any, ...] = ()
    continuation: tuple[Any, ...] = ()
    weight: float = 1.0
    request_id: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", tuple(self.prompt))
        object.__setattr__(self, "continuation", tuple(self.continuation))
        weight = float(self.weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("support weight must be finite and non-negative")
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True)
class PrefixHint:
    """An optional, cache-visible prefix hint.

    ``tokens`` must be a prefix of the effective prompt.  ``name`` allows
    callers to provide two hints for the same token prefix without creating a
    cache collision.  Contradictory hints are rejected before generation.
    """

    tokens: tuple[Any, ...] = ()
    name: str = "default"

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", tuple(self.tokens))


def _normalize_support_row(row: Any) -> SupportRow:
    if isinstance(row, SupportRow):
        return row
    if isinstance(row, Mapping):
        prompt = row.get("prompt", row.get("input", row.get("tokens", ())))
        continuation = row.get("continuation", row.get("target", row.get("completion", row.get("output", ()))))
        return SupportRow(prompt, continuation, row.get("weight", 1.0), row.get("request_id", row.get("id")))
    if isinstance(row, (tuple, list)):
        if len(row) == 2 and isinstance(row[0], (tuple, list)):
            return SupportRow(row[0], row[1])
        return SupportRow(tuple(row), ())
    raise TypeError("support rows must be SupportRow, mapping, or token sequence")


def _normalize_prefix_hint(hint: Any) -> PrefixHint:
    if isinstance(hint, PrefixHint):
        return hint
    if isinstance(hint, Mapping):
        return PrefixHint(tuple(hint.get("tokens", hint.get("prompt", ()))), str(hint.get("name", "default")))
    return PrefixHint(tuple(hint))


def normalize_support_rows(rows: Iterable[Any] | None) -> tuple[SupportRow, ...]:
    """Normalize support rows without deduplicating their input positions."""

    if rows is None:
        return ()
    if isinstance(rows, (SupportRow, Mapping)):
        return (_normalize_support_row(rows),)
    return tuple(_normalize_support_row(row) for row in rows)


def _normalize_stop_sources(*sources: Any) -> tuple[Any, ...]:
    """Treat scalar text/byte markers as one stop sequence.

    ``bytes`` and ``str`` are iterable, but callers commonly provide one
    marker directly.  Splitting such a marker into one-byte/one-character
    stops changes byte-level matching semantics, so scalar markers are wrapped
    before combining aliases.
    """

    values: list[Any] = []
    for source in sources:
        if source is None:
            continue
        if isinstance(source, (str, bytes, bytearray, memoryview, Integral)):
            values.append(source)
        else:
            values.extend(source)
    return tuple(values)


@dataclass(frozen=True)
class EvalRequest:
    """One evaluation item.

    ``request_id`` is metadata only.  It is deliberately excluded from cache
    keys, so duplicate IDs and repeated semantic requests remain independent
    input positions while still allowing work reuse.
    """

    request_id: str
    prompt: tuple[Any, ...]
    model: str = "tiny"
    adapter: str = "base"
    tenant: str = "default"
    temperature: float = 0.0
    max_new_tokens: int = 4
    min_new_tokens: int = 0
    eos_token_id: Optional[int] = DEFAULT_EOS_TOKEN_ID
    weight: float = 1.0
    reference: Optional[tuple[Any, ...]] = None
    # v3 fields are appended to preserve the v1/v2 positional API.
    support_rows: tuple[SupportRow, ...] = ()
    prefix_hint: Optional[tuple[Any, ...] | PrefixHint] = None
    prefix_hints: tuple[tuple[Any, ...] | PrefixHint, ...] = ()
    context_length: Optional[int] = None
    stop_sequences: tuple[bytes | str, ...] = ()
    # ``stop`` is a short compatibility alias used by some harnesses.
    stop: tuple[bytes | str, ...] = ()
    seed: Optional[int] = None
    shard: Optional[int] = None
    # Common harness aliases retained as keyword-only-compatible appended
    # fields.  They normalize into the canonical fields above.
    support: tuple[Any, ...] = ()
    context_window: Optional[int] = None
    max_context_tokens: Optional[int] = None
    continuation: Optional[tuple[Any, ...]] = None
    stop_strings: tuple[bytes | str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", tuple(self.prompt))
        if self.reference is not None:
            object.__setattr__(self, "reference", tuple(self.reference))
        def _items(value: Any) -> tuple[Any, ...]:
            if value is None:
                return ()
            if isinstance(value, (SupportRow, Mapping, str, bytes, bytearray, memoryview)):
                return (value,)
            return tuple(value)

        rows = _items(self.support_rows) + _items(self.support)
        object.__setattr__(self, "support_rows", normalize_support_rows(rows))
        contexts = [value for value in (self.context_length, self.context_window, self.max_context_tokens) if value is not None]
        if contexts and any(value != contexts[0] for value in contexts):
            raise ValueError("context length aliases disagree")
        if contexts:
            object.__setattr__(self, "context_length", contexts[0])
        object.__setattr__(self, "support", tuple(self.support_rows))
        if self.continuation is not None:
            object.__setattr__(self, "continuation", tuple(self.continuation))
        hints = [self.prefix_hints] if isinstance(self.prefix_hints, (PrefixHint, Mapping)) else list(self.prefix_hints)
        if self.prefix_hint is not None:
            hints.insert(0, self.prefix_hint)
        normalized_hints: list[PrefixHint] = []
        for hint in hints:
            normalized_hints.append(_normalize_prefix_hint(hint))
        object.__setattr__(self, "prefix_hints", tuple(normalized_hints))
        object.__setattr__(self, "prefix_hint", normalized_hints[0] if normalized_hints else None)
        stops = _normalize_stop_sources(self.stop_sequences, self.stop, self.stop_strings)
        normalized_stops: list[bytes] = []
        for value in stops:
            if isinstance(value, bytes):
                normalized_stops.append(value)
            elif isinstance(value, str):
                normalized_stops.append(value.encode("utf-8"))
            elif isinstance(value, (bytearray, memoryview)):
                normalized_stops.append(bytes(value))
            elif isinstance(value, Integral):
                normalized_stops.append(bytes((int(value) % 256,)))
            else:
                raise TypeError("stop sequences must be bytes, strings, or integer byte markers")
        object.__setattr__(self, "stop_sequences", tuple(normalized_stops))
        object.__setattr__(self, "stop", tuple(normalized_stops))
        object.__setattr__(self, "stop_strings", tuple(normalized_stops))

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
            support_rows=self.support_rows,
            prefix_hints=self.prefix_hints,
            context_length=self.context_length,
            stop_sequences=self.stop_sequences,
            seed=self.seed,
            shard=self.shard,
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


def _token_value(token: object) -> int:
    """Convert a token to the tiny model's deterministic integer domain.

    Integer and boolean token ids retain their normal arithmetic behaviour.
    Strings and bytes are accepted for byte/Unicode edge cases and map to a
    stable UTF-8 checksum.  This is intentionally independent of Python's
    process-randomized ``hash`` implementation.
    """

    if isinstance(token, bool):
        return int(token)
    if isinstance(token, Integral):
        return int(token)
    if isinstance(token, bytes):
        return sum((index + 1) * value for index, value in enumerate(token))
    if isinstance(token, str):
        return sum((index + 1) * value for index, value in enumerate(token.encode("utf-8")))
    raise ValueError("prompt tokens must be integer, bytes, or string values")


def _effective_prompt(request: EvalRequest) -> tuple[Any, ...]:
    if request.context_length is None:
        return tuple(request.prompt)
    # Causal models retain the right-most context when a prompt is too long.
    return tuple(request.prompt[-request.context_length :]) if request.context_length else ()


def _validate_prefix_hints(request: EvalRequest, prompt: Sequence[Any]) -> None:
    seen: dict[str, tuple[Any, ...]] = {}
    for hint in request.prefix_hints:
        tokens = tuple(hint.tokens)
        if tuple(prompt[: len(tokens)]) != tokens:
            raise ValueError("prefix hint conflicts with prompt")
        previous = seen.get(str(hint.name))
        if previous is not None and previous != tokens:
            raise ValueError("conflicting prefix hints")
        seen[str(hint.name)] = tokens


def _apply_stop_bytes(
    tokens: Sequence[int],
    stop_sequences: Sequence[bytes],
    minimum: int = 0,
) -> tuple[int, ...]:
    """Apply byte-level stop markers without violating minimum length."""

    if not stop_sequences:
        return tuple(int(token) for token in tokens)
    output: list[int] = []
    starts: list[int] = []
    encoded = bytearray()
    for index, token in enumerate(tokens):
        starts.append(len(encoded))
        value = int(token)
        output.append(value)
        encoded.append(value % 256)
        if index + 1 < minimum:
            continue
        for marker in stop_sequences:
            if not marker or not encoded.endswith(marker):
                continue
            marker_start = len(encoded) - len(marker)
            cut = index
            for token_index, token_start in enumerate(starts):
                token_end = starts[token_index + 1] if token_index + 1 < len(starts) else len(encoded)
                if token_start <= marker_start < token_end:
                    cut = token_index
                    break
            if cut >= minimum:
                return tuple(output[:cut])
    return tuple(output)


def _validate_request(request: EvalRequest, vocab_size: int) -> None:
    if not isinstance(request.max_new_tokens, Integral) or isinstance(request.max_new_tokens, bool):
        raise ValueError("max_new_tokens must be an integer")
    if not isinstance(request.min_new_tokens, Integral) or isinstance(request.min_new_tokens, bool):
        raise ValueError("min_new_tokens must be an integer")
    if request.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if request.min_new_tokens < 0:
        raise ValueError("min_new_tokens must be non-negative")
    if request.min_new_tokens > request.max_new_tokens:
        raise ValueError("min_new_tokens cannot exceed max_new_tokens")
    if isinstance(request.temperature, bool) or not isinstance(request.temperature, Real) or not math.isfinite(float(request.temperature)):
        raise ValueError("temperature must be finite")
    if request.eos_token_id is not None and (
        not isinstance(request.eos_token_id, Integral)
        or isinstance(request.eos_token_id, bool)
        or not 0 <= int(request.eos_token_id) < vocab_size
    ):
        raise ValueError("eos_token_id is outside the vocabulary")
    if not math.isfinite(float(request.weight)) or float(request.weight) < 0:
        raise ValueError("weight must be a finite non-negative number")
    if request.context_length is not None:
        if not isinstance(request.context_length, Integral) or isinstance(request.context_length, bool):
            raise ValueError("context_length must be a positive integer or None")
        if request.context_length < 0:
            raise ValueError("context_length must be non-negative")
    if request.seed is not None and (not isinstance(request.seed, Integral) or isinstance(request.seed, bool)):
        raise ValueError("seed must be an integer or None")
    if request.shard is not None and (not isinstance(request.shard, Integral) or isinstance(request.shard, bool)):
        raise ValueError("shard must be an integer or None")
    prompt = _effective_prompt(request)
    _validate_prefix_hints(request, prompt)
    for token in prompt:
        _token_value(token)
    for support in request.support_rows:
        for token in support.prompt + support.continuation:
            _token_value(token)


def _context_atom(value: object) -> tuple[object, ...]:
    """Return a deterministic, type-sensitive context component.

    Python's equality makes ``True`` and ``1`` interchangeable.  Context
    fields are tenant/model/adapter identifiers, so treating those values as
    the same would leak a cached prefix across isolated contexts.  Keeping the
    fully-qualified type and a stable textual representation also handles
    Unicode and non-string identifiers without relying on process-randomized
    ``hash`` values.
    """

    value_type = type(value)
    return (
        value_type.__module__,
        value_type.__qualname__,
        repr(value),
    )


def _context_key(request: EvalRequest) -> tuple[object, ...]:
    temperature = float(request.temperature)
    return (
        _context_atom(request.model),
        _context_atom(request.adapter),
        _context_atom(request.tenant),
        ("temperature", temperature),
        request.context_length,
        request.seed,
        request.shard,
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

    effective = [_effective_prompt(request) for request in requests]
    lengths = [len(prompt) for prompt in effective]
    width = max(lengths, default=0)
    rows = [list(prompt) + [pad_token_id] * (width - len(prompt)) for prompt in effective]
    return rows, lengths


def pack_batch(requests: Sequence[EvalRequest]) -> tuple[list[int], list[int]]:
    """Return flattened prompts and exclusive-end offsets."""

    tokens: list[int] = []
    offsets = [0]
    for request in requests:
        # Keep the original token object in the packed layout. Arithmetic
        # generation can coerce it later, while cache identity must keep bool
        # and int prompt values distinct.
        tokens.extend(_effective_prompt(request))
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
            + (0 if request.seed is None else int(request.seed) * 7)
            + (0 if request.shard is None else int(request.shard) * 11)
        ) % 1009
        return (context + 97) % 1009

    def _prefix_state(self, request: EvalRequest) -> int:
        prompt = _effective_prompt(request)
        _validate_prefix_hints(request, prompt)
        context = _context_key(request)
        state = self._initial_state(request)
        prefix: tuple[object, ...] = ()
        for index, token in enumerate(prompt):
            prefix = prefix + (token,)
            key = (context, tuple(_token_key(value) for value in prefix))
            if self.shared_prefix_cache and key in self._prefix_cache:
                state = self._prefix_cache[key]
                self.stats.prefix_cache_hits += 1
                continue
            state = (state * 31 + (index + 1) * _token_value(token) + 17) % 1009
            if self.shared_prefix_cache:
                self._prefix_cache[key] = state
            self.stats.prefix_cache_misses += 1
        if not prompt and self.shared_prefix_cache:
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
        return _apply_stop_bytes(output, request.stop_sequences, min(request.min_new_tokens, request.max_new_tokens))

    def _sequence_key(self, request: EvalRequest) -> tuple[object, ...]:
        return (
            _context_key(request),
            tuple(_token_key(token) for token in _effective_prompt(request)),
            int(request.max_new_tokens),
            int(request.min_new_tokens),
            None if request.eos_token_id is None else _token_key(request.eos_token_id),
            tuple(request.stop_sequences),
            tuple((hint.name, tuple(_token_key(token) for token in hint.tokens)) for hint in request.prefix_hints),
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
                if tuple(row[:length]) != tuple(_effective_prompt(request)):
                    raise AssertionError("padded representation changed a prompt")
        else:
            tokens, offsets = pack_batch(requests)
            self.stats.packed_tokens += len(tokens)
            for index, request in enumerate(requests):
                if tuple(tokens[offsets[index] : offsets[index + 1]]) != tuple(_effective_prompt(request)):
                    raise AssertionError("packed representation changed a prompt")

        outputs: list[Optional[tuple[int, ...]]] = [None] * len(requests)
        for index, request in enumerate(requests):
            # A zero-token continuation is a valid no-op and must not invoke
            # model or prefix work.  Keeping it out of the sequence cache also
            # avoids a zero-length entry being mistaken for a generated row.
            _validate_request(request, self.vocab_size)
            if request.max_new_tokens == 0:
                outputs[index] = ()
                continue
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
    total_weight = math.fsum(float(request.weight) for request in requests)
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
        exact_weight = math.fsum((exact_weight, weight * exact))
        token_weight = math.fsum((token_weight, weight * token_score))
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


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def sequence_logprob(
    request: EvalRequest,
    continuation: Sequence[Any] | None = None,
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> float:
    """Compute a deterministic causal log-probability for a continuation.

    The tiny model has no learned weights, so logits are generated from the
    independent oracle's prompt state.  The routine is useful for parity and
    contrastive metrics and deliberately avoids sampling or global state.
    """

    _validate_request(request, vocab_size)
    if continuation is None and request.continuation is not None:
        continuation = request.continuation
    target = tuple(oracle_generate(request.as_oracle_request(), vocab_size=vocab_size) if continuation is None else continuation)
    state = oracle_prompt_state(request.as_oracle_request(), vocab_size=vocab_size)
    total = 0.0
    temperature = max(abs(float(request.temperature)), 1e-6)
    for index, token in enumerate(target):
        target_value = _token_value(token) % vocab_size
        logits = [
            -abs(((state + 13 * index + 3 + candidate) % vocab_size) - candidate) / temperature
            for candidate in range(vocab_size)
        ]
        total += logits[target_value] - _logsumexp(logits)
        state = (state * 37 + target_value + index + 11) % 1009
    return float(total)


def logprob_metrics(
    requests: Sequence[EvalRequest],
    outputs: Sequence[Sequence[Any]] | None = None,
    *,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Return weighted and unweighted continuation log-probability metrics."""

    if outputs is not None and len(requests) != len(outputs):
        raise ValueError("one output is required for every request position")
    values = [
        sequence_logprob(request, None if outputs is None else outputs[index])
        for index, request in enumerate(requests)
    ]
    row_weights = [float(request.weight) for request in requests] if weights is None else [float(value) for value in weights]
    if len(row_weights) != len(values):
        raise ValueError("weights must match requests")
    if any(not math.isfinite(weight) or weight < 0 for weight in row_weights):
        raise ValueError("weights must be finite and non-negative")
    total = sum(row_weights)
    return {
        "mean_logprob": math.fsum(values) / len(values) if values else 0.0,
        "weighted_logprob": math.fsum(value * weight for value, weight in zip(values, row_weights)) / total if total else 0.0,
        "total_weight": total,
        "num_requests": float(len(values)),
    }


# Common scorer spellings used by evaluator integrations.
logprob = sequence_logprob
conditional_logprob = sequence_logprob
score_logprob = sequence_logprob


def pmi_score(
    request: EvalRequest,
    continuation: Sequence[Any] | None = None,
    *,
    baseline_prompt: Sequence[Any] = (),
) -> float:
    """Pointwise mutual information, ``log p(y|x) - log p(y)``."""

    conditional = sequence_logprob(request, continuation)
    baseline = EvalRequest(
        request_id=request.request_id,
        prompt=tuple(baseline_prompt),
        model=request.model,
        adapter=request.adapter,
        tenant=request.tenant,
        temperature=request.temperature,
        max_new_tokens=request.max_new_tokens,
        min_new_tokens=request.min_new_tokens,
        eos_token_id=request.eos_token_id,
        weight=request.weight,
        reference=request.reference,
        context_length=request.context_length,
        seed=request.seed,
        shard=request.shard,
    )
    return float(conditional - sequence_logprob(baseline, continuation))


def dc_pmi_score(
    request: EvalRequest,
    continuation: Sequence[Any] | None = None,
    *,
    support_rows: Iterable[Any] | None = None,
) -> float:
    """Debiased contrastive PMI using the mean support conditional score.

    With no support rows this is ordinary PMI against an empty baseline.  An
    empty support list is therefore well-defined and does not fabricate a
    denominator.  Duplicate support rows remain duplicate observations.
    """

    pmi = pmi_score(request, continuation)
    supports = normalize_support_rows(request.support_rows if support_rows is None else support_rows)
    if not supports:
        return pmi
    values: list[float] = []
    support_weights: list[float] = []
    for row in supports:
        support_request = EvalRequest(
            request_id=row.request_id,
            prompt=row.prompt,
            model=request.model,
            adapter=request.adapter,
            tenant=request.tenant,
            temperature=request.temperature,
            max_new_tokens=request.max_new_tokens,
            min_new_tokens=request.min_new_tokens,
            eos_token_id=request.eos_token_id,
            context_length=request.context_length,
        )
        target = row.continuation if continuation is None else tuple(continuation)
        if row.weight <= 0:
            continue
        values.append(pmi_score(support_request, target))
        support_weights.append(float(row.weight))
    if not values:
        return float(pmi)
    baseline = math.fsum(value * weight for value, weight in zip(values, support_weights)) / math.fsum(support_weights)
    return float(pmi - baseline)


# Short aliases are intentionally exported because independent benchmark
# harnesses use both noun and verb forms.
pmi = pmi_score
dc_pmi = dc_pmi_score


def batch_calibration(
    predictions: Sequence[float],
    labels: Sequence[float | int | bool],
    *,
    weights: Sequence[float] | None = None,
    bins: int = 10,
) -> dict[str, float]:
    """Compute deterministic weighted calibration diagnostics.

    Predictions are clipped to the closed unit interval once, labels are
    normalized to 0/1, and all aggregate values remain finite on empty input.
    """

    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have equal length")
    if bins <= 0:
        raise ValueError("bins must be positive")
    row_weights = [1.0] * len(predictions) if weights is None else [float(value) for value in weights]
    if len(row_weights) != len(predictions):
        raise ValueError("weights must match predictions")
    if any(not math.isfinite(weight) or weight < 0 for weight in row_weights):
        raise ValueError("weights must be finite and non-negative")
    probs = [min(1.0, max(0.0, float(probability))) for probability in predictions]
    truths = [1.0 if bool(label) else 0.0 for label in labels]
    total = math.fsum(row_weights)
    if not probs or total <= 0:
        return {"brier": 0.0, "logloss": 0.0, "ece": 0.0, "accuracy": 0.0, "total_weight": total, "num_rows": float(len(probs))}
    brier = math.fsum(weight * (probability - truth) ** 2 for probability, truth, weight in zip(probs, truths, row_weights)) / total
    eps = 1e-15
    logloss = -math.fsum(weight * (truth * math.log(max(eps, probability)) + (1 - truth) * math.log(max(eps, 1 - probability))) for probability, truth, weight in zip(probs, truths, row_weights)) / total
    accuracy = math.fsum(weight for probability, truth, weight in zip(probs, truths, row_weights) if (probability >= 0.5) == bool(truth)) / total
    ece = 0.0
    for bin_index in range(bins):
        low = bin_index / bins
        high = (bin_index + 1) / bins
        selected = [index for index, probability in enumerate(probs) if low <= probability < high or (bin_index == bins - 1 and probability == high)]
        weight = math.fsum(row_weights[index] for index in selected)
        if weight:
            mean_probability = math.fsum(probs[index] * row_weights[index] for index in selected) / weight
            mean_truth = math.fsum(truths[index] * row_weights[index] for index in selected) / weight
            ece += weight / total * abs(mean_probability - mean_truth)
    return {"brier": float(brier), "logloss": float(logloss), "ece": float(ece), "accuracy": float(accuracy), "total_weight": total, "num_rows": float(len(probs))}


calibration = batch_calibration


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

    def evaluate(self, requests: Iterable[EvalRequest], references: Iterable[Sequence[Any]] | None = None, *, mode: str = "packed") -> dict[str, Any]:
        """Return a v2-compatible parity report with v3 diagnostics."""
        rows = list(requests)
        refs = None if references is None else [tuple(value) for value in references]
        if refs is not None and len(refs) != len(rows):
            raise ValueError("references must align with requests")
        scoring_rows = rows if refs is None else [replace(row, reference=refs[index]) for index, row in enumerate(rows)]
        result = self.run(scoring_rows, mode=mode)
        expected = [refs[i] if refs is not None else tuple(oracle_generate(row.as_oracle_request())) for i, row in enumerate(rows)]
        padded = self.runtime.generate_batch(rows, mode="padded")
        return {"outputs": result.outputs, "predictions": result.outputs, "padded": padded, "packed": result.outputs, "mode_parity": padded == result.outputs, "oracle_parity": result.outputs == [list(value) for value in expected], "metrics": result.metrics, "records": result.records, "exact_match": result.metrics["weighted_exact_match"], "token_accuracy": result.metrics["weighted_token_accuracy"], "total_weight": result.metrics["total_weight"]}

    def logprob(self, request: EvalRequest, continuation: Sequence[Any] | None = None) -> float:
        return sequence_logprob(request, continuation)

    def pmi(self, request: EvalRequest, continuation: Sequence[Any] | None = None) -> float:
        return pmi_score(request, continuation)

    def dc_pmi(self, request: EvalRequest, continuation: Sequence[Any] | None = None, *, support_rows: Iterable[Any] | None = None) -> float:
        return dc_pmi_score(request, continuation, support_rows=support_rows)

    def calibrate(self, predictions: Sequence[float], labels: Sequence[float | int | bool], *, weights: Sequence[float] | None = None, bins: int = 10) -> dict[str, float]:
        return batch_calibration(predictions, labels, weights=weights, bins=bins)


__all__ = [
    "EvalRequest",
    "EvaluationResult",
    "PrefixHint",
    "RuntimeStats",
    "SupportRow",
    "TinyBatchRuntime",
    "TinyEvaluator",
    "batch_calibration",
    "calibration",
    "dc_pmi",
    "dc_pmi_score",
    "logprob_metrics",
    "logprob",
    "conditional_logprob",
    "score_logprob",
    "normalize_support_rows",
    "pack_batch",
    "pad_batch",
    "pmi",
    "pmi_score",
    "sequence_logprob",
    "weighted_metrics",
]
