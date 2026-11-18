"""Numerically stable scoring helpers for the v3 tiny causal-LM evaluator.

The model used by this repository is intentionally tiny, but the contracts in
the benchmark mirror a serving evaluator.  This module keeps probability and
calibration math independent from batching and cache state.  It accepts both
``EvalRequest`` objects and plain prompt/continuation sequences so private
callers can use the helpers without depending on one constructor shape.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Iterable, Mapping, Sequence

from .oracle import DEFAULT_VOCAB_SIZE


def _value(token: Any) -> int:
    if isinstance(token, Integral):
        return int(token)
    if isinstance(token, bytes):
        return sum((i + 1) * b for i, b in enumerate(token))
    if isinstance(token, str):
        return sum((i + 1) * b for i, b in enumerate(token.encode("utf-8")))
    if isinstance(token, (bytearray, memoryview)):
        return sum((i + 1) * b for i, b in enumerate(bytes(token)))
    raise ValueError("tokens must be integer, bytes, or string values")


def _stable_text(value: Any) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(str(value))) % 1009


def _prompt_of(request_or_prompt: Any) -> tuple[Any, ...]:
    if hasattr(request_or_prompt, "prompt"):
        prompt = request_or_prompt.prompt
        limit = getattr(request_or_prompt, "context_length", None)
    elif isinstance(request_or_prompt, Mapping):
        prompt = request_or_prompt.get("prompt", ())
        limit = request_or_prompt.get("context_length")
    else:
        prompt, limit = request_or_prompt, None
    prompt = tuple(prompt)
    if limit is not None:
        limit = int(limit)
        prompt = prompt[-limit:] if limit else ()
    return prompt


def _context(request_or_prompt: Any) -> tuple[int, float]:
    model = getattr(request_or_prompt, "model", "tiny")
    adapter = getattr(request_or_prompt, "adapter", "base")
    tenant = getattr(request_or_prompt, "tenant", "default")
    temperature = float(getattr(request_or_prompt, "temperature", 0.0))
    seed = int(getattr(request_or_prompt, "seed", 0) or 0)
    shard = int(getattr(request_or_prompt, "shard", 0) or 0)
    state = (_stable_text(model) + 3 * _stable_text(adapter)
             + 5 * _stable_text(tenant) + seed * 7 + shard * 11) % 1009
    return (state + 97) % 1009, temperature


def _state_after(prompt: Sequence[Any], request_or_prompt: Any) -> int:
    state, _ = _context(request_or_prompt)
    for index, token in enumerate(prompt):
        state = (state * 31 + (index + 1) * _value(token) + 17) % 1009
    return state


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(v - maximum) for v in values))


def _token_logprob(state: int, token: Any, temperature: float, vocab_size: int) -> float:
    """Return a deterministic categorical log-probability for one token."""

    token_id = _value(token) % vocab_size
    # At temperature zero use a finite, deterministic low-temperature limit.
    scale = max(abs(float(temperature)), 1e-3)
    logits = [-abs(((state + 17 * index + 3) % vocab_size) - token_id) / scale
              for index in range(vocab_size)]
    return logits[token_id] - _logsumexp(logits)


def sequence_logprob(
    request_or_prompt: Any,
    continuation: Sequence[Any] | None = None,
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    normalize: bool = False,
) -> float:
    """Score a continuation conditioned on a request's prompt.

    ``request_or_prompt`` can be an ``EvalRequest`` or a raw prompt.  When a
    request carries no explicit continuation, ``reference`` is used when
    present.  Empty continuations have log-probability zero.
    """

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    request = request_or_prompt
    if continuation is None:
        continuation = getattr(request, "reference", None)
        if continuation is None and isinstance(request, Mapping):
            continuation = request.get("continuation", request.get("reference", ()))
        continuation = () if continuation is None else continuation
    continuation = tuple(continuation)
    state = _state_after(_prompt_of(request), request)
    _, temperature = _context(request)
    values: list[float] = []
    for index, token in enumerate(continuation):
        values.append(_token_logprob(state, token, temperature, vocab_size))
        token_id = _value(token) % vocab_size
        state = (state * 37 + token_id + index + 11) % 1009
    total = math.fsum(values)
    return total / len(values) if normalize and values else total


logprob = sequence_logprob
conditional_logprob = sequence_logprob
score_logprob = sequence_logprob


def _support_values(request: Any, support: Iterable[Any] | None) -> tuple[Any, ...]:
    if support is None:
        support = getattr(request, "support_rows", ())
    return tuple(support or ())


def _support_parts(row: Any) -> tuple[tuple[Any, ...], tuple[Any, ...], float]:
    if isinstance(row, Mapping):
        prompt = row.get("prompt", row.get("input", row.get("tokens", ())))
        continuation = row.get("continuation", row.get("target", row.get("completion", row.get("output", ()))))
        weight = row.get("weight", 1.0)
    else:
        prompt = getattr(row, "prompt", ())
        continuation = getattr(row, "continuation", ())
        weight = getattr(row, "weight", 1.0)
    weight = float(weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("support weights must be finite and non-negative")
    return tuple(prompt), tuple(continuation), weight


def pmi_score(
    prompt: Sequence[Any] | Any,
    continuation: Sequence[Any],
    *,
    request: Any | None = None,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    normalize: bool = False,
) -> float:
    """Pointwise mutual information ``log P(y|x) - log P(y)``."""

    base = request if request is not None else prompt
    conditional = sequence_logprob(base, continuation, vocab_size=vocab_size, normalize=normalize)
    if request is None:
        # Preserve context metadata when a request object was not supplied.
        unconditional = sequence_logprob((), continuation, vocab_size=vocab_size, normalize=normalize)
    else:
        class _Empty:
            pass
        empty = _Empty()
        for key in ("model", "adapter", "tenant", "temperature", "seed", "shard"):
            setattr(empty, key, getattr(request, key, None))
        empty.prompt = ()
        empty.context_length = None
        unconditional = sequence_logprob(empty, continuation, vocab_size=vocab_size, normalize=normalize)
    return conditional - unconditional


pmi = pmi_score


def dc_pmi_score(
    prompt: Sequence[Any] | Any,
    continuation: Sequence[Any] | None = None,
    *,
    request: Any | None = None,
    support: Iterable[Any] | None = None,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    normalize: bool = False,
) -> float:
    """Support-corrected PMI.

    The support baseline is a weighted mean of support continuation scores.
    Empty support falls back to ordinary PMI, while zero-weight support is
    ignored. Duplicate support rows contribute duplicate weight as intended.
    """

    base = request if request is not None else prompt
    if continuation is None:
        continuation = getattr(base, "reference", ())
    value = pmi_score(prompt, continuation, request=request, vocab_size=vocab_size, normalize=normalize)
    rows = _support_values(base, support)
    weighted: list[float] = []
    weights: list[float] = []
    for row in rows:
        row_prompt, row_continuation, weight = _support_parts(row)
        if weight <= 0:
            continue
        row_request = request if request is not None else row_prompt
        weighted.append(pmi_score(row_prompt, row_continuation, request=row_request, vocab_size=vocab_size, normalize=normalize))
        weights.append(weight)
    if not weights:
        return value
    baseline = math.fsum(score * weight for score, weight in zip(weighted, weights)) / math.fsum(weights)
    return value - baseline


dc_pmi = dc_pmi_score


def batch_logprob(rows: Iterable[Any], continuations: Iterable[Sequence[Any]] | None = None, *, normalize: bool = False) -> list[float]:
    rows = list(rows)
    if continuations is None:
        continuations = [None] * len(rows)
    continuations = list(continuations)
    if len(rows) != len(continuations):
        raise ValueError("rows and continuations must align")
    return [sequence_logprob(row, continuation, normalize=normalize) for row, continuation in zip(rows, continuations)]


@dataclass(frozen=True)
class CalibrationResult:
    count: int
    total_weight: float
    mean_logprob: float
    mean_nll: float
    brier: float
    ece: float

    def as_dict(self) -> dict[str, float]:
        return {"count": self.count, "total_weight": self.total_weight,
                "mean_logprob": self.mean_logprob, "mean_nll": self.mean_nll,
                "brier": self.brier, "ece": self.ece}


def batch_calibration(
    probabilities: Sequence[float] | None = None,
    outcomes: Sequence[float | bool] | None = None,
    *,
    weights: Sequence[float] | None = None,
    bins: int = 10,
    logprobs: Sequence[float] | None = None,
) -> dict[str, float]:
    """Compute weighted Brier/ECE/NLL with stable floating-point reduction."""

    if logprobs is not None:
        probabilities = [math.exp(float(value)) for value in logprobs]
    if probabilities is None or outcomes is None:
        raise ValueError("probabilities and outcomes are required")
    probabilities, outcomes = list(probabilities), list(outcomes)
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must align")
    if weights is None:
        weights = [1.0] * len(probabilities)
    weights = [float(weight) for weight in weights]
    if len(weights) != len(probabilities) or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("weights must align and be finite non-negative")
    if bins <= 0:
        raise ValueError("bins must be positive")
    clipped: list[float] = []
    for probability in probabilities:
        probability = float(probability)
        if not math.isfinite(probability):
            raise ValueError("probabilities must be finite")
        clipped.append(min(1.0, max(0.0, probability)))
    probabilities = clipped
    total = math.fsum(weights)
    if not probabilities:
        return {"count": 0.0, "total_weight": 0.0, "mean_logprob": 0.0,
                "mean_nll": 0.0, "brier": 0.0, "ece": 0.0}
    if total <= 0:
        return {"count": float(len(probabilities)), "total_weight": total,
                "mean_logprob": 0.0, "mean_nll": 0.0, "brier": 0.0,
                "ece": 0.0}
    labels = [1.0 if bool(outcome) else 0.0 for outcome in outcomes]
    brier = math.fsum(w * (p - y) ** 2 for p, y, w in zip(probabilities, labels, weights)) / total
    nll = math.fsum(-w * (y * math.log(max(p, 1e-15)) + (1 - y) * math.log(max(1 - p, 1e-15))) for p, y, w in zip(probabilities, labels, weights)) / total
    ece_terms: list[float] = []
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        if index == bins - 1:
            members = [(p, y, w) for p, y, w in zip(probabilities, labels, weights)
                       if low <= p <= high]
        else:
            members = [(p, y, w) for p, y, w in zip(probabilities, labels, weights)
                       if low <= p < high]
        if not members:
            continue
        mass = math.fsum(item[2] for item in members)
        confidence = math.fsum(item[0] * item[2] for item in members) / mass
        accuracy = math.fsum(item[1] * item[2] for item in members) / mass
        ece_terms.append(mass / total * abs(confidence - accuracy))
    return {"count": float(len(probabilities)), "total_weight": total,
            "mean_logprob": -nll, "mean_nll": nll, "brier": brier,
            "ece": math.fsum(ece_terms)}


calibrate_batch = batch_calibration
calibration = batch_calibration
calibrate = batch_calibration
compute_logprob = sequence_logprob
compute_pmi = pmi_score
compute_dc_pmi = dc_pmi_score


__all__ = [
    "CalibrationResult", "batch_calibration", "batch_logprob", "calibrate_batch", "calibration", "calibrate",
    "compute_logprob", "compute_pmi", "compute_dc_pmi",
    "conditional_logprob", "dc_pmi", "dc_pmi_score", "logprob", "pmi", "pmi_score",
    "score_logprob", "sequence_logprob",
]
