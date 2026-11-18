"""Dependency-free reference evaluator for the batched generation task.

The oracle deliberately does not call a candidate model to compute the expected
sequence.  It mirrors the small arithmetic contract documented by ``ToyModel``
and applies EOS/minimum-length handling itself.  The evaluator is also used by
the private tests to keep scoring independent from engine cache state.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    request: Any
    expected: tuple
    weight: float = 1.0


def oracle_tokens(
    model_id: str,
    adapter: str,
    prompt: Sequence[int],
    temperature: float,
    max_new_tokens: int,
    min_new_tokens: int = 0,
    eos_token_id: Any = None,
) -> tuple:
    """Return the expected deterministic ToyModel continuation.

    EOS is considered only after ``min_new_tokens`` have been emitted.  The
    result never exceeds ``max_new_tokens`` and is immutable for safe reuse.
    """
    salt = sum(ord(char) for char in (model_id + adapter)) % 97
    seed = (sum(prompt) + salt + int(temperature * 1000)) % 1000
    raw = [((seed + index * 17) % 101) for index in range(max_new_tokens)]
    if eos_token_id is None:
        return tuple(raw)
    result = []
    minimum = max(0, min(int(min_new_tokens), int(max_new_tokens)))
    for token in raw:
        result.append(token)
        if len(result) >= minimum and token == eos_token_id:
            break
    return tuple(result)


def weighted_exact_match(
    predictions: Mapping[str, Sequence[int]], cases: Iterable[EvalCase]
) -> float:
    """Compute a weighted exact-match score without normalizing by item count."""
    numerator = 0.0
    denominator = 0.0
    for case in cases:
        weight = float(case.weight)
        if weight < 0:
            raise ValueError("case weights must be non-negative")
        denominator += weight
        if tuple(predictions[case.case_id]) == tuple(case.expected):
            numerator += weight
    return numerator / denominator if denominator else 0.0


class TinyEvaluator:
    """Run both engine batch modes and return order-preserving predictions."""

    MODES = ("padded", "packed")

    def run(self, engine: Any, cases: Sequence[EvalCase], mode: str = "packed"):
        if mode not in self.MODES:
            raise ValueError("mode must be 'padded' or 'packed'")
        requests = [case.request for case in cases]
        outputs = engine.generate_batch(requests, mode=mode)
        if len(outputs) != len(cases):
            raise AssertionError("engine returned a different number of outputs")
        return {case.case_id: tuple(output) for case, output in zip(cases, outputs)}

    def score(self, engine: Any, cases: Sequence[EvalCase], mode: str = "packed"):
        predictions = self.run(engine, cases, mode=mode)
        return weighted_exact_match(predictions, cases)

