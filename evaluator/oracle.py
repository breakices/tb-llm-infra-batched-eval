"""Independent deterministic oracle for the local batched-eval task.

The oracle intentionally has no dependency on ``repo.llm_runtime``.  It is
small enough to run in the private verifier and gives the verifier a second
implementation of the generation contract.  Prefix state is represented by a
single integer, which keeps expected outputs deterministic without requiring a
model package or network access.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence


DEFAULT_VOCAB_SIZE = 31
DEFAULT_EOS_TOKEN_ID = 0


@dataclass(frozen=True)
class OracleRequest:
    """The input fields used by the evaluator and by the independent oracle."""

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


def _stable_text(value: object) -> int:
    """Return a process-independent small checksum for a context value."""

    text = str(value)
    return sum((index + 1) * ord(char) for index, char in enumerate(text)) % 1009


def _validate_request(request: OracleRequest, vocab_size: int) -> None:
    if request.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if request.min_new_tokens < 0:
        raise ValueError("min_new_tokens must be non-negative")
    if request.min_new_tokens > request.max_new_tokens:
        raise ValueError("min_new_tokens cannot exceed max_new_tokens")
    if not math.isfinite(float(request.temperature)):
        raise ValueError("temperature must be finite")
    if request.eos_token_id is not None:
        if not 0 <= int(request.eos_token_id) < vocab_size:
            raise ValueError("eos_token_id is outside the vocabulary")


def prompt_state(
    request: OracleRequest,
    prompt: Sequence[int] | None = None,
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> int:
    """Compute the deterministic state after ``prompt``.

    This is exposed so the runtime prototype can populate its own shared
    prefix cache while keeping the verifier's expected-output path separate.
    The context fields are part of the state, so equal token prefixes from
    different tenants or adapters cannot share state accidentally.
    """

    _validate_request(request, vocab_size)
    tokens = request.prompt if prompt is None else tuple(prompt)
    context = (
        _stable_text(request.model)
        + 3 * _stable_text(request.adapter)
        + 5 * _stable_text(request.tenant)
        + int(round(float(request.temperature) * 1000))
    ) % 1009
    state = (context + 97) % 1009
    for index, token in enumerate(tokens):
        state = (state * 31 + (index + 1) * int(token) + 17) % 1009
    return state


def _next_token(state: int, step: int, vocab_size: int) -> int:
    return (state + 13 * step + 3) % vocab_size


def generate(
    request: OracleRequest,
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> tuple[int, ...]:
    """Generate the reference token sequence for one request.

    An EOS token only terminates a sequence after ``min_new_tokens`` have
    been emitted.  If no EOS is configured, generation always runs to the
    requested maximum length.
    """

    _validate_request(request, vocab_size)
    state = prompt_state(request, vocab_size=vocab_size)
    generated: list[int] = []
    for step in range(request.max_new_tokens):
        token = _next_token(state, step, vocab_size)
        generated.append(token)
        state = (state * 37 + token + step + 11) % 1009
        if (
            request.eos_token_id is not None
            and token == int(request.eos_token_id)
            and len(generated) >= request.min_new_tokens
        ):
            break
    return tuple(generated)


def generate_batch(
    requests: Iterable[OracleRequest],
    *,
    mode: str = "packed",
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> list[tuple[int, ...]]:
    """Generate outputs in caller order for either representation mode.

    ``mode`` only changes how a batch would be represented by a production
    scheduler.  The oracle deliberately computes one request at a time so
    that padding, packing, cache, and ordering bugs cannot leak into expected
    results.
    """

    if mode not in {"padded", "packed"}:
        raise ValueError("mode must be 'padded' or 'packed'")
    return [generate(request, vocab_size=vocab_size) for request in requests]


__all__ = [
    "DEFAULT_EOS_TOKEN_ID",
    "DEFAULT_VOCAB_SIZE",
    "OracleRequest",
    "generate",
    "generate_batch",
    "prompt_state",
]
