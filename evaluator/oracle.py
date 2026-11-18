"""Independent deterministic oracle for the local batched-eval task.

The oracle intentionally has no dependency on ``repo.llm_runtime`` or the
candidate evaluator.  It mirrors the tiny causal model one request at a time,
including context truncation, prefix-hint validation, EOS/minimum-length and
byte-level stop handling.  Keeping this implementation separate prevents a
bug in the runtime's batching or cache path from becoming its own expected
value.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Iterable, Mapping, Optional, Sequence


DEFAULT_VOCAB_SIZE = 31
DEFAULT_EOS_TOKEN_ID = 0


@dataclass(frozen=True)
class OracleRequest:
    """Input fields used by the independent oracle.

    New v3 fields are appended so v1/v2 positional construction remains
    valid.  Support rows and prefix hints are validated for consistency but do
    not alter the tiny model's logits.
    """

    request_id: Any
    prompt: tuple[Any, ...]
    model: Any = "tiny"
    adapter: Any = "base"
    tenant: Any = "default"
    temperature: float = 0.0
    max_new_tokens: int = 4
    min_new_tokens: int = 0
    eos_token_id: Optional[int] = DEFAULT_EOS_TOKEN_ID
    weight: float = 1.0
    reference: Optional[tuple[Any, ...]] = None
    support_rows: tuple[Any, ...] = ()
    prefix_hints: tuple[Any, ...] = ()
    context_length: Optional[int] = None
    stop_sequences: tuple[Any, ...] = ()
    seed: Optional[int] = None
    shard: Optional[int] = None
    stop: tuple[Any, ...] = ()
    # Compatibility aliases accepted by evaluator harnesses.
    support: tuple[Any, ...] = ()
    context_window: Optional[int] = None
    max_context_tokens: Optional[int] = None
    continuation: Optional[tuple[Any, ...]] = None
    stop_strings: tuple[Any, ...] = ()
    # Compatibility aliases used by older hidden harnesses.
    support: tuple[Any, ...] = ()
    context_window: Optional[int] = None
    max_context_tokens: Optional[int] = None
    continuation: Optional[tuple[Any, ...]] = None
    stop_strings: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", tuple(self.prompt))
        if self.reference is not None:
            object.__setattr__(self, "reference", tuple(self.reference))
        def _items(value: Any) -> tuple[Any, ...]:
            if value is None:
                return ()
            if isinstance(value, (str, bytes, bytearray, memoryview, Integral, Mapping)):
                return (value,)
            if hasattr(value, "tokens") or hasattr(value, "prompt"):
                return (value,)
            return tuple(value)

        rows = _items(self.support_rows) + _items(self.support)
        object.__setattr__(self, "support_rows", rows)
        object.__setattr__(self, "support", rows)
        object.__setattr__(self, "prefix_hints", _items(self.prefix_hints))
        contexts = [value for value in (self.context_length, self.context_window, self.max_context_tokens) if value is not None]
        if contexts and any(value != contexts[0] for value in contexts):
            raise ValueError("context length aliases disagree")
        if contexts:
            object.__setattr__(self, "context_length", contexts[0])
            object.__setattr__(self, "context_window", contexts[0])
            object.__setattr__(self, "max_context_tokens", contexts[0])
        if self.continuation is not None:
            object.__setattr__(self, "continuation", tuple(self.continuation))
        values = _items(self.stop_sequences) + _items(self.stop) + _items(self.stop_strings)
        normalized: list[Any] = []
        for value in values:
            if isinstance(value, str):
                normalized.append(value.encode("utf-8"))
            elif isinstance(value, (bytes, bytearray, memoryview)):
                normalized.append(bytes(value))
            else:
                normalized.append(value)
        object.__setattr__(self, "stop_sequences", tuple(normalized))
        object.__setattr__(self, "stop", tuple(normalized))
        object.__setattr__(self, "stop_strings", tuple(normalized))
        object.__setattr__(self, "stop_strings", tuple(normalized))


def _stable_text(value: object) -> int:
    """Return a process-independent small checksum for a context value."""

    text = str(value)
    return sum((index + 1) * ord(char) for index, char in enumerate(text)) % 1009


def _token_value(token: object) -> int:
    """Map numeric, Unicode, and byte tokens to the tiny arithmetic domain."""

    if isinstance(token, Integral):
        return int(token)
    if isinstance(token, bytes):
        return sum((index + 1) * value for index, value in enumerate(token))
    if isinstance(token, bytearray):
        return sum((index + 1) * value for index, value in enumerate(bytes(token)))
    if isinstance(token, memoryview):
        return sum((index + 1) * value for index, value in enumerate(token.tobytes()))
    if isinstance(token, str):
        return sum((index + 1) * value for index, value in enumerate(token.encode("utf-8")))
    raise ValueError("prompt tokens must be integer, bytes, or string values")


def _effective_prompt(request: OracleRequest) -> tuple[Any, ...]:
    if request.context_length is None:
        return tuple(request.prompt)
    limit = int(request.context_length)
    return tuple(request.prompt[-limit:]) if limit else ()


def _hint_parts(hint: Any) -> tuple[tuple[Any, ...], str]:
    if isinstance(hint, Mapping):
        return tuple(hint.get("tokens", hint.get("prompt", ()))), str(hint.get("name", "default"))
    tokens = getattr(hint, "tokens", hint)
    name = getattr(hint, "name", "default")
    return tuple(tokens), str(name)


def _validate_prefix_hints(request: OracleRequest, prompt: Sequence[Any]) -> None:
    seen: dict[str, tuple[Any, ...]] = {}
    for hint in request.prefix_hints:
        tokens, name = _hint_parts(hint)
        if tuple(prompt[: len(tokens)]) != tokens:
            raise ValueError("prefix hint conflicts with prompt")
        previous = seen.get(name)
        if previous is not None and previous != tokens:
            raise ValueError("conflicting prefix hints")
        seen[name] = tokens


def _support_parts(row: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    if isinstance(row, Mapping):
        return (
            tuple(row.get("prompt", row.get("tokens", ()))),
            tuple(row.get("continuation", row.get("target", row.get("completion", ())))),
        )
    return tuple(getattr(row, "prompt", ())), tuple(getattr(row, "continuation", ()))


def _normalize_stop(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        return value.encode("utf-8")
    # Integer markers are interpreted as one raw byte by the tiny runtime.
    if isinstance(value, Integral):
        return bytes((int(value) % 256,))
    return bytes(value)


def _stop_values(request: OracleRequest) -> tuple[bytes, ...]:
    values: list[Any] = []
    for source in (request.stop_sequences, request.stop):
        if source is None:
            continue
        if isinstance(source, (str, bytes, bytearray, memoryview, Integral)):
            values.append(source)
        else:
            values.extend(source)
    return tuple(marker for marker in (_normalize_stop(value) for value in values) if marker)


def _apply_stop_bytes(
    tokens: Sequence[int],
    stop_sequences: Sequence[bytes],
    minimum: int = 0,
) -> tuple[int, ...]:
    """Apply byte-level markers, excluding the whole marker from output.

    Markers may span token boundaries.  We therefore retain byte offsets for
    each token and cut before the token containing the first marker byte.
    ``minimum`` gates stopping just like EOS.
    """

    output: list[int] = []
    starts: list[int] = []
    encoded = bytearray()
    for index, token in enumerate(tokens):
        starts.append(len(encoded))
        output.append(int(token))
        value = int(token)
        if 0 <= value <= 255:
            encoded.append(value)
        else:
            encoded.extend(str(value).encode("utf-8"))
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


def _validate_request(request: OracleRequest, vocab_size: int) -> None:
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
    if request.eos_token_id is not None:
        if isinstance(request.eos_token_id, bool) or not isinstance(request.eos_token_id, Integral) or not 0 <= int(request.eos_token_id) < vocab_size:
            raise ValueError("eos_token_id is outside the vocabulary")
    if not math.isfinite(float(request.weight)) or float(request.weight) < 0:
        raise ValueError("weight must be a finite non-negative number")
    if request.context_length is not None:
        if not isinstance(request.context_length, Integral) or isinstance(request.context_length, bool) or request.context_length < 0:
            raise ValueError("context_length must be a non-negative integer or None")
    if request.seed is not None and (not isinstance(request.seed, Integral) or isinstance(request.seed, bool)):
        raise ValueError("seed must be an integer or None")
    if request.shard is not None and (not isinstance(request.shard, Integral) or isinstance(request.shard, bool)):
        raise ValueError("shard must be an integer or None")
    prompt = _effective_prompt(request)
    _validate_prefix_hints(request, prompt)
    for token in prompt:
        _token_value(token)
    for row in request.support_rows:
        if isinstance(row, Mapping):
            values = (row.get("prompt", row.get("tokens", ())), row.get("continuation", row.get("target", row.get("completion", ()))))
        elif hasattr(row, "prompt") and hasattr(row, "continuation"):
            values = (row.prompt, row.continuation)
        else:
            values = (row,)
        for sequence in values:
            for token in sequence:
                _token_value(token)


def prompt_state(
    request: OracleRequest,
    prompt: Sequence[Any] | None = None,
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> int:
    """Compute the deterministic state after ``prompt``."""

    _validate_request(request, vocab_size)
    tokens = _effective_prompt(request) if prompt is None else tuple(prompt)
    _validate_prefix_hints(request, tokens)
    context = (
        _stable_text(request.model)
        + 3 * _stable_text(request.adapter)
        + 5 * _stable_text(request.tenant)
        + int(round(float(request.temperature) * 1000))
        + (0 if request.seed is None else int(request.seed) * 7)
        + (0 if request.shard is None else int(request.shard) * 11)
    ) % 1009
    state = (context + 97) % 1009
    for index, token in enumerate(tokens):
        state = (state * 31 + (index + 1) * _token_value(token) + 17) % 1009
    return state


def _next_token(state: int, step: int, vocab_size: int) -> int:
    return (state + 13 * step + 3) % vocab_size


def generate(
    request: OracleRequest,
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> tuple[int, ...]:
    """Generate the reference token sequence for one request."""

    _validate_request(request, vocab_size)
    state = prompt_state(request, vocab_size=vocab_size)
    generated: list[int] = []
    for step in range(int(request.max_new_tokens)):
        token = _next_token(state, step, vocab_size)
        generated.append(token)
        state = (state * 37 + token + step + 11) % 1009
        if (
            request.eos_token_id is not None
            and token == int(request.eos_token_id)
            and len(generated) >= int(request.min_new_tokens)
        ):
            break
    return _apply_stop_bytes(generated, _stop_values(request), min(request.min_new_tokens, request.max_new_tokens))


def generate_batch(
    requests: Iterable[OracleRequest],
    *,
    mode: str = "packed",
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> list[tuple[int, ...]]:
    """Generate outputs in caller order for either representation mode."""

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
