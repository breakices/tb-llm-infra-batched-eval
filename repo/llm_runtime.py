"""Deterministic, dependency-free toy runtime used by the batching task.

The module intentionally has a small public surface. ``BatchEngine`` keeps
serving concerns (batching, cache isolation, and result ownership) out of the
model, while ``TinyEvaluator`` lives in :mod:`tiny_evaluator` so it can be
used by local parity checks without a network model or a credential.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any


@dataclass(frozen=True)
class Request:
    """One deterministic generation request.

    The first six fields are the original task API. Optional fields are
    appended so old positional callers continue to work. ``request_id`` is
    deliberately excluded from cache keys because two rows with the same
    content should share work even when their ids are duplicated.
    """

    tenant: str
    model: str
    adapter: str
    prompt: tuple[int, ...]
    max_new_tokens: int = 4
    temperature: float = 0.0
    min_new_tokens: int = 0
    eos_token_id: int | None = None
    request_id: Any = None
    weight: float = 1.0


def _freeze(value: Any) -> Any:
    """Build a type-sensitive, hashable representation for cache keys."""

    if isinstance(value, tuple):
        return ("tuple", tuple(_freeze(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_freeze(item) for item in value))
    if isinstance(value, dict):
        pairs = tuple(sorted((_freeze(k), _freeze(v)) for k, v in value.items()))
        return ("dict", pairs)
    try:
        hash(value)
    except TypeError:
        return (type(value).__module__, type(value).__qualname__, repr(value))
    return (type(value).__module__, type(value).__qualname__, value)


def _is_int(value: Any) -> bool:
    """Accept integer-like configuration values while excluding booleans."""

    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_prompt_token(value: Any) -> bool:
    """Prompt tokens may be booleans, whose identity remains cache-visible."""

    return isinstance(value, Integral)


class ToyModel:
    """A deterministic model with scalar and batched execution counters."""

    def __init__(self, model_id):
        self.model_id = model_id
        self.forward_calls = 0
        self.batch_forward_calls = 0
        self.prefill_calls = 0

    def _raw_tokens(self, prompt, adapter, temperature, max_new_tokens):
        salt = sum(ord(c) for c in (str(self.model_id) + str(adapter))) % 97
        seed = (sum(prompt) + salt + int(temperature * 1000)) % 1000
        return [((seed + i * 17) % 101) for i in range(max_new_tokens)]

    def next_tokens(self, prompt, adapter, temperature, max_new_tokens):
        self.forward_calls += 1
        return self._raw_tokens(prompt, adapter, temperature, max_new_tokens)

    def next_tokens_batch(
        self, prompts, adapter, temperature, max_new_tokens, mode="padded"
    ):
        """Generate a compatible group with one model-forward accounting unit.

        ``mode`` records the layout used by the scheduler. The toy model
        computes from the unpadded prompt in either layout, making parity
        between padded and packed execution explicit.
        """

        self.forward_calls += 1
        self.batch_forward_calls += 1
        return [
            self._raw_tokens(prompt, adapter, temperature, max_new_tokens)
            for prompt in prompts
        ]

    def prefill(self, tokens, adapter, temperature):
        """Record a prefix prefill and return an immutable toy KV state."""

        self.prefill_calls += 1
        return (tuple(tokens), str(adapter), float(temperature))


class KVCache:
    """A copy-on-read/write cache suitable for token lists and prefix state."""

    def __init__(self):
        self._entries = {}

    def get(self, prompt):
        value = self._entries.get(_freeze(prompt))
        if value is None:
            return None
        if isinstance(value, list):
            return list(value)
        return value

    def put(self, prompt, value):
        if isinstance(value, list):
            value = list(value)
        self._entries[_freeze(prompt)] = value

    def clear(self):
        self._entries.clear()

    def contains(self, prompt):
        return _freeze(prompt) in self._entries

    def __len__(self):
        return len(self._entries)


class BatchEngine:
    """Batch compatible requests while keeping deterministic request order."""

    def __init__(self, models=None, mode="padded", pad_token_id=0):
        self.models = {} if models is None else models
        if mode not in {"padded", "packed"}:
            raise ValueError("mode must be 'padded' or 'packed'")
        self.mode = mode
        self.pad_token_id = pad_token_id
        self.cache = KVCache()
        self.prefix_cache = KVCache()
        self._generation = 0
        self.stats = {
            "batch_calls": 0,
            "forward_calls": 0,
            "prefix_hits": 0,
            "padded_tokens": 0,
            "packed_tokens": 0,
        }
        self.last_batch_mode = mode

    def register(self, model):
        self.models[model.model_id] = model
        # A model object can retain different behavior under the same id.
        self._generation += 1
        self.cache.clear()
        self.prefix_cache.clear()

    def clear(self):
        self._generation += 1
        self.cache.clear()
        self.prefix_cache.clear()

    @staticmethod
    def _validate_request(request):
        if not _is_int(request.max_new_tokens):
            raise ValueError("max_new_tokens must be an integer")
        if not _is_int(request.min_new_tokens):
            raise ValueError("min_new_tokens must be an integer")
        if request.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if request.min_new_tokens < 0:
            raise ValueError("min_new_tokens must be non-negative")
        if request.min_new_tokens > request.max_new_tokens:
            raise ValueError("min_new_tokens cannot exceed max_new_tokens")
        if request.eos_token_id is not None and not _is_int(request.eos_token_id):
            raise ValueError("eos_token_id must be an integer or None")
        if (
            not isinstance(request.temperature, Real)
            or isinstance(request.temperature, bool)
            or not math.isfinite(float(request.temperature))
        ):
            raise ValueError("temperature must be finite")
        try:
            prompt = tuple(request.prompt)
        except TypeError:
            raise ValueError("prompt must be a sequence of token ids") from None
        if not all(_is_prompt_token(token) for token in prompt):
            raise ValueError("prompt must contain only integer token ids")

    def _context_key(self, request):
        model = self.models.get(request.model)
        return (
            self._generation,
            _freeze(request.tenant),
            _freeze(request.model),
            id(model),
            _freeze(request.adapter),
            _freeze(request.temperature),
        )

    def _result_key(self, request):
        return (self._context_key(request), _freeze(tuple(request.prompt)))

    def _prefix_key(self, context, prefix):
        return (context, _freeze(tuple(prefix)))

    def _ensure_prefix(self, request, model):
        """Populate prefix state and count reuse of a previously seen prefix."""

        prompt = tuple(request.prompt)
        if not prompt:
            return
        context = self._context_key(request)
        longest = 0
        for length in range(len(prompt), 0, -1):
            if self.prefix_cache.contains(self._prefix_key(context, prompt[:length])):
                longest = length
                break
        if longest:
            self.stats["prefix_hits"] += 1
        if longest == len(prompt):
            return
        extension = prompt[longest:]
        if hasattr(model, "prefill"):
            state = model.prefill(extension, request.adapter, request.temperature)
        else:
            state = tuple(extension)
        for length in range(longest + 1, len(prompt) + 1):
            self.prefix_cache.put(self._prefix_key(context, prompt[:length]), state)

    @staticmethod
    def _apply_stop(tokens, request):
        """Apply max/min/EOS policy to a raw deterministic token sequence."""

        result = list(tokens[: request.max_new_tokens])
        eos = request.eos_token_id
        if eos is None:
            return result
        minimum = min(request.min_new_tokens, request.max_new_tokens)
        for index, token in enumerate(result):
            if token == eos and index + 1 >= minimum:
                return result[: index + 1]
        return result

    def _cached_raw(self, request, required):
        cached = self.cache.get(self._result_key(request))
        if cached is None:
            return None
        # A cached EOS is sufficient even when the original request asked for
        # fewer tokens. Otherwise a short cache entry must be extended.
        if len(cached) >= required:
            return cached
        if request.eos_token_id is not None:
            minimum = min(request.min_new_tokens, request.max_new_tokens)
            for index, token in enumerate(cached):
                if token == request.eos_token_id and index + 1 >= minimum:
                    return cached
        return None

    def _run_model_batch(self, model, requests, mode):
        if not requests:
            return []
        max_tokens = max(request.max_new_tokens for request in requests)
        prompts = [tuple(request.prompt) for request in requests]
        if mode == "padded":
            width = max((len(prompt) for prompt in prompts), default=0)
            self.stats["padded_tokens"] += sum(width - len(prompt) for prompt in prompts)
        else:
            self.stats["packed_tokens"] += sum(len(prompt) for prompt in prompts)
        batch_method = getattr(model, "next_tokens_batch", None)
        if batch_method is not None:
            self.stats["forward_calls"] += 1
            try:
                return batch_method(
                    prompts,
                    requests[0].adapter,
                    requests[0].temperature,
                    max_tokens,
                    mode=mode,
                )
            except TypeError:
                # Permit simple test doubles exposing a four-argument batch
                # method while retaining the mode-aware toy implementation.
                return batch_method(
                    prompts,
                    requests[0].adapter,
                    requests[0].temperature,
                    max_tokens,
                )
        self.stats["forward_calls"] += len(requests)
        return [
            model.next_tokens(
                request.prompt,
                request.adapter,
                request.temperature,
                max_tokens,
            )
            for request in requests
        ]

    def generate(self, request):
        return self.generate_batch([request], mode=self.mode)[0]

    def generate_batch(self, requests, mode=None):
        requests = list(requests)
        mode = self.mode if mode is None else mode
        if mode not in {"padded", "packed"}:
            raise ValueError("mode must be 'padded' or 'packed'")
        self.last_batch_mode = mode
        self.stats["batch_calls"] += 1
        if not requests:
            return []
        for request in requests:
            self._validate_request(request)

        # Group by context so one compatible model forward handles all prompt
        # rows. Sorting only chooses a deterministic execution order. Results
        # are scattered back to their original positions below.
        groups = {}
        for index, request in enumerate(requests):
            key = self._context_key(request)
            groups.setdefault(key, []).append((index, request))

        outputs = [None] * len(requests)
        for context_key in sorted(groups, key=repr):
            rows = groups[context_key]
            first_request = rows[0][1]
            model = self.models[first_request.model]
            pending_by_prompt = {}
            for index, request in rows:
                if request.max_new_tokens == 0:
                    outputs[index] = []
                    continue
                required = request.max_new_tokens
                cached = self._cached_raw(request, required)
                if cached is not None:
                    outputs[index] = self._apply_stop(cached, request)
                    continue
                prompt_key = _freeze(tuple(request.prompt))
                current = pending_by_prompt.get(prompt_key)
                if current is None or request.max_new_tokens > current[1]:
                    pending_by_prompt[prompt_key] = (request, request.max_new_tokens)

            pending = list(pending_by_prompt.values())
            pending.sort(key=lambda pair: repr(_freeze(tuple(pair[0].prompt))))
            if pending:
                for request, _ in pending:
                    self._ensure_prefix(request, model)
                raw_values = self._run_model_batch(
                    model, [pair[0] for pair in pending], mode
                )
                for (request, _), raw in zip(pending, raw_values):
                    raw = list(raw)
                    self.cache.put(self._result_key(request), raw)
                    prompt_key = _freeze(tuple(request.prompt))
                    for index, row_request in rows:
                        if _freeze(tuple(row_request.prompt)) == prompt_key:
                            outputs[index] = self._apply_stop(raw, row_request)
        return outputs
