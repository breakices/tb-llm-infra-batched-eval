"""Verifier-owned scalar formula and an instrumented generation backend."""

from llm_runtime import Request


def request(**fields):
    values = {
        "tenant": "t1",
        "model": "m1",
        "adapter": "base",
        "prompt": (1, 2, 3),
        "max_new_tokens": 4,
        "temperature": 0.0,
    }
    values.update(fields)
    return Request(**values)


def token_value(token):
    if isinstance(token, int):
        return int(token)
    if isinstance(token, str):
        units = map(ord, token)
    else:
        units = bytes(token)
    return sum(index * unit for index, unit in enumerate(units, 1))


def oracle_tokens(model_id, adapter, prompt, temperature, max_new_tokens,
                  min_new_tokens=0, eos_token_id=None):
    """Compute raw ToyModel tokens, stopping at the first eligible EOS."""
    salt = sum(map(ord, str(model_id) + str(adapter))) % 97
    seed = (sum(map(token_value, prompt)) + salt + int(temperature * 1000)) % 1000
    tokens = []
    for step in range(max_new_tokens):
        token = (seed + 17 * step) % 101
        tokens.append(token)
        if token == eos_token_id and len(tokens) >= min_new_tokens:
            break
    return tuple(tokens)


def expected(row):
    prompt = row.prompt
    limit = getattr(row, "context_length", None)
    if limit is not None:
        prompt = prompt[-limit:] if limit else ()
    tokens = oracle_tokens(
        row.model, row.adapter, prompt, row.temperature, row.max_new_tokens,
        getattr(row, "min_new_tokens", 0), getattr(row, "eos_token_id", None),
    )
    return list(tokens)


class RecordingModel:
    """Record actual backend inputs so assertions don't rely on engine counters."""

    def __init__(self, model_id="m1", revision=0):
        self.model_id = model_id
        self.revision = revision
        self.calls = []
        self.prefilled = []

    def _tokens(self, prompt, adapter, temperature, limit):
        raw = oracle_tokens(self.model_id, adapter, prompt, temperature, limit)
        return [(token + self.revision) % 101 for token in raw]

    def next_tokens(self, prompt, adapter, temperature, max_new_tokens):
        self.calls.append(("scalar", [tuple(prompt)]))
        return self._tokens(prompt, adapter, temperature, max_new_tokens)

    def next_tokens_batch(self, prompts, adapter, temperature, max_new_tokens, mode="padded"):
        prompts = [tuple(prompt) for prompt in prompts]
        self.calls.append((mode, prompts))
        return [self._tokens(prompt, adapter, temperature, max_new_tokens) for prompt in prompts]

    def prefill(self, tokens, adapter, temperature):
        self.prefilled.extend(tokens)
        return tuple(tokens)
