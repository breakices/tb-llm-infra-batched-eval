"""Small deterministic LLM serving runtime."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    tenant: str
    model: str
    adapter: str
    prompt: tuple[int, ...]
    max_new_tokens: int = 4
    temperature: float = 0.0


class ToyModel:
    def __init__(self, model_id):
        self.model_id = model_id
        self.forward_calls = 0

    def next_tokens(self, prompt, adapter, temperature, max_new_tokens):
        self.forward_calls += 1
        salt = sum(ord(c) for c in (self.model_id + adapter)) % 97
        seed = (sum(prompt) + salt + int(temperature * 1000)) % 1000
        return [((seed + i * 17) % 101) for i in range(max_new_tokens)]


class KVCache:
    def __init__(self):
        self._entries = {}

    def get(self, prompt):
        return self._entries.get(tuple(prompt))

    def put(self, prompt, value):
        self._entries[tuple(prompt)] = list(value)

    def clear(self):
        self._entries.clear()


class BatchEngine:
    def __init__(self, models=None):
        self.models = models or {}
        self.cache = KVCache()

    def register(self, model):
        self.models[model.model_id] = model

    def clear(self):
        self.cache.clear()

    def generate(self, request):
        cached = self.cache.get(request.prompt)
        if cached is not None:
            return list(cached[: request.max_new_tokens])
        model = self.models[request.model]
        result = model.next_tokens(
            request.prompt, request.adapter, request.temperature, request.max_new_tokens
        )
        self.cache.put(request.prompt, result)
        return list(result)

    def generate_batch(self, requests):
        return [self.generate(request) for request in requests]
