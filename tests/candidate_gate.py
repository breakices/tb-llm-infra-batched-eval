"""Candidate-facing parity and work checks for a complete 192-request shard."""

from collections import Counter
from dataclasses import dataclass, field, replace
from itertools import product
from random import Random

from llm_runtime import BatchEngine, Request, ToyModel
from tiny_oracle import oracle_tokens


SEED = 20261118
MODEL_IDS = ("primary", "影")


@dataclass(frozen=True)
class Row:
    tenant: str | int
    model: str
    adapter: str
    prompt: tuple
    temperature: float
    max_new_tokens: int
    min_new_tokens: int
    request_id: str
    weight: float
    eos_token_id: int | None = None


def backend_input(prompt):
    """Give the probe model order-sensitive tokenization, including bool identity."""
    return tuple(
        (index + 1) * (31 if value is True else 47 if value is False else value)
        for index, value in enumerate(prompt)
    )


def expected(request):
    """Scalar uncached oracle, independent of candidate dispatch and cache code."""
    return list(oracle_tokens(
        "weights/" + request.model, request.adapter, backend_input(request.prompt),
        request.temperature, request.max_new_tokens, request.min_new_tokens,
        request.eos_token_id,
    ))


def corpus():
    """Cross every backend configuration with every prompt and tenant before shuffling."""
    rows = []
    for tenant, model, adapter, temperature in product(
        ("租户", True, 1), MODEL_IDS, ("base", "alt"), (0.0, 0.125)
    ):
        local = []
        prompts = ((), (11, True), (11, 1), (17, 11), (11, 17), (11, 17, -3),
                   (11, 17, 9), (11, 17, -3))
        for i, prompt in enumerate(prompts):
            length = 0 if i == 6 else 3 + i % 6
            req = Row(
                tenant=tenant, model=model, adapter=adapter, prompt=prompt,
                max_new_tokens=length, temperature=temperature,
                min_new_tokens=1 if length else 0, request_id=f"id-{i % 3}",
                weight=(i % 3) + 0.5,
            )
            if i in (3, 4):
                # Select a legal EOS from the raw answer; test both eligible
                # and premature EOS. These rows do not depend on cache state.
                choices = [(j, token) for j, token in enumerate(expected(req)) if token < 31]
                if choices:
                    j, eos = choices[0]
                    minimum = j + 1 if i == 3 else min(length, j + 2)
                    req = replace(req, eos_token_id=eos, min_new_tokens=minimum)
            if i == 7:
                req = replace(local[5], request_id="duplicate", weight=0)
            local.append(req)
        rows.extend(local)
    Random(SEED).shuffle(rows)
    return rows


@dataclass
class Work:
    forwards: int = 0
    generated: Counter = field(default_factory=Counter)
    prefilled: Counter = field(default_factory=Counter)


def typed_prompt(prompt):
    return tuple((type(token).__name__, token) for token in prompt)


class ObservedBackend:
    """Count actual calls at the injected model boundary, outside BatchEngine."""

    def __init__(self, model_id, work):
        self.model_id = model_id
        self.work = work
        # A registered implementation is allowed to differ from the default
        # model with the same lookup id; the engine must call this backend.
        self._weights = ToyModel("weights/" + model_id)

    def next_tokens(self, prompt, adapter, temperature, max_new_tokens):
        self.work.forwards += 1
        self.work.generated[self.model_id, adapter, temperature, typed_prompt(prompt)] += 1
        return self._weights.next_tokens(
            backend_input(prompt), adapter, temperature, max_new_tokens
        )

    def next_tokens_batch(self, prompts, adapter, temperature, max_new_tokens, mode="padded"):
        prompts = list(prompts)
        self.work.forwards += 1
        for prompt in prompts:
            self.work.generated[self.model_id, adapter, temperature, typed_prompt(prompt)] += 1
        return self._weights.next_tokens_batch(
            [backend_input(prompt) for prompt in prompts], adapter, temperature,
            max_new_tokens, mode=mode,
        )

    def prefill(self, tokens, adapter, temperature):
        tokens = tuple(tokens)
        self.work.prefilled[self.model_id, adapter, temperature] += len(tokens)
        return self._weights.prefill(tokens, adapter, temperature)


def observed_engine():
    work = Work()
    models = {key: ObservedBackend(key, work) for key in MODEL_IDS}
    return BatchEngine(models=models), work


def generate(engine, rows, mode):
    return engine.generate_batch([Request(**vars(r)) for r in rows], mode=mode)


def assert_row(wanted, actual, mode, row):
    where = f"seed={SEED} mode={mode} row={row:03d}"
    assert not isinstance(actual, Exception), f"{where}: batch raised {actual!r}"
    assert isinstance(actual, list), f"{where}: expected a list of output rows"
    assert row < len(actual), f"{where}: missing row (received {len(actual)})"
    if row == len(wanted) - 1:
        assert len(actual) == len(wanted), f"{where}: unexpected extra output rows"
    assert actual[row] == wanted[row], f"{where}: expected {wanted[row]}, got {actual[row]}"


def assert_parity(engine, rows, mode):
    wanted = [expected(row) for row in rows]
    actual = generate(engine, rows, mode)
    for i in range(len(wanted)):
        assert_row(wanted, actual, mode, i)


def work_budget(requests):
    # Each typed prompt has one generation policy per scope in this shard.
    # Identical backend inputs from distinct tenants must both do work, even
    # when their deterministic outputs happen to coincide.
    prompts_by_scope = {}
    for row in requests:
        if row.max_new_tokens:
            scope = ((type(row.tenant).__name__, row.tenant), row.model, row.adapter, row.temperature)
            prompts_by_scope.setdefault(scope, set()).add(typed_prompt(row.prompt))
    max_forwards = 0
    required_rows = Counter()
    minimum_prefill = Counter()
    for (_, model, adapter, temperature), prompts in prompts_by_scope.items():
        backend_context = (model, adapter, temperature)
        max_forwards += (len(prompts) + 7) // 8
        required_rows.update((*backend_context, prompt) for prompt in prompts)
        # Even an ideal per-tenant prefix cache must ingest each distinct
        # prefix edge once. This is a lower bound, independent of iteration order.
        edges = {prompt[:n] for prompt in prompts for n in range(1, len(prompt) + 1)}
        minimum_prefill[backend_context] += len(edges)
    return max_forwards, required_rows, minimum_prefill
