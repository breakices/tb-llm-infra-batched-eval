"""Private parity tests for the hard batched-evaluation candidate.

These tests intentionally exercise the public runtime through a tiny local
oracle.  They do not use network services, credentials, or implementation
internals.  The 36 tests are kept in one file so ``score.py`` can count and
report the complete private suite.
"""

import inspect
import math

import pytest

from llm_runtime import BatchEngine, Request, ToyModel

from tiny_oracle import EvalCase, TinyEvaluator, oracle_tokens


def make_request(
    prompt=(1, 2, 3),
    model="m1",
    adapter="base",
    tenant="tenant-a",
    max_new_tokens=6,
    temperature=0.0,
    request_id="case-1",
    min_new_tokens=0,
    eos_token_id=None,
):
    return Request(
        tenant=tenant,
        model=model,
        adapter=adapter,
        prompt=tuple(prompt),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        request_id=request_id,
        min_new_tokens=min_new_tokens,
        eos_token_id=eos_token_id,
    )


def make_engine():
    return BatchEngine(models={"m1": ToyModel("m1"), "m2": ToyModel("m2")})


def run_mode(engine, requests, mode):
    signature = inspect.signature(engine.generate_batch)
    if "mode" not in signature.parameters:
        pytest.fail("generate_batch must expose the padded/packed mode")
    return engine.generate_batch(requests, mode=mode)


def expected(request):
    return list(
        oracle_tokens(
            request.model,
            request.adapter,
            request.prompt,
            request.temperature,
            request.max_new_tokens,
            request.min_new_tokens,
            request.eos_token_id,
        )
    )


def test_oracle_matches_toy_model_formula():
    model = ToyModel("m1")
    request = make_request()
    assert model.next_tokens(request.prompt, request.adapter, request.temperature, request.max_new_tokens) == expected(request)


def test_single_packed_request_matches_oracle():
    engine = make_engine()
    request = make_request()
    assert run_mode(engine, [request], "packed") == [expected(request)]


def test_single_padded_request_matches_oracle():
    engine = make_engine()
    request = make_request()
    assert run_mode(engine, [request], "padded") == [expected(request)]


def test_padded_and_packed_outputs_match():
    requests = [make_request(prompt=(i, i + 1), request_id="r" + str(i)) for i in range(4)]
    padded = run_mode(make_engine(), requests, "padded")
    packed = run_mode(make_engine(), requests, "packed")
    assert padded == packed


def test_input_order_is_preserved_in_padded_mode():
    requests = [make_request(prompt=(i,), request_id="r" + str(i)) for i in (5, 1, 3)]
    outputs = run_mode(make_engine(), requests, "padded")
    assert outputs == [expected(request) for request in requests]


def test_input_order_is_preserved_in_packed_mode():
    requests = [make_request(prompt=(i,), request_id="r" + str(i)) for i in (5, 1, 3)]
    outputs = run_mode(make_engine(), requests, "packed")
    assert outputs == [expected(request) for request in requests]


def test_reordering_requests_only_reorders_outputs():
    requests = [make_request(prompt=(i,), request_id="r" + str(i)) for i in (2, 4, 6)]
    engine = make_engine()
    first = run_mode(engine, requests, "packed")
    reordered = run_mode(engine, list(reversed(requests)), "packed")
    assert reordered == list(reversed(first))


def test_duplicate_request_ids_keep_duplicate_rows():
    requests = [make_request(prompt=(2,), request_id="same"), make_request(prompt=(7,), request_id="same")]
    outputs = run_mode(make_engine(), requests, "packed")
    assert len(outputs) == 2
    assert outputs[0] == expected(requests[0])
    assert outputs[1] == expected(requests[1])


def test_duplicate_identical_rows_keep_duplicate_outputs():
    request = make_request(request_id="same")
    outputs = run_mode(make_engine(), [request, request], "padded")
    assert outputs == [expected(request), expected(request)]


def test_empty_batch_returns_empty_list_in_padded_mode():
    assert run_mode(make_engine(), [], "padded") == []


def test_empty_batch_returns_empty_list_in_packed_mode():
    assert run_mode(make_engine(), [], "packed") == []


def test_shared_prefix_does_not_change_tokens():
    requests = [
        make_request(prompt=(9, 8, 7, 1), request_id="left"),
        make_request(prompt=(9, 8, 7, 2), request_id="right"),
    ]
    assert run_mode(make_engine(), requests, "packed") == [expected(request) for request in requests]


def test_shared_prefix_is_reused_for_repeated_batch():
    model = ToyModel("m1")
    engine = BatchEngine(models={"m1": model})
    requests = [make_request(prompt=(9, 8, 7, 1), request_id="left"), make_request(prompt=(9, 8, 7, 2), request_id="right")]
    run_mode(engine, requests, "packed")
    calls_after_first = model.forward_calls
    run_mode(engine, requests, "packed")
    assert model.forward_calls == calls_after_first


def test_shared_prefix_cache_does_not_alias_different_tenants():
    model = ToyModel("m1")
    engine = BatchEngine(models={"m1": model})
    requests = [make_request(tenant="a", request_id="a"), make_request(tenant="b", request_id="b")]
    run_mode(engine, requests, "packed")
    assert model.forward_calls >= 2


def test_cache_isolated_by_model():
    engine = make_engine()
    requests = [make_request(model="m1", request_id="a"), make_request(model="m2", request_id="b")]
    outputs = run_mode(engine, requests, "packed")
    assert outputs == [expected(request) for request in requests]
    assert outputs[0] != outputs[1]


def test_cache_isolated_by_adapter():
    engine = make_engine()
    requests = [make_request(adapter="base", request_id="a"), make_request(adapter="alt", request_id="b")]
    assert run_mode(engine, requests, "packed") == [expected(request) for request in requests]


def test_cache_isolated_by_tenant():
    engine = make_engine()
    requests = [make_request(tenant="a", request_id="a"), make_request(tenant="b", request_id="b")]
    assert run_mode(engine, requests, "packed") == [expected(request) for request in requests]


def test_cache_isolated_by_temperature():
    engine = make_engine()
    requests = [make_request(temperature=0.0, request_id="a"), make_request(temperature=0.25, request_id="b")]
    assert run_mode(engine, requests, "packed") == [expected(request) for request in requests]


def test_cache_isolated_by_max_new_tokens():
    engine = make_engine()
    requests = [make_request(max_new_tokens=2, request_id="a"), make_request(max_new_tokens=5, request_id="b")]
    assert run_mode(engine, requests, "packed") == [expected(request) for request in requests]


def test_clear_forces_recomputation():
    model = ToyModel("m1")
    engine = BatchEngine(models={"m1": model})
    request = make_request()
    run_mode(engine, [request], "packed")
    engine.clear()
    run_mode(engine, [request], "packed")
    assert model.forward_calls == 2


def test_register_replacement_forces_recomputation():
    old = ToyModel("m1")
    engine = BatchEngine(models={"m1": old})
    request = make_request()
    run_mode(engine, [request], "packed")
    replacement = ToyModel("m1")
    engine.register(replacement)
    run_mode(engine, [request], "packed")
    assert replacement.forward_calls == 1


def test_returned_tokens_are_detached_in_padded_mode():
    engine = make_engine()
    request = make_request()
    first = run_mode(engine, [request], "padded")[0]
    first[0] = -123
    assert run_mode(engine, [request], "padded")[0] == expected(request)


def test_returned_tokens_are_detached_in_packed_mode():
    engine = make_engine()
    request = make_request()
    first = run_mode(engine, [request], "packed")[0]
    first[0] = -123
    assert run_mode(engine, [request], "packed")[0] == expected(request)


def test_longer_request_does_not_truncate_shorter_duplicate():
    engine = make_engine()
    short = make_request(max_new_tokens=2, request_id="short")
    long = make_request(max_new_tokens=6, request_id="long")
    outputs = run_mode(engine, [short, long], "packed")
    assert outputs == [expected(short), expected(long)]


def test_zero_max_new_tokens_is_supported():
    request = make_request(max_new_tokens=0)
    assert run_mode(make_engine(), [request], "packed") == [[]]


def test_min_new_tokens_is_never_exceeded_by_eos():
    request = make_request(max_new_tokens=6, min_new_tokens=3, eos_token_id=0)
    output = run_mode(make_engine(), [request], "packed")[0]
    assert len(output) >= 3
    if 0 in output:
        assert output.index(0) + 1 >= 3


def test_eos_stops_after_first_allowed_eos():
    request = make_request(max_new_tokens=9, min_new_tokens=0, eos_token_id=0)
    output = run_mode(make_engine(), [request], "padded")[0]
    if 0 in output:
        assert output[-1] == 0


def test_eos_never_emits_more_than_max_new_tokens():
    request = make_request(max_new_tokens=2, min_new_tokens=1, eos_token_id=0)
    output = run_mode(make_engine(), [request], "packed")[0]
    assert len(output) <= 2


def test_min_new_tokens_equal_max_is_valid():
    request = make_request(max_new_tokens=4, min_new_tokens=4, eos_token_id=0)
    output = run_mode(make_engine(), [request], "padded")[0]
    assert len(output) == 4


def test_impossible_minimum_is_rejected():
    request = make_request(max_new_tokens=2, min_new_tokens=3)
    with pytest.raises(ValueError):
        run_mode(make_engine(), [request], "packed")


def test_invalid_mode_is_rejected():
    with pytest.raises((ValueError, KeyError)):
        run_mode(make_engine(), [make_request()], "ragged")


def test_tiny_evaluator_scores_all_correct_rows():
    requests = [make_request(prompt=(i,), request_id="r" + str(i)) for i in range(3)]
    cases = [EvalCase(request.request_id, request, tuple(expected(request)), 1.0) for request in requests]
    assert TinyEvaluator().score(make_engine(), cases, mode="packed") == 1.0


def test_tiny_evaluator_uses_weighted_metrics():
    requests = [make_request(prompt=(1,), request_id="low"), make_request(prompt=(2,), request_id="high")]
    cases = [
        EvalCase("low", requests[0], (999,), 1.0),
        EvalCase("high", requests[1], tuple(expected(requests[1])), 9.0),
    ]
    assert TinyEvaluator().score(make_engine(), cases, mode="packed") == pytest.approx(0.9)


def test_tiny_evaluator_keeps_duplicate_ids_out_of_metric_map():
    requests = [make_request(prompt=(1,), request_id="same"), make_request(prompt=(2,), request_id="same")]
    cases = [
        EvalCase("same-0", requests[0], tuple(expected(requests[0])), 1.0),
        EvalCase("same-1", requests[1], tuple(expected(requests[1])), 1.0),
    ]
    predictions = TinyEvaluator().run(make_engine(), cases, mode="packed")
    assert set(predictions) == {"same-0", "same-1"}


def test_tiny_evaluator_rejects_negative_weight():
    request = make_request()
    cases = [EvalCase("negative", request, tuple(expected(request)), -1.0)]
    with pytest.raises(ValueError):
        TinyEvaluator().score(make_engine(), cases)


def test_tiny_evaluator_zero_total_weight_is_zero():
    request = make_request()
    cases = [EvalCase("zero", request, tuple(expected(request)), 0.0)]
    assert TinyEvaluator().score(make_engine(), cases) == 0.0


def test_weighted_metric_does_not_depend_on_input_order():
    first = make_request(prompt=(1,), request_id="first")
    second = make_request(prompt=(2,), request_id="second")
    cases = [EvalCase("first", first, tuple(expected(first)), 2.0), EvalCase("second", second, (999,), 1.0)]
    evaluator = TinyEvaluator()
    score_a = evaluator.score(make_engine(), cases, mode="packed")
    score_b = evaluator.score(make_engine(), list(reversed(cases)), mode="packed")
    assert score_a == score_b == pytest.approx(2.0 / 3.0)


def test_batch_results_have_independent_rows():
    requests = [make_request(prompt=(1,), request_id="a"), make_request(prompt=(2,), request_id="b")]
    outputs = run_mode(make_engine(), requests, "packed")
    outputs[0].append(1000)
    assert outputs[1] == expected(requests[1])


def test_model_call_count_is_bounded_for_duplicate_rows():
    model = ToyModel("m1")
    engine = BatchEngine(models={"m1": model})
    request = make_request()
    run_mode(engine, [request, request, request], "packed")
    assert model.forward_calls <= 1


def test_model_call_count_is_bounded_for_repeated_prefix_rows():
    model = ToyModel("m1")
    engine = BatchEngine(models={"m1": model})
    requests = [make_request(prompt=(1, 2, 3, i), request_id=str(i)) for i in range(3)]
    run_mode(engine, requests, "packed")
    assert model.forward_calls <= len(requests)


def test_model_calls_do_not_depend_on_request_order():
    requests = [make_request(prompt=(i,), request_id=str(i)) for i in range(4)]
    first_model = ToyModel("m1")
    first_engine = BatchEngine(models={"m1": first_model})
    run_mode(first_engine, requests, "packed")
    second_model = ToyModel("m1")
    second_engine = BatchEngine(models={"m1": second_model})
    run_mode(second_engine, list(reversed(requests)), "packed")
    assert first_model.forward_calls == second_model.forward_calls


def test_request_prompt_is_not_mutated():
    request = make_request(prompt=(4, 5, 6))
    before = request.prompt
    run_mode(make_engine(), [request], "padded")
    assert request.prompt == before


def test_cache_handles_bool_and_int_prompt_values_without_crashing():
    model = ToyModel("m1")
    engine = BatchEngine(models={"m1": model})
    requests = [make_request(prompt=(1,), request_id="int"), make_request(prompt=(True,), request_id="bool")]
    outputs = run_mode(engine, requests, "packed")
    assert len(outputs) == 2
    assert outputs[0] == outputs[1]
    # The rows share one compatible batch forward, while their typed cache
    # keys remain separate entries.
    assert model.forward_calls == 1
    assert len(engine.cache) == 2


def test_unknown_model_raises_key_error():
    with pytest.raises(KeyError):
        run_mode(make_engine(), [make_request(model="missing")], "packed")


# v3 expansion: broad matrix of deterministic parity and boundary contracts.
# Parameterized rows count as individual private tests in the scorer.

V3_PARITY_CASES = [
    ((1,), "m1", "base", "tenant-a", 0, 0.0, 0, None),
    ((), "m1", "base", "tenant-a", 1, 0.0, 0, None),
    ((0,), "m1", "base", "tenant-a", 2, 0.0, 0, None),
    ((-1, 2), "m1", "base", "tenant-a", 3, 0.0, 0, None),
    ((1, 2, 3), "m1", "base", "tenant-a", 4, 0.0, 0, None),
    ((9, 8, 7, 6, 5), "m1", "base", "tenant-a", 6, 0.0, 0, None),
    ((True,), "m1", "base", "tenant-a", 3, 0.0, 0, None),
    ((False, 1), "m1", "base", "tenant-a", 3, 0.0, 0, None),
    ((1, True, 2), "m1", "base", "tenant-a", 5, 0.0, 0, None),
    ((1, 2), "m1", "base", "租户/一", 3, 0.0, 0, None),
    ((3, 4), "m1", "适配器-β", "tenant-a", 3, 0.125, 0, None),
    ((5,), "m1", "base", "tenant-a🚀", 3, 0.25, 0, None),
    ((2, 4, 6), "m2", "alt", "tenant-b", 7, 0.5, 2, None),
    ((2, 4, 6), "m2", "alt", "tenant-b", 7, 1.25, 3, 0),
    ((2, 4, 6), "m2", "alt", "tenant-b", 7, 2.0, 7, 0),
    ((10, 11), "m1", "base", "tenant-a", 8, 0.01, 1, 30),
    ((10, 11), "m1", "base", "tenant-a", 8, 0.99, 4, 30),
    ((31, 32, 33), "m1", "base", "tenant-a", 10, 0.0, 0, 1),
    ((31, 32, 33), "m1", "base", "tenant-a", 10, 0.0, 0, 30),
    ((100, 101, 102), "m1", "base", "tenant-a", 12, 0.333, 5, None),
    ((4, 5, 6, 7), "m2", "base", "tenant-c", 2, 0.667, 0, None),
    ((4, 5, 6, 7), "m2", "base", "tenant-c", 2, 1.5, 1, 0),
    ((8, 9), "m1", "adapter/δ", "租户🚀", 5, 0.0, 0, None),
    ((8, 9), "m1", "adapter/δ", "租户🚀", 5, 0.0, 5, 30),
    ((0, 0, 0), "m1", "base", "tenant-a", 20, 0.0, 0, None),
    ((-100, 100), "m1", "base", "tenant-a", 4, 0.0, 0, None),
    ((1, 2, 3), "m1", "base", "tenant-a", 0, 0.0, 0, None),
    ((1, 2, 3), "m1", "base", "tenant-a", 1, 0.0, 1, None),
    ((1, 2, 3), "m1", "base", "tenant-a", 1, 0.0, 1, 0),
    ((1, 2, 3), "m1", "base", "tenant-a", 9, 3.14159, 2, 30),
]


@pytest.mark.parametrize(
    "prompt,model,adapter,tenant,max_new,temperature,min_new,eos",
    V3_PARITY_CASES,
    ids=[f"case-{i}" for i in range(len(V3_PARITY_CASES))],
)
def test_v3_padded_packed_parity_matrix(
    prompt, model, adapter, tenant, max_new, temperature, min_new, eos
):
    request = make_request(
        prompt=prompt,
        model=model,
        adapter=adapter,
        tenant=tenant,
        max_new_tokens=max_new,
        temperature=temperature,
        min_new_tokens=min_new,
        eos_token_id=eos,
        request_id=f"v3-{len(prompt)}-{max_new}",
    )
    engine = make_engine()
    padded = run_mode(engine, [request], "padded")
    packed = run_mode(make_engine(), [request], "packed")
    assert padded == packed == [expected(request)]


V3_BATCH_CASES = [
    [(1,), (2, 3)],
    [(), (1,), (2, 3, 4)],
    [(True,), (1,), (False, 1)],
    [(9, 8, 7, 1), (9, 8, 7, 2), (9, 8, 6)],
    [(1, 2, 3), (1, 2, 3), (1, 2)],
    [(5,), (4, 3), (2, 1, 0, -1)],
    [(0,), (0,)],
    [(-10, 20), (30,), ()],
    [(100, 101), (101, 100), (100, 101)],
    [(1, True), (1, 1), (True, 1)],
    [(7, 8, 9, 10, 11), (7,), (7, 8)],
    [(2, 4, 6), (2, 4, 6, 8), (2, 4, 6, 8, 10)],
]


@pytest.mark.parametrize("prompts", V3_BATCH_CASES, ids=[f"batch-{i}" for i in range(len(V3_BATCH_CASES))])
def test_v3_batch_layout_parity_and_order(prompts):
    requests = [make_request(prompt=p, request_id=f"row-{i}") for i, p in enumerate(prompts)]
    assert run_mode(make_engine(), requests, "padded") == [expected(r) for r in requests]
    assert run_mode(make_engine(), requests, "packed") == [expected(r) for r in requests]


@pytest.mark.parametrize("order", [
    (0, 1, 2), (2, 1, 0), (1, 0, 2), (2, 0, 1),
    (0, 2, 1), (1, 2, 0),
])
def test_v3_scatter_order_permutations(order):
    base = [make_request(prompt=(i, i + 1), request_id=f"id-{i}") for i in range(3)]
    shuffled = [base[i] for i in order]
    output = run_mode(make_engine(), shuffled, "packed")
    assert output == [expected(r) for r in shuffled]


@pytest.mark.parametrize("ids", [
    ("same", "same"), ("x", "x", "x"), ("", ""),
    ("重复", "重复"), ("same", "other", "same"),
    ("0", "0", "1", "0"), ("tenant/id", "tenant/id"),
])
def test_v3_duplicate_ids_are_row_stable(ids):
    requests = [make_request(prompt=(i,), request_id=ids[i]) for i in range(len(ids))]
    outputs = run_mode(make_engine(), requests, "packed")
    assert outputs == [expected(r) for r in requests]


@pytest.mark.parametrize("field,values", [
    ("model", ("m1", "m2")),
    ("adapter", ("base", "alt")),
    ("tenant", ("a", "b")),
    ("temperature", (0.0, 0.25)),
    ("temperature", (0.25, 1.5)),
    ("adapter", ("适配器", "base")),
    ("tenant", ("租户", "default")),
])
def test_v3_context_fields_are_cache_isolated(field, values):
    kwargs_a = {field: values[0]}
    kwargs_b = {field: values[1]}
    first = make_request(prompt=(9, 8, 7), request_id="a", **kwargs_a)
    second = make_request(prompt=(9, 8, 7), request_id="b", **kwargs_b)
    outputs = run_mode(make_engine(), [first, second], "packed")
    assert outputs == [expected(first), expected(second)]


@pytest.mark.parametrize("max_new,min_new,eos", [
    (0, 0, None), (0, 0, 0), (1, 0, 0), (1, 1, 0),
    (2, 0, 0), (2, 1, 0), (2, 2, 0), (3, 0, 1),
    (3, 1, 1), (3, 3, 1), (6, 2, 0), (6, 6, 0),
    (8, 0, 30), (8, 3, 30), (8, 8, 30), (10, 5, None),
])
def test_v3_stop_eos_minimum_matrix(max_new, min_new, eos):
    request = make_request(prompt=(8, 1), max_new_tokens=max_new, min_new_tokens=min_new, eos_token_id=eos)
    output = run_mode(make_engine(), [request], "packed")[0]
    assert output == expected(request)
    assert len(output) <= max_new
    if eos is not None and eos in output:
        assert output.index(eos) + 1 >= min_new


@pytest.mark.parametrize("minimum", list(range(0, 9)))
def test_v3_minimum_boundaries(minimum):
    request = make_request(prompt=(3, 4), max_new_tokens=8, min_new_tokens=minimum, eos_token_id=0)
    output = run_mode(make_engine(), [request], "padded")[0]
    assert output == expected(request)
    assert len(output) >= min(minimum, 8)


@pytest.mark.parametrize("bad", [(-1, 0), (-2, 1), (1, 2), (0, 1), (3, -1), (4, 5)])
def test_v3_invalid_token_limits_rejected(bad):
    max_new, min_new = bad
    with pytest.raises(ValueError):
        run_mode(make_engine(), [make_request(max_new_tokens=max_new, min_new_tokens=min_new)], "packed")


@pytest.mark.parametrize("bad_temp", [float("nan"), float("inf"), float("-inf"), True, "0.1"])
def test_v3_invalid_temperature_rejected(bad_temp):
    with pytest.raises((ValueError, TypeError)):
        run_mode(make_engine(), [make_request(temperature=bad_temp)], "packed")


@pytest.mark.parametrize("bad_eos", [-1, 31, 100, True, 1.5, "0"])
def test_v3_invalid_eos_rejected(bad_eos):
    with pytest.raises((ValueError, TypeError)):
        run_mode(make_engine(), [make_request(eos_token_id=bad_eos)], "packed")


@pytest.mark.parametrize("weight", [0.0, 0.1, 1.0, 1.5, 10.0, 1e-9, 2**10])
def test_v3_weighted_metrics_normalize_float_weights(weight):
    request = make_request(prompt=(1,), request_id=f"w-{weight}")
    cases = [EvalCase(request.request_id, request, tuple(expected(request)), weight)]
    expected_score = 0.0 if weight == 0 else 1.0
    assert TinyEvaluator().score(make_engine(), cases, mode="packed") == pytest.approx(expected_score)


@pytest.mark.parametrize("weights", [(1.0, 1.0), (1.0, 3.0), (0.1, 0.9), (1e-9, 1.0), (2.5, 7.5), (1000.0, 1.0)])
def test_v3_weighted_denominator_is_sum(weights):
    first = make_request(prompt=(1,), request_id="first")
    second = make_request(prompt=(2,), request_id="second")
    cases = [
        EvalCase("first", first, (999,), weights[0]),
        EvalCase("second", second, tuple(expected(second)), weights[1]),
    ]
    expected_score = weights[1] / (weights[0] + weights[1])
    assert TinyEvaluator().score(make_engine(), cases, mode="packed") == pytest.approx(expected_score)


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 8])
def test_v3_duplicate_rows_have_independent_outputs(count):
    request = make_request(prompt=(2, 3), request_id="duplicate")
    outputs = run_mode(make_engine(), [request] * count, "packed")
    assert outputs == [expected(request) for _ in range(count)]


@pytest.mark.parametrize("prompt", [(), (1,), (1, 2), (True,), (1, True), (False, 0), (9, 8, 7, 6)])
def test_v3_prompt_input_is_not_mutated(prompt):
    request = make_request(prompt=prompt)
    before = request.prompt
    run_mode(make_engine(), [request], "padded")
    assert request.prompt == before


@pytest.mark.parametrize("index", list(range(10)))
def test_v3_returned_rows_are_detached(index):
    requests = [make_request(prompt=(index,), request_id=str(index)), make_request(prompt=(index + 1,), request_id=str(index + 1))]
    output = run_mode(make_engine(), requests, "packed")
    original = list(output[1])
    output[0].append(999)
    output[0][:] = [-1]
    assert output[1] == original
    assert run_mode(make_engine(), requests, "packed") == [expected(r) for r in requests]


@pytest.mark.parametrize("mode", ["padded", "packed"])
def test_v3_empty_batch_modes(mode):
    engine = make_engine()
    assert run_mode(engine, [], mode) == []


@pytest.mark.parametrize("mode", ["ragged", "left", "", 1, True])
def test_v3_invalid_layout_modes(mode):
    with pytest.raises((ValueError, KeyError, TypeError)):
        run_mode(make_engine(), [make_request()], mode)


@pytest.mark.parametrize("variant", [
    {"model": "m1"}, {"adapter": "base"}, {"tenant": "a"},
    {"temperature": 0.0}, {"max_new_tokens": 2}, {"min_new_tokens": 0},
    {"eos_token_id": None}, {"prompt": ()}, {"prompt": (True,)},
])
def test_v3_cache_repeat_is_semantically_stable(variant):
    request = make_request(**variant)
    engine = make_engine()
    first = run_mode(engine, [request], "packed")
    second = run_mode(engine, [request], "packed")
    assert first == second == [expected(request)]


@pytest.mark.parametrize("mode", ["padded", "packed"])
def test_v3_cache_clear_removes_reuse(mode):
    model = ToyModel("m1")
    engine = BatchEngine(models={"m1": model})
    request = make_request()
    run_mode(engine, [request], mode)
    calls = model.forward_calls
    engine.clear()
    run_mode(engine, [request], mode)
    assert model.forward_calls > calls


@pytest.mark.parametrize("prompt", [(), (1,), (1, 2), (1, 2, 3), (9, 8, 7, 6)])
def test_v3_prefix_reuse_does_not_change_results(prompt):
    left = make_request(prompt=prompt + (1,), request_id="left")
    right = make_request(prompt=prompt + (2,), request_id="right")
    outputs = run_mode(make_engine(), [left, right], "packed")
    assert outputs == [expected(left), expected(right)]


@pytest.mark.parametrize("mode", ["padded", "packed"])
def test_v3_stats_mode_and_token_accounting(mode):
    requests = [make_request(prompt=(1,), request_id="a"), make_request(prompt=(1, 2, 3), request_id="b")]
    engine = make_engine()
    run_mode(engine, requests, mode)
    if mode == "packed":
        assert engine.stats.get("packed_tokens", 0) == 4
    else:
        assert engine.stats.get("padded_tokens", 0) >= 4


@pytest.mark.parametrize("token", [True, False, 0, 1, -1, 31, 100])
def test_v3_bool_int_prompt_tokens_are_accepted(token):
    request = make_request(prompt=(token,))
    outputs = run_mode(make_engine(), [request], "packed")
    assert outputs == [expected(request)]

# Evaluator v3 contract tests.  These exercise the independent tiny oracle and
# keep evaluator semantics distinct from the serving runtime above.
from evaluator.oracle import generate as v3_oracle_generate
from evaluator.tiny_evaluator import (
    EvalRequest as V3Request,
    PrefixHint,
    SupportRow,
    TinyBatchRuntime as V3Runtime,
    TinyEvaluator as V3Evaluator,
    batch_calibration,
    dc_pmi_score,
    logprob_metrics,
    pmi_score,
    sequence_logprob,
)


def v3_req(prompt=(1, 2), **kwargs):
    values = dict(
        request_id="v3",
        prompt=tuple(prompt),
        model="tiny",
        adapter="base",
        tenant="default",
        temperature=0.0,
        max_new_tokens=6,
        min_new_tokens=0,
        eos_token_id=None,
        weight=1.0,
    )
    values.update(kwargs)
    return V3Request(**values)


@pytest.mark.parametrize("prompt", [(), (1,), (1, 2, 3), (True,), (1, True), ("λ",), (b"x",), ("😀", "字")])
def test_v3_evaluator_oracle_parity_for_unicode_tokens(prompt):
    request = v3_req(prompt=prompt)
    runtime = V3Runtime()
    output = runtime.generate(request)
    assert output == list(v3_oracle_generate(request.as_oracle_request()))


@pytest.mark.parametrize("context_length", [0, 1, 2, 3, 5, 10])
def test_v3_context_truncation_matches_oracle(context_length):
    request = v3_req(prompt=(1, 2, 3, 4), context_length=context_length)
    assert V3Runtime().generate(request) == list(v3_oracle_generate(request.as_oracle_request()))


@pytest.mark.parametrize("seed,shard", [(None, None), (0, None), (1, None), (7, 2), (99, 3), (-1, -2)])
def test_v3_hidden_seed_shard_match_oracle(seed, shard):
    request = v3_req(seed=seed, shard=shard)
    assert V3Runtime().generate(request) == list(v3_oracle_generate(request.as_oracle_request()))


@pytest.mark.parametrize("hint", [(), (1,), (1, 2), (True,), ("λ",)])
def test_v3_prefix_hints_valid_prefixes(hint):
    prompt = tuple(hint) + (9,)
    request = v3_req(prompt=prompt, prefix_hint=hint)
    assert V3Runtime().generate(request) == list(v3_oracle_generate(request.as_oracle_request()))


@pytest.mark.parametrize("hints", [
    ((1,), (2,)),
    ((1,), (1, 2)),
    (("λ",), ("μ",)),
])
def test_v3_conflicting_prefix_hints_rejected(hints):
    request = v3_req(prompt=(1, 2, 3), prefix_hints=(PrefixHint(hints[0], name="same"), PrefixHint(hints[1], name="same")))
    with pytest.raises(ValueError):
        V3Runtime().generate(request)


@pytest.mark.parametrize("hint", [(9,), (1, 9), ("bad",)])
def test_v3_prefix_hint_prompt_conflict_rejected(hint):
    with pytest.raises(ValueError):
        V3Runtime().generate(v3_req(prompt=(1, 2, 3), prefix_hint=hint))


@pytest.mark.parametrize("rows", [(), (SupportRow(),), (SupportRow((1,), (2,)),), (SupportRow((1,), (2,)), SupportRow((1,), (2,))), ({"prompt": (1,), "continuation": (2,)},)])
def test_v3_support_rows_preserve_empty_and_duplicate_positions(rows):
    request = v3_req(support_rows=rows)
    assert len(request.support_rows) == len(rows)
    if rows:
        assert dc_pmi_score(request, (3,), support_rows=rows) == pytest.approx(dc_pmi_score(request, (3,), support_rows=rows))


@pytest.mark.parametrize("continuation", [(), (0,), (1,), (1, 2), (True,), ("λ",), (b"x",)])
def test_v3_logprob_is_finite_and_deterministic(continuation):
    request = v3_req()
    value = sequence_logprob(request, continuation)
    assert math.isfinite(value)
    assert value == sequence_logprob(request, continuation)


@pytest.mark.parametrize("count", [0, 1, 2, 3, 5])
def test_v3_logprob_metrics_weighted_count(count):
    requests = [v3_req(prompt=(i,), request_id=str(i), weight=i + 1) for i in range(count)]
    values = logprob_metrics(requests)
    assert values["num_requests"] == count
    assert math.isfinite(values["mean_logprob"])
    assert math.isfinite(values["weighted_logprob"])


@pytest.mark.parametrize("prompt", [(), (1,), (2, 3), ("λ",)])
def test_v3_pmi_and_dc_pmi_are_finite(prompt):
    request = v3_req(prompt=prompt, support_rows=(SupportRow((1,), (2,)),))
    assert math.isfinite(pmi_score(request, (3,)))
    assert math.isfinite(dc_pmi_score(request, (3,)))
    assert dc_pmi_score(request, (3,), support_rows=()) == pytest.approx(pmi_score(request, (3,)))


@pytest.mark.parametrize("predictions,labels", [
    ([], []), ([0.0], [False]), ([1.0], [True]), ([0.2, 0.8], [0, 1]),
    ([0.1, 0.9, 0.5], [False, True, True]), ([2.0, -1.0], [1, 0]),
])
def test_v3_batch_calibration_finite(predictions, labels):
    result = batch_calibration(predictions, labels)
    assert set(("brier", "logloss", "ece", "accuracy", "total_weight", "num_rows")) <= set(result)
    assert all(math.isfinite(float(value)) for value in result.values())


@pytest.mark.parametrize("weights", [(1, 1), (1, 3), (0.1, 0.9), (0, 1), (1e-9, 1)])
def test_v3_batch_calibration_weight_normalization(weights):
    result = batch_calibration([0.0, 1.0], [0, 1], weights=weights)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["total_weight"] == pytest.approx(sum(weights))


@pytest.mark.parametrize("marker", [b"", b"\x00", b"\x01", b"\xff", "\x01"])
def test_v3_byte_stop_markers_are_stable(marker):
    request = v3_req(prompt=(2, 3), max_new_tokens=8, stop_sequences=(marker,))
    output = V3Runtime().generate(request)
    oracle = list(v3_oracle_generate(request.as_oracle_request()))
    assert output == oracle


@pytest.mark.parametrize("mode", ["packed", "padded"])
def test_v3_evaluator_result_contains_records(mode):
    requests = [v3_req(prompt=(1,), request_id="same"), v3_req(prompt=(2,), request_id="same")]
    result = V3Evaluator().run(requests, mode=mode)
    assert len(result.records) == 2
    assert [record["request_id"] for record in result.records] == ["same", "same"]
    assert result.metrics["weighted_exact_match"] == pytest.approx(1.0)
