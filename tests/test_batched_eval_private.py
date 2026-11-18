"""Private parity tests for the hard batched-evaluation candidate.

These tests intentionally exercise the public runtime through a tiny local
oracle.  They do not use network services, credentials, or implementation
internals.  The 36 tests are kept in one file so ``score.py`` can count and
report the complete private suite.
"""

import inspect

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

