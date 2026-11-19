"""Contract tests for the dependency-free tiny evaluator and oracle."""

import math

import pytest

from evaluator.oracle import OracleRequest, generate as oracle_generate, generate_batch as oracle_batch
from evaluator.tiny_evaluator import (
    EvalRequest,
    TinyBatchRuntime,
    TinyEvaluator,
    pack_batch,
    pad_batch,
    weighted_metrics,
)


def req(number=1, **kwargs):
    values = dict(
        request_id="r" + str(number),
        prompt=(number, number + 1),
        model="tiny",
        adapter="base",
        tenant="default",
        temperature=0.0,
        max_new_tokens=5,
        min_new_tokens=0,
        eos_token_id=None,
        weight=1.0,
    )
    values.update(kwargs)
    return EvalRequest(**values)


def test_oracle_is_deterministic():
    request = req()
    assert oracle_generate(request.as_oracle_request()) == oracle_generate(request.as_oracle_request())


def test_oracle_batch_matches_scalar():
    requests = [req(i) for i in range(3)]
    assert oracle_batch([r.as_oracle_request() for r in requests]) == [oracle_generate(r.as_oracle_request()) for r in requests]


def test_oracle_unknown_mode_rejected():
    with pytest.raises(ValueError):
        oracle_batch([req().as_oracle_request()], mode="invalid")


def test_pad_batch_keeps_lengths():
    requests = [req(1, prompt=(1,)), req(2, prompt=(2, 3, 4))]
    rows, lengths = pad_batch(requests, pad_token_id=-99)
    assert lengths == [1, 3]
    assert rows == [[1, -99, -99], [2, 3, 4]]


def test_pack_batch_keeps_offsets():
    requests = [req(1, prompt=(1,)), req(2, prompt=(2, 3, 4))]
    tokens, offsets = pack_batch(requests)
    assert tokens == [1, 2, 3, 4]
    assert offsets == [0, 1, 4]


def test_pad_and_pack_empty_batch():
    assert pad_batch([]) == ([], [])
    assert pack_batch([]) == ([], [0])


def test_runtime_padded_matches_oracle():
    request = req()
    assert TinyBatchRuntime().generate_batch([request], mode="padded") == [list(oracle_generate(request.as_oracle_request()))]


def test_runtime_packed_matches_oracle():
    request = req()
    assert TinyBatchRuntime().generate_batch([request], mode="packed") == [list(oracle_generate(request.as_oracle_request()))]


def test_runtime_modes_have_equal_outputs():
    requests = [req(i, prompt=(i, i + 1, i + 2)) for i in range(4)]
    assert TinyBatchRuntime().generate_batch(requests, mode="padded") == TinyBatchRuntime().generate_batch(requests, mode="packed")


def test_runtime_rejects_bad_mode():
    with pytest.raises(ValueError):
        TinyBatchRuntime().generate_batch([req()], mode="ragged")


def test_runtime_preserves_input_order():
    requests = [req(i, prompt=(i,)) for i in (5, 1, 3)]
    outputs = TinyBatchRuntime().generate_batch(requests)
    assert outputs == [list(oracle_generate(r.as_oracle_request())) for r in requests]


def test_runtime_reordering_only_reorders_outputs():
    requests = [req(i, prompt=(i,)) for i in (1, 2, 3)]
    runtime = TinyBatchRuntime()
    first = runtime.generate_batch(requests)
    second = runtime.generate_batch(list(reversed(requests)))
    assert second == list(reversed(first))


def test_duplicate_ids_keep_rows():
    requests = [req(1, request_id="same"), req(2, request_id="same")]
    outputs = TinyBatchRuntime().generate_batch(requests)
    assert outputs == [list(oracle_generate(r.as_oracle_request())) for r in requests]


def test_identical_rows_are_cached():
    request = req()
    runtime = TinyBatchRuntime()
    runtime.generate_batch([request, request])
    assert runtime.stats.model_calls == 1
    assert runtime.stats.sequence_cache_hits == 1


def test_shared_prefix_cache_hits():
    runtime = TinyBatchRuntime()
    runtime.generate_batch([req(1, prompt=(9, 8, 7, 1)), req(2, prompt=(9, 8, 7, 2))])
    assert runtime.stats.prefix_cache_hits > 0


def test_shared_prefix_cache_can_be_disabled():
    runtime = TinyBatchRuntime(shared_prefix_cache=False)
    runtime.generate_batch([req(1, prompt=(9, 8, 7, 1)), req(2, prompt=(9, 8, 7, 2))])
    assert runtime.stats.prefix_cache_hits == 0


def test_shared_prefix_is_context_scoped():
    runtime = TinyBatchRuntime()
    runtime.generate_batch([req(1, tenant="a", prompt=(9, 8)), req(2, tenant="b", prompt=(9, 8))])
    # The first token in each tenant has no reusable state from the other.
    assert runtime.stats.prefix_cache_misses >= 4


def test_stats_mode_records_last_mode():
    runtime = TinyBatchRuntime()
    runtime.generate_batch([req()], mode="padded")
    assert runtime.stats.mode == "padded"
    runtime.generate_batch([req(2)], mode="packed")
    assert runtime.stats.mode == "packed"


def test_padded_stats_include_padding():
    runtime = TinyBatchRuntime()
    runtime.generate_batch([req(1, prompt=(1,)), req(2, prompt=(2, 3, 4))], mode="padded")
    assert runtime.stats.padded_tokens == 2 * 3


def test_packed_stats_include_only_real_tokens():
    runtime = TinyBatchRuntime()
    runtime.generate_batch([req(1, prompt=(1,)), req(2, prompt=(2, 3, 4))], mode="packed")
    assert runtime.stats.packed_tokens == 4


def test_clear_drops_sequence_and_prefix_cache():
    runtime = TinyBatchRuntime()
    runtime.generate_batch([req()])
    assert runtime.sequence_cache_size and runtime.prefix_cache_size
    runtime.clear()
    assert runtime.sequence_cache_size == 0
    assert runtime.prefix_cache_size == 0


def test_reset_stats_does_not_drop_cache():
    runtime = TinyBatchRuntime()
    runtime.generate_batch([req()])
    runtime.reset_stats()
    runtime.generate_batch([req()])
    assert runtime.stats.sequence_cache_hits == 1


def test_min_new_tokens_delays_eos():
    base = req(max_new_tokens=10, min_new_tokens=0, eos_token_id=0)
    delayed = req(max_new_tokens=10, min_new_tokens=3, eos_token_id=0)
    short = oracle_generate(base.as_oracle_request())
    long = oracle_generate(delayed.as_oracle_request())
    assert len(long) >= 3
    if 0 in short:
        assert len(short) == short.index(0) + 1


def test_eos_none_runs_to_maximum():
    request = req(max_new_tokens=7, eos_token_id=None)
    assert len(TinyBatchRuntime().generate(request)) == 7


def test_zero_max_tokens_is_empty():
    request = req(max_new_tokens=0)
    assert TinyBatchRuntime().generate(request) == []


def test_invalid_minimum_is_rejected():
    with pytest.raises(ValueError):
        TinyBatchRuntime().generate(req(max_new_tokens=1, min_new_tokens=2))


def test_invalid_weight_is_rejected():
    with pytest.raises(ValueError):
        TinyBatchRuntime().generate_batch([req(weight=-1)])


def test_weighted_metrics_exact_match():
    requests = [req(1, weight=1), req(2, weight=3)]
    outputs = [list(oracle_generate(r.as_oracle_request())) for r in requests]
    metrics, records = weighted_metrics(requests, outputs)
    assert metrics["weighted_exact_match"] == 1.0
    assert metrics["total_weight"] == 4.0
    assert len(records) == 2


def test_weighted_metrics_not_count_average():
    requests = [req(1, weight=1), req(2, weight=9)]
    correct = list(oracle_generate(requests[1].as_oracle_request()))
    metrics, _ = weighted_metrics(requests, [[999], correct])
    assert metrics["weighted_exact_match"] == pytest.approx(0.9)
    assert metrics["exact_match"] == 0.5


def test_weighted_metrics_preserves_duplicate_ids():
    requests = [req(1, request_id="same"), req(2, request_id="same")]
    result = TinyEvaluator().run(requests)
    assert [record["request_id"] for record in result.records] == ["same", "same"]
    assert len(result.records) == 2


def test_weighted_metrics_rejects_length_mismatch():
    with pytest.raises(ValueError):
        weighted_metrics([req()], [])


def test_weighted_metrics_rejects_zero_total_weight():
    with pytest.raises(ValueError):
        weighted_metrics([req(weight=0)], [[]])


def test_tiny_evaluator_returns_serializable_result():
    result = TinyEvaluator().run([req()])
    value = result.as_dict()
    assert set(value) == {"outputs", "metrics", "stats", "records"}
    assert value["outputs"] == result.outputs


def test_tiny_evaluator_supports_both_modes():
    requests = [req(i) for i in range(3)]
    assert TinyEvaluator().run(requests, mode="padded").outputs == TinyEvaluator().run(requests, mode="packed").outputs


def test_tiny_evaluator_uses_reference_when_provided():
    request = req(reference=(99,))
    result = TinyEvaluator().run([request])
    assert result.metrics["weighted_exact_match"] == 0.0
    assert result.records[0]["expected"] == [99]


def test_tiny_evaluator_accepts_empty_batch():
    result = TinyEvaluator().run([])
    assert result.outputs == []
    assert result.metrics["exact_match"] == 1.0


def test_evaluator_metric_is_finite():
    result = TinyEvaluator().run([req()])
    assert all(math.isfinite(value) for value in result.metrics.values())


def test_oracle_context_changes_state():
    base = req().as_oracle_request()
    altered = req(tenant="other").as_oracle_request()
    assert oracle_generate(base) != oracle_generate(altered)


def test_oracle_temperature_changes_state():
    base = req().as_oracle_request()
    altered = req(temperature=0.5).as_oracle_request()
    assert oracle_generate(base) != oracle_generate(altered)


def test_runtime_outputs_are_detached():
    runtime = TinyBatchRuntime()
    request = req()
    first = runtime.generate(request)
    first[0] = -10
    assert runtime.generate(request) == list(oracle_generate(request.as_oracle_request()))


def test_runtime_request_prompt_is_not_mutated():
    request = req(prompt=(4, 5, 6))
    before = request.prompt
    TinyBatchRuntime().generate(request)
    assert request.prompt == before

