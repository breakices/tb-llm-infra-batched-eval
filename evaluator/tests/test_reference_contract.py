"""Maintainer checks for the reference evaluator; excluded from candidate grading."""

import math

import pytest

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
