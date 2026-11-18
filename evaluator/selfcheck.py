"""Fast local self-check for the tiny evaluator prototype."""

from __future__ import annotations

import json

from .oracle import generate as oracle_generate
from .tiny_evaluator import EvalRequest, TinyBatchRuntime, TinyEvaluator


def run() -> dict[str, object]:
    requests = [
        EvalRequest("same", (4, 5, 6), max_new_tokens=7, min_new_tokens=2),
        EvalRequest("same", (4, 5, 7), max_new_tokens=7, min_new_tokens=2),
        EvalRequest("same", (4, 5, 6), max_new_tokens=7, min_new_tokens=2),
        EvalRequest("short", (), max_new_tokens=3, eos_token_id=None, weight=3.0),
    ]
    packed_runtime = TinyBatchRuntime()
    packed = TinyEvaluator(packed_runtime).run(requests, mode="packed")
    padded_runtime = TinyBatchRuntime()
    padded = TinyEvaluator(padded_runtime).run(requests, mode="padded")
    assert packed.outputs == padded.outputs
    assert packed.metrics["weighted_exact_match"] == 1.0
    assert packed.stats.sequence_cache_hits == 1
    assert packed.stats.packed_tokens == sum(len(request.prompt) for request in requests)
    assert padded.stats.padded_tokens == len(requests) * max(len(request.prompt) for request in requests)

    reordered = list(reversed(requests))
    reordered_result = TinyEvaluator(TinyBatchRuntime()).run(reordered, mode="packed")
    assert reordered_result.outputs == list(reversed(packed.outputs))

    # Bool is a valid numeric token value.  It has the same arithmetic value
    # as its corresponding integer, but it must occupy a separate cache row.
    identity_requests = [
        EvalRequest("int-token", (1,), max_new_tokens=3, eos_token_id=None),
        EvalRequest("bool-token", (True,), max_new_tokens=3, eos_token_id=None),
    ]
    identity_runtime = TinyBatchRuntime()
    identity_outputs = identity_runtime.generate_batch(identity_requests)
    assert identity_outputs[0] == identity_outputs[1]
    assert identity_runtime.sequence_cache_size == 2
    assert identity_runtime.prefix_cache_size == 2
    assert identity_runtime.stats.sequence_cache_hits == 0

    # The EOS contract is exercised with oracle-generated expected output.
    eos_request = EvalRequest("eos", (8, 1), max_new_tokens=20, min_new_tokens=4, eos_token_id=0)
    eos_output = TinyBatchRuntime().generate(eos_request)
    expected_eos = list(oracle_generate(eos_request.as_oracle_request()))
    assert eos_output == expected_eos
    assert len(eos_output) <= eos_request.max_new_tokens
    if 0 in eos_output:
        assert eos_output.index(0) + 1 >= eos_request.min_new_tokens

    return {
        "ok": True,
        "packed": packed.as_dict(),
        "padded": padded.as_dict(),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
