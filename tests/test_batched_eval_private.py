"""Behavioral serving contracts exercised through the submitted batch engine."""

from dataclasses import replace

from llm_runtime import BatchEngine
from tiny_oracle import RecordingModel, expected, request


def make_engine():
    return BatchEngine({"m1": RecordingModel(), "m2": RecordingModel("m2")})


def test_cold_layouts_match_scalar_oracle():
    """A cold padded or packed ragged batch yields exactly the scalar continuation for each row."""
    rows = [request(prompt=p) for p in [(), (0,), (0, -3, 0), (9, 8), (True, 1)]]
    for mode in ("padded", "packed"):
        assert make_engine().generate_batch(rows, mode=mode) == [expected(r) for r in rows]


def test_internal_grouping_restores_request_order():
    """Mixing models, changing order and replaying cached rows preserve their output positions."""
    rows = [request(model="m2", prompt=(8,)), request(prompt=(11,)), request(model="m2", prompt=(2,))]
    for mode in ("padded", "packed"):
        engine = make_engine()
        assert engine.generate_batch(rows, mode=mode) == [expected(r) for r in rows]
        reordered = [rows[2], rows[0], rows[1]]
        assert engine.generate_batch(reordered, mode=mode) == [expected(r) for r in reordered]


def test_duplicate_ids_and_rows_preserve_all_positions():
    """Rows with equal request ids and repeated input rows all appear at their original positions."""
    a = request(request_id="same", prompt=(2,))
    b = request(request_id="same", prompt=(8,))
    assert make_engine().generate_batch([a, b, a], mode="packed") == [expected(a), expected(b), expected(a)]


def test_returned_duplicate_rows_and_cache_are_detached():
    """Mutating either returned duplicate list cannot change the other row or a later cache hit."""
    row = request()
    engine = make_engine()
    result = engine.generate_batch([row, row], mode="packed")
    result[0].clear()
    assert result[1] == expected(row)
    result[1][:] = [-99]
    assert engine.generate(row) == expected(row)


def test_prompt_lists_survive_padding_and_cache_hits():
    """Mutable prompt lists keep their submitted tokens across cold batches and repeated hits."""
    prompts = [[4], [7, 6, 8, 0]]
    rows = [request(prompt=prompt) for prompt in prompts]
    for mode in ("padded", "packed"):
        engine = make_engine()
        for _ in range(2):
            assert engine.generate_batch(rows, mode=mode) == [expected(r) for r in rows]
            assert prompts == [[4], [7, 6, 8, 0]]


def test_compatible_rows_use_one_batched_forward():
    """Two compatible, distinct prompts reach the backend together, each exactly once."""
    model = RecordingModel()
    rows = [request(prompt=p) for p in [(7,), (8, 9)]]
    result = BatchEngine({"m1": model}).generate_batch(rows, mode="packed")
    assert result == [expected(r) for r in rows]
    assert len(model.calls) == 1
    assert sorted(model.calls[0][1]) == [(7,), (8, 9)]


def test_duplicate_semantics_submit_one_backend_row():
    """Repeating an identical request three times produces three answers from one backend row."""
    row = request()
    model = RecordingModel()
    engine = BatchEngine({"m1": model})
    result = engine.generate_batch([row, row, row])
    assert result == [expected(row)] * 3
    assert sum(len(prompts) for _, prompts in model.calls) == 1


def test_request_metadata_and_layout_reuse_sequence():
    """Changing row id, weight and output layout leaves an already-generated sequence reusable."""
    row = request(request_id="left", weight=0.0)
    model = RecordingModel()
    engine = BatchEngine({"m1": model})
    assert engine.generate_batch([row], mode="packed") == [expected(row)]
    assert model.calls
    model.calls.clear()
    other = replace(row, request_id="right", weight=9.0)
    assert engine.generate_batch([other], mode="padded") == [expected(row)]
    assert model.calls == []


def assert_distinct_contexts(first, second):
    model = RecordingModel()
    engine = BatchEngine({"m1": model})
    for row in (first, second):
        model.calls.clear()
        assert engine.generate(row) == expected(row)
        assert model.calls, "a cold context was served from another context's cache"
    model.calls.clear()
    assert engine.generate_batch([first, second], mode="packed") == [expected(first), expected(second)]
    assert model.calls == [], "warm requests did not hit their own entries"


def test_model_context_uses_its_own_backend():
    """Requests for the same prompt under different model ids both reach their configured model."""
    models = {name: RecordingModel(name) for name in ["m1", "m2"]}
    rows = [request(model=name) for name in models]
    assert BatchEngine(models).generate_batch(rows) == [expected(r) for r in rows]
    assert all(model.calls for model in models.values())


def test_adapter_context_uses_its_own_state():
    """Changing adapters requires fresh work while later repeats hit their own cached sequences."""
    assert_distinct_contexts(request(adapter="base"), request(adapter="测试"))


def test_temperature_context_uses_its_own_state():
    """Changing temperature requires fresh work while later repeats hit their own cached sequences."""
    assert_distinct_contexts(request(temperature=0.25), request(temperature=0.5))


def test_tenant_sequence_cache_isolated_by_value_and_type():
    """Both tenant value and tenant type distinguish cold requests and subsequent warm hits."""
    for left, right in [("a", "b"), (1, True)]:
        rows = [request(tenant=left), request(tenant=right)]
        assert_distinct_contexts(*rows)
        # Independent rows may travel in the same model invocation. What is
        # forbidden is deduplicating one tenant's prompt against the other's.
        model = RecordingModel()
        result = BatchEngine({"m1": model}).generate_batch(rows, mode="packed")
        assert result == [expected(row) for row in rows]
        assert sum(len(prompts) for _, prompts in model.calls) == 2


def test_different_tenants_have_independent_cold_prefix_work():
    """Only requests from the original tenant may reuse an already-prefilled common prefix."""
    model = RecordingModel()
    engine = BatchEngine({"m1": model})
    first = request(tenant="a", prompt=(7, 8, 1))
    other = request(tenant="b", prompt=(7, 8, 2))
    last = request(tenant="a", prompt=(7, 8, 3))
    for row in (first, other):
        model.prefilled.clear()
        assert engine.generate(row) == expected(row)
        assert model.prefilled == list(row.prompt)
    model.prefilled.clear()
    assert engine.generate(last) == expected(last)
    assert model.prefilled in ([3], [8, 3])


def test_identical_prompts_with_distinct_types_keep_their_own_state():
    """Prompt tokens equal in Python arithmetic still get separate cold state for bool and int."""
    assert_distinct_contexts(request(prompt=(1,)), request(prompt=(True,)))
    assert_distinct_contexts(request(prompt=(False,)), request(prompt=(0,)))


def test_continuation_length_isolated_on_warm_cache():
    """Alternating long and short requests cannot truncate long outputs or extend short ones."""
    row = request(max_new_tokens=7)
    shorter = replace(row, max_new_tokens=2)
    for rows in [[shorter, row, shorter], [row, shorter, row]]:
        engine = make_engine()
        assert [engine.generate(r) for r in rows] == [expected(r) for r in rows]


def test_zero_length_request_does_no_model_work():
    """A zero-length request returns an empty continuation with no prefill or generation."""
    model = RecordingModel()
    engine = BatchEngine({"m1": model})
    assert engine.generate(request(max_new_tokens=0)) == []
    assert model.calls == [] and model.prefilled == []


def test_empty_batch_has_no_model_work():
    """Empty batches in both layouts return no rows and never touch the model."""
    model = RecordingModel()
    engine = BatchEngine({"m1": model})
    for mode in ("padded", "packed"):
        assert engine.generate_batch([], mode=mode) == []
    assert model.calls == [] and model.prefilled == []


def test_eos_included_at_first_eligible_position():
    """An eligible EOS is included once; disabling EOS returns the independent full sequence."""
    # The literal continuation starts [0, 17, 34, ...] for this model/prompt.
    row = request(prompt=(17,), max_new_tokens=7, eos_token_id=17, min_new_tokens=2)
    engine = make_engine()
    assert engine.generate(row) == [0, 17]
    unrestricted = replace(row, eos_token_id=None)
    assert engine.generate(unrestricted) == expected(unrestricted)


def test_early_eos_kept_and_ignored_until_minimum():
    """An EOS before the minimum remains in the output and does not prematurely stop generation."""
    # EOS is the very first token. The previous EOS must remain in the output.
    row = request(prompt=(17,), max_new_tokens=7, eos_token_id=0, min_new_tokens=3)
    output = make_engine().generate_batch([row], mode="packed")
    assert output == [[0, 17, 34, 51, 68, 85, 1]]


def test_padded_work_counts_every_submitted_row():
    """Padded statistics include duplicates and warm hits using the full rectangular input size."""
    model = RecordingModel()
    engine = BatchEngine({"m1": model})
    rows = [request(prompt=(1,)), request(prompt=(2, 3, 4)), request(prompt=(1,))]
    for invocation in [1, 2]:
        engine.generate_batch(rows, mode="padded")
        assert engine.stats["padded_tokens"] == 9 * invocation
        assert engine.stats["packed_tokens"] == 0
        assert engine.stats["mode"] == "padded"


def test_packed_work_counts_each_real_effective_token_once():
    """Packed statistics count all submitted effective prompt tokens, including duplicates and hits."""
    model = RecordingModel()
    engine = BatchEngine({"m1": model})
    row = request(prompt=(2, 3, 4, 5), context_length=2)
    rows = [row, row, request(prompt=(9,))]
    for invocation in [1, 2]:
        engine.generate_batch(rows, mode="packed")
        assert engine.stats["packed_tokens"] == 5 * invocation
        assert engine.stats["padded_tokens"] == 0
        assert engine.stats["mode"] == "packed"


def test_effective_context_is_right_truncated():
    """Context lengths zero through beyond the prompt yield the correctly truncated scalar output."""
    rows = [request(prompt=(4, 5, 6, 7), context_length=limit) for limit in [0, 1, 2, 9, None]]
    for mode in ("padded", "packed"):
        assert make_engine().generate_batch(rows, mode=mode) == [expected(r) for r in rows]


def test_unicode_and_bytes_prompts_match_the_uncached_backend():
    """Both layouts preserve non-ASCII and bytes-like tokens through batch dispatch and warm cache reads."""
    prompts = [("λ", "字", "🙂"), (b"\xc3\xa9",), (b"\xff", "e\u0301"), ("x", b"x")]
    rows = [request(prompt=prompt) for prompt in prompts]
    for mode in ("packed", "padded"):
        engine = make_engine()
        for _ in range(2):
            assert engine.generate_batch(rows, mode=mode) == [expected(row) for row in rows]


def test_scalar_only_backend_receives_the_effective_prompt():
    """A backend with only a scalar method sees the same right-truncated inputs as a batch backend."""
    backend = RecordingModel()
    backend.next_tokens_batch = None
    engine = BatchEngine({"m1": backend})
    rows = [request(prompt=(4, 3, 2), context_length=limit) for limit in [0, 1, 9]]
    assert engine.generate_batch(rows, mode="padded") == [expected(row) for row in rows]
    forwarded = [prompt for _, prompts in backend.calls for prompt in prompts]
    assert sorted(forwarded) == [(), (2,), (4, 3, 2)]


def test_rollout_seed_and_shard_isolate_sequence_state():
    """Changing either rollout seed or shard creates a cold context without disturbing warm entries."""
    original = request(seed=3, shard=7)
    assert_distinct_contexts(original, replace(original, seed=4))
    assert_distinct_contexts(original, replace(original, shard=8))


class ByteTokenModel:
    model_id = "m1"

    def next_tokens(self, prompt, adapter, temperature, max_new_tokens):
        return [65, 0xC3, 0xA9, 66, 67, 0xC3, 0xA9, 66, 68][:max_new_tokens]


def test_utf8_stop_markers_span_output_tokens():
    """Both text and byte markers terminate at the start of a UTF-8 sequence spanning three tokens."""
    for name, marker in [("stop_sequences", "éB"), ("stop", b"\xc3\xa9B")]:
        row = request(max_new_tokens=9, **{name: (marker,)})
        engine = BatchEngine({"m1": ByteTokenModel()})
        assert engine.generate(row) == [65]


def test_stop_marker_before_minimum_is_retained():
    """A marker that would shorten the result below the minimum is emitted; the next eligible match stops."""
    engine = BatchEngine({"m1": ByteTokenModel()})
    early = request(max_new_tokens=9, stop_sequences=(b"\xc3\xa9B",))
    delayed = replace(early, min_new_tokens=2)
    result = engine.generate_batch([early, delayed], mode="packed")
    assert result == [[65], [65, 0xC3, 0xA9, 66, 67]]
