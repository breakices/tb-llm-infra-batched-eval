"""Compatibility, lifecycle and atomic validation at the public runtime API."""

import pytest

from llm_runtime import BatchEngine, KVCache, Request, ToyModel
from tiny_oracle import RecordingModel, expected, oracle_tokens, request


def test_positional_request_generation():
    """Generating from a six-field positional request preserves the scalar model's token sequence."""
    row = Request("t1", "m1", "adapter", (2, 7), 5, 0.25)
    assert BatchEngine({"m1": ToyModel("m1")}).generate(row) == expected(row)


def test_toy_model_token_formula():
    """Integer, boolean, Unicode and byte prompts produce the documented scalar token values."""
    model = ToyModel("check")
    for prompt in [(), (1,), (-9, 20), (False, 1), ("字", b"\xff")]:
        actual = model.next_tokens(prompt, "adapter", 0.125, 7)
        assert actual == list(oracle_tokens("check", "adapter", prompt, 0.125, 7))


def test_register_replacement_invalidates_old_cache():
    """Replacing a model under the same id recomputes its prefix and returns the new model's output."""
    original = RecordingModel()
    engine = BatchEngine({"m1": original})
    row = request()
    previous = engine.generate(row)
    replacement = RecordingModel(revision=1)
    engine.register(replacement)
    assert engine.generate(row) == [(token + 1) % 101 for token in previous]
    assert replacement.calls
    assert replacement.prefilled == list(row.prompt)


def test_clear_discards_warm_state():
    """After two clears, the next identical request performs a fresh prefill and forward."""
    model = RecordingModel()
    engine = BatchEngine({"m1": model})
    row = request()
    assert engine.generate(row) == expected(row)
    model.calls.clear()
    model.prefilled.clear()
    engine.clear()
    engine.clear()
    assert engine.generate(row) == expected(row)
    assert model.calls
    assert model.prefilled == list(row.prompt)


def test_kv_cache_detaches_lists_on_write_and_read():
    """Mutating a submitted or retrieved list leaves the stored token list intact."""
    cache = KVCache()
    source = [4, 5]
    cache.put((1, 2), source)
    source.clear()
    result = cache.get((1, 2))
    assert result == [4, 5]
    result[0] = 999
    assert cache.get((1, 2)) == [4, 5]


def test_kv_cache_preserves_distinct_typed_keys():
    """Equal-valued int and bool keys store distinct entries, survive overwrites, and all clear."""
    cache = KVCache()
    for i, token in enumerate([True, 1, False, 0]):
        cache.put((token,), [i])
    cache.put((True,), [17])
    assert len(cache) == 4
    assert [cache.get((t,)) for t in [True, 1, False, 0]] == [[17], [1], [2], [3]]
    cache.clear()
    assert len(cache) == 0
    assert cache.get((1,)) is None


def test_unknown_model_rejected_before_batch_work():
    """A later unknown model raises KeyError before any earlier row reaches the backend."""
    model = RecordingModel()
    engine = BatchEngine({"m1": model})
    with pytest.raises(KeyError):
        engine.generate_batch([request(), request(model="zzz")], mode="packed")
    assert model.calls == []
    assert model.prefilled == []


def test_invalid_generation_limit_rejected_before_batch_work():
    """Reject invalid integer token budgets and impossible minimums before backend work."""
    for field, value in [
        ("max_new_tokens", -1), ("max_new_tokens", True), ("max_new_tokens", 1.5),
        ("min_new_tokens", -1), ("min_new_tokens", False), ("min_new_tokens", 5),
    ]:
        model = RecordingModel()
        with pytest.raises(ValueError):
            BatchEngine({"m1": model}).generate_batch([request(), request(**{field: value})])
        assert model.calls == []
        assert model.prefilled == []


def test_invalid_temperature_rejected_before_batch_work():
    """Reject boolean, string and nonfinite temperatures for the entire batch before any work."""
    for value in ["0.1", True, float("nan"), float("inf"), -float("inf")]:
        model = RecordingModel()
        with pytest.raises((TypeError, ValueError)):
            BatchEngine({"m1": model}).generate_batch([request(), request(temperature=value)])
        assert model.calls == []
        assert model.prefilled == []


def test_invalid_eos_rejected_before_batch_work():
    """Reject invalid EOS id types and ids outside the vocabulary before any backend work."""
    for value in [True, -1, 31, 100, 1.5, "0"]:
        model = RecordingModel()
        with pytest.raises((TypeError, ValueError)):
            BatchEngine({"m1": model}).generate_batch([request(), request(eos_token_id=value)])
        assert model.calls == []
        assert model.prefilled == []


def test_invalid_mode_rejected_before_batch_work():
    """An unsupported layout fails without a prefill or generation call."""
    model = RecordingModel()
    with pytest.raises((TypeError, KeyError, ValueError)):
        BatchEngine({"m1": model}).generate_batch([request()], mode="ragged")
    assert model.calls == []
    assert model.prefilled == []
