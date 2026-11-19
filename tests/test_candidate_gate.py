"""Verify the candidate that the runner placed first on its import path."""

from copy import deepcopy
from dataclasses import replace

import pytest

from candidate_gate import (
    assert_parity, assert_row, corpus, expected, generate, observed_engine, work_budget,
)


def cold_batch(mode):
    rows = corpus()
    wanted = [expected(row) for row in rows]
    # Do not turn a submitted runtime error into 192 fixture-setup errors.
    # Record the batch outcome once, and grade each expected row independently.
    try:
        engine, _ = observed_engine()
        actual = generate(engine, rows, mode)
    except Exception as exc:
        actual = exc
    return wanted, actual


@pytest.fixture(scope="module")
def packed_batch():
    return cold_batch("packed")


@pytest.fixture(scope="module")
def padded_batch():
    return cold_batch("padded")


@pytest.mark.parametrize("row", range(192), ids=lambda row: f"row-{row:03d}")
def test_packed_cold_192(packed_batch, row):
    assert_row(*packed_batch, "packed", row)


@pytest.mark.parametrize("row", range(192), ids=lambda row: f"row-{row:03d}")
def test_padded_cold_192(padded_batch, row):
    assert_row(*padded_batch, "padded", row)


def test_replay_work_budget_192():
    requests = corpus()
    engine, work = observed_engine()
    assert_parity(engine, requests, "packed")
    max_forwards, required_rows, minimum_prefill = work_budget(requests)
    assert 0 < work.forwards <= max_forwards, f"cold forwards={work.forwards}, budget={max_forwards}"
    assert work.generated == required_rows, "cold generation must account for each tenant and distinct backend input"
    for context, required in minimum_prefill.items():
        assert work.prefilled[context] >= required, (
            f"{context}: observed {work.prefilled[context]} prefilled tokens, "
            f"tenant-isolated prompts require at least {required}"
        )

    # Reuse across layouts, reordered rows, duplicate ids and changed weights.
    replay = [replace(r, request_id="repeat", weight=1) for r in reversed(requests)]
    before = deepcopy(work)
    assert_parity(engine, replay, "padded")
    assert work == before, f"replay repeated model work: before={before}, after={work}"
