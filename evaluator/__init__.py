"""Offline evaluator package for the batched-parity candidate."""

from .oracle import OracleRequest, generate, generate_batch, prompt_state
from .tiny_evaluator import EvalRequest, EvaluationResult, TinyBatchRuntime, TinyEvaluator, weighted_metrics
from .gate import (
    DEFAULT_CASE_COUNT,
    DEFAULT_SEED,
    GateConfig,
    GateResult,
    correctness_gate,
    make_cases,
    performance_gate,
    run_gate,
    select_shard,
)

__all__ = [
    "EvalRequest",
    "EvaluationResult",
    "OracleRequest",
    "TinyBatchRuntime",
    "TinyEvaluator",
    "generate",
    "generate_batch",
    "prompt_state",
    "weighted_metrics",
    "DEFAULT_CASE_COUNT",
    "DEFAULT_SEED",
    "GateConfig",
    "GateResult",
    "correctness_gate",
    "make_cases",
    "performance_gate",
    "run_gate",
    "select_shard",
]
