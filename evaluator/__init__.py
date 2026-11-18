"""Offline evaluator package for the batched-parity candidate."""

from .oracle import OracleRequest, generate, generate_batch, prompt_state
from .tiny_evaluator import EvalRequest, EvaluationResult, TinyBatchRuntime, TinyEvaluator, weighted_metrics

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
]

