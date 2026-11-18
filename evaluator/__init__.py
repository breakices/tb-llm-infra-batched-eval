"""Offline evaluator package for the batched-parity candidate."""

from .oracle import OracleRequest, generate, generate_batch, prompt_state
from .tiny_evaluator import (
    EvalRequest,
    EvaluationResult,
    PrefixHint,
    SupportRow,
    TinyBatchRuntime,
    TinyEvaluator,
    batch_calibration,
    calibration,
    dc_pmi,
    dc_pmi_score,
    logprob_metrics,
    logprob,
    conditional_logprob,
    score_logprob,
    normalize_support_rows,
    pmi,
    pmi_score,
    sequence_logprob,
    weighted_metrics,
)
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
    "PrefixHint",
    "SupportRow",
    "OracleRequest",
    "TinyBatchRuntime",
    "TinyEvaluator",
    "generate",
    "generate_batch",
    "prompt_state",
    "batch_calibration",
    "calibration",
    "dc_pmi",
    "dc_pmi_score",
    "logprob_metrics",
    "logprob",
    "conditional_logprob",
    "score_logprob",
    "normalize_support_rows",
    "pmi",
    "pmi_score",
    "sequence_logprob",
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
    "PressureGateResult",
    "run_pressure_gate",
    "seeded_pressure_requests",
]

from .performance import PressureGateResult, run_pressure_gate, seeded_pressure_requests
from .metrics_v3 import CalibrationResult, batch_logprob, calibrate_batch

__all__ += ["CalibrationResult", "batch_logprob", "calibrate_batch"]
