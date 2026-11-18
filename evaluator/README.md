# Local batched evaluator

The evaluator is dependency-free and does not call a network service.  It has
two independent pieces.

- `oracle.py` computes expected tokens one request at a time.  Its EOS rule
  honours `min_new_tokens` before stopping.
- `tiny_evaluator.py` runs a deterministic runtime through `padded` or
  `packed` prompts.  It keys sequence and prefix state by model, adapter,
  tenant, and temperature, keeps duplicate rows in input order, and reports
  weighted exact-match and token-accuracy metrics.

Run the fast prototype check from the checkout root.

```bash
python3 -m evaluator.selfcheck
```

The scorer accepts a candidate runtime directory.  This lets the NOP checkout
and the golden checkout be measured without changing either one.

```bash
python3 -m evaluator.score --repo repo --out /tmp/nop-score.json
python3 -m evaluator.score --repo golden-repo --out /tmp/golden-score.json
```

The private suite is intentionally kept outside the runtime package in a
normal benchmark deployment.  The local prototype stores it under `tests/` so
the Golden/NOP loop can be run on the development machine.

## v3 parity surface

The v3 evaluator keeps the v1/v2 constructors and adds `SupportRow` and
`PrefixHint` records, optional support aliases, finite right-context
truncation, deterministic seed/shard isolation, and byte/token stop markers.
`TinyBatchRuntime` runs padded and packed layouts through the same independent
oracle while preserving duplicate input rows and weights. `sequence_logprob`,
`pmi_score`, `dc_pmi_score`, and `batch_calibration` provide deterministic
probability diagnostics. `evaluator.gate` builds a 192-row Unicode/bool/int
corpus and runs correctness before the model-work pressure gate. Use
`python -m evaluator.selfcheck` for the complete local closure.
