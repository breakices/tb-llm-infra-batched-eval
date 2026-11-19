# Reference evaluator

`oracle.py` computes expected tokens for the separate `TinyBatchRuntime`.
`tiny_evaluator.py` implements the reference padded/packed model, prefix hints,
support rows, log-probability metrics, and weighted evaluation.
`gate.py` and `performance.py` exercise this reference implementation.

From the checkout root, maintainers can run the verifier's own checks:

```sh
python -m pytest -q evaluator/tests
python -m evaluator.selfcheck
```

These checks validate verifier code independently of submissions. Candidate
scoring, including checks that monitor work performed by the submitted serving
runtime, is run by `tests/score.py`. See the task README for its invocation.
