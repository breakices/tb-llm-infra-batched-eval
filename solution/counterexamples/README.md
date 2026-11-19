# Offline cache observations

`cache_observations.py` calls the same public runtime, model registry, and SQLite response-cache entrypoints that callers use. Run it with Python from the task image, passing the candidate checkout as its only import root for the implementation.

```sh
python -I -B /author-probes/cache_observations.py /workspace/repo --require correct
```

For an intentionally faulty control, use `--require defect`. The program prints the expected and observed values for each check. It exits successfully when the requested condition holds. The backend is deterministic and runs on CPU. Its boolean token has a different vocabulary entry from integer one. Its call counter observes actual invocations at the injected backend.

| Observation | Observable contract |
|---|---|
| Boolean and integer prompt | The respective continuations are `[19, 20]` and `[1, 2]`. |
| Same input in a new tenant | One new forward is needed even if the two tenants would return equal tokens. |
| Model replacement | The continuation changes to the new backend's `[29, 30]`. |
| External list edit | Reading a modified result still returns the stored `[3, 4]`. |
| Partial checkpoint after restart | A second tenant executes a new forward. Reopening the writer's scope reuses the checkpoint. |

To reproduce the comparison, run `tests/calibration/check.py --image <task-image> --out <output-directory>` from the author checkout. The matrix uses a new container and temporary SQLite database for every repetition. The full `observations.json` from each run is embedded in its compressed evidence record. A comparison of the measured results is in `observed.md`.
