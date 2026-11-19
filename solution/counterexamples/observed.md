# Observations on the release controls

All three runs of each of these controls produced the same observations in separate, offline Linux containers.

| Control | Observed result |
|---|---|
| Golden | All six observations matched the expected result. |
| Unmodified starter and guessed control | The second tenant executed zero cold forwards, though one was required. It also reused another tenant's SQLite checkpoint. |
| Defective control | The integer prompt received the boolean continuation. Mutating a list returned from the cache changed the next read. Reopening the original tenant's scope required an additional forward after a checkpoint. |

The underlying values are saved as `observations.json` in [each run's evidence file](../../tests/assets/controls/). [Run the checks](README.md).
