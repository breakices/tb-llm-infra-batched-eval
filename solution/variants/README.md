# Alternative complete submissions

Apply each patch with `git apply` from a fresh, prepared candidate root. Each patch contains the serving changes and the native harness/response-cache integration.

- **Variant A** uses tagged immutable tuples for cached token-list payloads. Writes take a value snapshot, and each read creates a new list. Prefix-state values retain their original representation. Keys still carry the type and value of every semantic component.
- **Variant B** uses bounded input-order chunks in the harness adapter, so a sorted-output scatter is unnecessary. The engine still groups compatible rows and deduplicates work. Its prefix lookup walks increasing prefix lengths until the first miss. Inserting a prefix records each ancestor and invalidation clears the whole cache, so the last hit is exactly the longest registered prefix.
- **The calibration equivalent** combines both variations. It is also a complete patch from the initial candidate.

The test contract accepts these differences in storage and scheduling strategy. The Linux container matrix in `tests/assets/controls/matrix.json` lists all named methods and the repetition count for each control. Each compressed run record contains all collected case outcomes.
