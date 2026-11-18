# LLM infrastructure parity task

Task contract version 3.1 accompanies `TASK.md`. The candidate implementation
lives in `repo/llm_runtime.py`. The runtime must preserve the original request
API while providing deterministic independent-generation parity, safe batching,
and tenant-scoped prefix and sequence caching.

Tenant value and type are required components of every prefix and sequence
cache identity. Equal prompts from two tenants must create separate entries and separate
prefix work, even when their deterministic outputs are equal. The runtime
statistics must always contain cumulative integer `padded_tokens` and
`packed_tokens` counters. A padded batch adds the rectangular width times row
count to `padded_tokens`, while a packed batch adds the sum of real prompt
lengths to `packed_tokens`. The other counter is unchanged for that call, and
`stats["mode"]` records the latest layout.

The separation is observable work isolation. For a packed batch containing two
otherwise identical requests with `prompt=(1, 2, 3)` and tenants `"a"` and
`"b"`, the engine must perform two independent prefix/model works. A cache hit
for tenant `"a"` can never satisfy tenant `"b"`, and a later hit must remain
scoped to the tenant that created the entry. Tenant value and type must also be
part of the batching/grouping key, so those two rows cannot share a single
model forward. On a cold cache, `ToyModel.forward_calls` must increase by at
least two for that example.

The v3.1 boundary is explicit. Prompt tokens accept integers, including
booleans, while typed cache identity keeps `True` separate from `1` and
`False` separate from `0`. Configuration fields reject boolean integers,
non-finite or non-real temperatures, out-of-range EOS ids, negative limits,
and impossible minimum lengths during generation. Padded and packed layouts
must return identical rows, preserve input order and prompt objects, and count
rectangular versus real prompt tokens in runtime statistics. Returned rows and
cache values are detached. Unknown model ids retain the existing `KeyError`
compatibility for callers that catch it, with an optional dual `ValueError`
subclass for validation callers.

The local evaluator and its independent oracle are under `evaluator/`. They
are verification infrastructure and remain outside the candidate runtime.

The existing `KVCache` API remains usable by callers. `put`, `get`, and
`clear` stay available, `len(cache)` reports the number of stored entries, and
values returned by `get` are detached copies.

Run the local contract with:

```bash
PYTHONPATH=repo:tests:. uvx --with pytest pytest -q
PYTHONPATH=. python -m evaluator.selfcheck
```

The v3.1 package contains 351 contract tests. A full score requires all tests
and a successful independent self-check.
