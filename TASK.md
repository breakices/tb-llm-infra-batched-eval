# Batched evaluation parity and KV-cache isolation

Task contract version 3.1

`repo/llm_runtime.py` is a dependency-free toy serving runtime used by a
local evaluation harness. It contains `Request`, `ToyModel`, `KVCache`, and
`BatchEngine`. The original six `Request` fields and the public methods on the
existing classes are used by older callers, so preserve their positional
meaning and return shapes.

Improve the runtime so a batch has the same results as independent uncached
generation while making batching and prefix reuse observable and safe.

## Request and generation semantics

`Request` keeps the existing fields `tenant`, `model`, `adapter`, `prompt`,
`max_new_tokens`, and `temperature`. It may add optional metadata fields for
`min_new_tokens`, `eos_token_id`, `request_id`, and `weight` after those
fields. `request_id` identifies an input row only. It must not make two equal
requests compute different tokens.

`max_new_tokens` bounds the returned continuation. A value of zero returns an
empty list without a model forward. `eos_token_id=None` disables early stop.
When EOS is configured, include the first EOS token and stop there once at
least `min_new_tokens` tokens have been emitted. An EOS observed earlier is
kept as a token and does not stop generation. Reject negative limits and an
impossible minimum with a clear `ValueError`.

The validation boundary is explicit. `max_new_tokens` and `min_new_tokens`
must be integer values and booleans are invalid configuration values. An
`eos_token_id` is either `None` or an integer in the tiny vocabulary range
`0..30`, and booleans are invalid there as well. `temperature` must be a
finite real number and must reject booleans, strings, NaN, and infinities.
Invalid requests may be constructed as data, but `generate_batch` must reject
them before model or cache work begins. Prompt tokens may be integers,
including booleans, and their arithmetic value may match, while cache
identity must keep `True` distinct from `1` and `False` distinct from `0`.
An unknown `Request.model` id must preserve the existing `KeyError` behavior so
callers that catch `KeyError` continue to work. It may use a custom exception
that is also a `ValueError` for the new validation boundary, but a plain
`ValueError` is insufficient.

## Batch and cache behavior

`BatchEngine.generate_batch(requests, mode=...)` accepts `mode="padded"` and
`mode="packed"`. The two modes must return byte-for-byte equal token lists.
Padded layouts must ignore padding tokens. Packed layouts must preserve each
row's prompt boundaries. The returned outer list always follows input order,
including when rows have duplicate `request_id` values or duplicate request
contents. Internal grouping or sorting is allowed only when results are
scattered back to their original positions.

Compatible rows should share a model batch forward. Exact duplicate semantic
requests should share sequence work. Requests with prompts sharing a prefix
should reuse prefix state. Prefix and sequence entries must include tenant,
model, adapter, temperature, and the current model implementation generation
in their scope. `register()` replacing a model id and `clear()` both invalidate
all related state. Returned token lists and cache values must be detached from
each other.

Preserve the `KVCache` container behavior used by existing callers. Its
`put(key, value)`, `get(key)`, and `clear()` methods remain available, and
`len(cache)` returns the number of distinct stored entries. `get()` must return
a detached value, so mutating the returned list cannot mutate the cached value.

Tenant is part of the prefix and sequence cache scope, using the tenant's
value and type as cache identity. Rows from different tenants must never reuse
one another's cached prefix or sequence state, even
when every other request field is equal. This applies to both a full sequence
hit and a shared prompt prefix hit. For two otherwise identical rows with
`tenant="a"` and `tenant="b"`, each tenant gets an independent cache entry and
independent prefix work. Their deterministic token values may happen to be
equal, but a hit for one tenant must not satisfy the other tenant.

This is a normative work-isolation requirement, not only a key-shape hint. For
example, a packed call equivalent to

```python
engine.generate_batch(
    [
        Request(tenant="a", model="m1", adapter=None,
                prompt=(1, 2, 3), max_new_tokens=2, temperature=0.0),
        Request(tenant="b", model="m1", adapter=None,
                prompt=(1, 2, 3), max_new_tokens=2, temperature=0.0),
    ],
    mode="packed",
)
```

must perform two independent prefix works or model forwards for the two
tenants. Neither row may be served by the other's prefix entry, even when the
returned continuations are byte-for-byte equal. A later request from tenant
`"a"` may hit only tenant `"a"`'s entry, and likewise for tenant `"b"`.
Tenant value and type are also part of the batching/grouping compatibility key.
Rows from different tenants may not share one model forward or one prefix work
in the same batch. In the example above, the model's forward-call counter must
increase by at least two on a cold cache, regardless of whether the two rows'
token values happen to be equal.

Runtime statistics expose integer counters named `padded_tokens` and
`packed_tokens`. They are cumulative for the engine. For every padded batch,
`padded_tokens` increases by `max(len(effective_prompt)) * len(requests)` and
`packed_tokens` does not increase. For every packed batch, `packed_tokens`
increases by `sum(len(effective_prompt) for request in requests)` and
`padded_tokens` does not increase. These updates happen for every input row,
including rows served from a cache and duplicate rows. The selected layout is
also exposed as `stats["mode"]`. A repeated request may reuse its cached
result without changing its tokens, and the cache must not mutate the input
prompt. Padded layouts count the rectangular prompt surface, including
padding positions. Packed layouts count each real prompt token once.

## Local evaluator contract

The repository includes `evaluator/` with a network-free `TinyBatchRuntime`,
`TinyEvaluator`, padded and packed layout helpers, weighted exact-match and
token metrics, and an independent arithmetic oracle. This evaluator is for
local parity checks only. Weighted metrics use the sum of row weights as the
denominator, keep duplicate ids as separate rows, and remain unchanged when
input rows are permuted.

Keep the implementation dependency-free. Do not use network services,
credentials, model downloads, or changes outside the task's `repo/` runtime
when implementing the requested runtime behavior.

## v3 local evaluator contract

The local evaluator now accepts support rows and named prefix hints, rejects
conflicting hints, truncates to a finite right context, and isolates caches by
model, adapter, tenant, normalized temperature, seed, shard, and model
lifecycle. Padded and packed execution must remain byte-for-byte identical.
Generation supports EOS and minimum lengths together with byte/token stop
markers. Independent logprob, PMI, DC-PMI, and weighted calibration helpers
must stay finite for empty support and duplicate rows. The deterministic gate
runs its 192-case corpus and hidden-style shards only after exact oracle
correctness passes, then checks model-work pressure.

## Version 3.1 acceptance boundary

The 351-case contract is unchanged. The independent oracle and evaluator
remain outside the candidate runtime and must not be modified to make a
candidate pass. A valid implementation preserves input prompt objects,
returns detached per-row lists, restores original input order after any
batch grouping, and produces byte-for-byte equal outputs for padded and
packed execution. A zero-length continuation performs no model forward.
