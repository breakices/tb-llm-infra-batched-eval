# Batched evaluation parity and KV-cache isolation

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
