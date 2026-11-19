# 单测清单

具名方法 47 个，其中 4 个直接针对历史试答中的行为。每项的参数行均通过才记该方法通过。

| ID | 行数 | 方法 | 维度 | 判定行为 |
|---|---:|---|---|---|
| U1 | 1 | `test_positional_request_generation` | 回归安全 | Generating from a six-field positional request preserves the scalar model's token sequence. |
| U2 | 1 | `test_toy_model_token_formula` | 语义正确性 | Integer, boolean, Unicode and byte prompts produce the documented scalar token values. |
| U3 | 1 | `test_register_replacement_invalidates_old_cache` | 改一查十完备性 | Replacing a model under the same id recomputes its prefix and returns the new model's output. |
| U4 | 1 | `test_clear_discards_warm_state` | 回归安全 | After two clears, the next identical request performs a fresh prefill and forward. |
| U5 | 1 | `test_kv_cache_detaches_lists_on_write_and_read` | 回归安全 | Mutating a submitted or retrieved list leaves the stored token list intact. |
| U6 | 1 | `test_kv_cache_preserves_distinct_typed_keys` | 改一查十完备性 | Equal-valued int and bool keys store distinct entries, survive overwrites, and all clear. |
| U7 | 1 | `test_unknown_model_rejected_before_batch_work` | 边界与异常 | A later unknown model raises KeyError before any earlier row reaches the backend. |
| U8 | 1 | `test_invalid_generation_limit_rejected_before_batch_work` | 边界与异常 | Reject invalid integer token budgets and impossible minimums before backend work. |
| U9 | 1 | `test_invalid_temperature_rejected_before_batch_work` | 边界与异常 | Reject boolean, string and nonfinite temperatures for the entire batch before any work. |
| U10 | 1 | `test_invalid_eos_rejected_before_batch_work` | 边界与异常 | Reject invalid EOS id types and ids outside the vocabulary before any backend work. |
| U11 | 1 | `test_invalid_mode_rejected_before_batch_work` | 边界与异常 | An unsupported layout fails without a prefill or generation call. |
| U12 | 1 | `test_cold_layouts_match_scalar_oracle` | 语义正确性 | A cold padded or packed ragged batch yields exactly the scalar continuation for each row. |
| U13 | 1 | `test_internal_grouping_restores_request_order` | 改一查十完备性 | Mixing models, changing order and replaying cached rows preserve their output positions. |
| U14 | 1 | `test_duplicate_ids_and_rows_preserve_all_positions` | 改一查十完备性 | Rows with equal request ids and repeated input rows all appear at their original positions. |
| U15 | 1 | `test_returned_duplicate_rows_and_cache_are_detached` | 改一查十完备性 | Mutating either returned duplicate list cannot change the other row or a later cache hit. |
| U16 | 1 | `test_prompt_lists_survive_padding_and_cache_hits` | 回归安全 | Mutable prompt lists keep their submitted tokens across cold batches and repeated hits. |
| U17 | 1 | `test_compatible_rows_use_one_batched_forward` | 语义正确性 | Two compatible, distinct prompts reach the backend together, each exactly once. |
| U18 | 1 | `test_duplicate_semantics_submit_one_backend_row` | 改一查十完备性 | Repeating an identical request three times produces three answers from one backend row. |
| U19 | 1 | `test_request_metadata_and_layout_reuse_sequence` | 改一查十完备性 | Changing row id, weight and output layout leaves an already-generated sequence reusable. |
| U20 | 1 | `test_model_context_uses_its_own_backend` | 改一查十完备性 | Requests for the same prompt under different model ids both reach their configured model. |
| U21 | 1 | `test_adapter_context_uses_its_own_state` | 改一查十完备性 | Changing adapters requires fresh work while later repeats hit their own cached sequences. |
| U22 | 1 | `test_temperature_context_uses_its_own_state` | 改一查十完备性 | Changing temperature requires fresh work while later repeats hit their own cached sequences. |
| U23 | 1 | `test_tenant_sequence_cache_isolated_by_value_and_type` | 单轮反幻觉 | Both tenant value and tenant type distinguish cold requests and subsequent warm hits. |
| U24 | 1 | `test_different_tenants_have_independent_cold_prefix_work` | 单轮反幻觉 | Only requests from the original tenant may reuse an already-prefilled common prefix. |
| U25 | 1 | `test_identical_prompts_with_distinct_types_keep_their_own_state` | 单轮反幻觉 | Prompt tokens equal in Python arithmetic still get separate cold state for bool and int. |
| U26 | 1 | `test_continuation_length_isolated_on_warm_cache` | 改一查十完备性 | Alternating long and short requests cannot truncate long outputs or extend short ones. |
| U27 | 1 | `test_zero_length_request_does_no_model_work` | 边界与异常 | A zero-length request returns an empty continuation with no prefill or generation. |
| U28 | 1 | `test_empty_batch_has_no_model_work` | 边界与异常 | Empty batches in both layouts return no rows and never touch the model. |
| U29 | 1 | `test_eos_included_at_first_eligible_position` | 语义正确性 | An eligible EOS is included once; disabling EOS returns the independent full sequence. |
| U30 | 1 | `test_early_eos_kept_and_ignored_until_minimum` | 边界与异常 | An EOS before the minimum remains in the output and does not prematurely stop generation. |
| U31 | 1 | `test_padded_work_counts_every_submitted_row` | 改一查十完备性 | Padded statistics include duplicates and warm hits using the full rectangular input size. |
| U32 | 1 | `test_packed_work_counts_each_real_effective_token_once` | 改一查十完备性 | Packed statistics count all submitted effective prompt tokens, including duplicates and hits. |
| U33 | 1 | `test_effective_context_is_right_truncated` | 语义正确性 | Context lengths zero through beyond the prompt yield the correctly truncated scalar output. |
| U34 | 1 | `test_unicode_and_bytes_prompts_match_the_uncached_backend` | 语义正确性 | Both layouts preserve non-ASCII and bytes-like tokens through batch dispatch and warm cache reads. |
| U35 | 1 | `test_scalar_only_backend_receives_the_effective_prompt` | 回归安全 | A backend with only a scalar method sees the same right-truncated inputs as a batch backend. |
| U36 | 1 | `test_rollout_seed_and_shard_isolate_sequence_state` | 改一查十完备性 | Changing either rollout seed or shard creates a cold context without disturbing warm entries. |
| U37 | 1 | `test_utf8_stop_markers_span_output_tokens` | 语义正确性 | Both text and byte markers terminate at the start of a UTF-8 sequence spanning three tokens. |
| U38 | 1 | `test_stop_marker_before_minimum_is_retained` | 边界与异常 | A marker that would shorten the result below the minimum is emitted; the next eligible match stops. |
| U39 | 192 | `test_packed_cold_192` | 语义正确性 | A cold packed batch returns every row in order with the independently expected continuation. |
| U40 | 192 | `test_padded_cold_192` | 语义正确性 | A cold padded batch returns every row in order with the independently expected continuation. |
| U41 | 1 | `test_replay_work_budget_192` | 改一查十完备性 | A 192-row cold shard stays within the model-work budget; replay preserves answers with zero additional work. |
| U42 | 1 | `test_harness_registry_renders_instances_in_order` | 语义正确性 | A registered backend accepts native Instance rows and returns UTF-8-based text in row order. |
| U43 | 1 | `test_harness_stops_at_first_text_match` | 语义正确性 | Stop at the first textual occurrence, excluding the marker for scalar and list forms. |
| U44 | 1 | `test_harness_response_cache_reopens_and_scatters_hits` | 改一查十完备性 | New Python processes with different hash seeds share only persisted hits in input order. |
| U45 | 1 | `test_harness_response_cache_isolates_serving_scopes` | 改一查十完备性 | Tenant, model, adapter, revision and independently replaced model lifetimes isolate saved text. |
| U46 | 1 | `test_harness_partial_checkpoint_uses_response_cache_scope` | 改一查十完备性 | An interrupted batch checkpoint survives reopening and is visible only to its writer's scope. |
| U47 | 1 | `test_harness_legacy_response_keys_remain_readable` | 回归安全 | An unscoped upstream provider can reuse a SQLite response written by the previous release. |
