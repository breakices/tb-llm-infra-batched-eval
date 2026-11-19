# full_score control

The reference scopes in-memory requests and durable responses, including partial checkpoints, and restores native request order before returning outputs.

The verifier passed 47/47 named methods and 429/429 collected cases. Reward 1.

The 3 runs used a separate offline Linux container each time. Complete stderr, stdout, reward, probe and per-case JSON are in `../assets/controls/full_score-{1,2,3}.json.gz`.

| Method | Observed outcome | Evidence |
|---|---|---|
| U1 `test_positional_request_generation` | passed | all assertions held |
| U2 `test_toy_model_token_formula` | passed | all assertions held |
| U3 `test_register_replacement_invalidates_old_cache` | passed | all assertions held |
| U4 `test_clear_discards_warm_state` | passed | all assertions held |
| U5 `test_kv_cache_detaches_lists_on_write_and_read` | passed | all assertions held |
| U6 `test_kv_cache_preserves_distinct_typed_keys` | passed | all assertions held |
| U7 `test_unknown_model_rejected_before_batch_work` | passed | all assertions held |
| U8 `test_invalid_generation_limit_rejected_before_batch_work` | passed | all assertions held |
| U9 `test_invalid_temperature_rejected_before_batch_work` | passed | all assertions held |
| U10 `test_invalid_eos_rejected_before_batch_work` | passed | all assertions held |
| U11 `test_invalid_mode_rejected_before_batch_work` | passed | all assertions held |
| U12 `test_cold_layouts_match_scalar_oracle` | passed | all assertions held |
| U13 `test_internal_grouping_restores_request_order` | passed | all assertions held |
| U14 `test_duplicate_ids_and_rows_preserve_all_positions` | passed | all assertions held |
| U15 `test_returned_duplicate_rows_and_cache_are_detached` | passed | all assertions held |
| U16 `test_prompt_lists_survive_padding_and_cache_hits` | passed | all assertions held |
| U17 `test_compatible_rows_use_one_batched_forward` | passed | all assertions held |
| U18 `test_duplicate_semantics_submit_one_backend_row` | passed | all assertions held |
| U19 `test_request_metadata_and_layout_reuse_sequence` | passed | all assertions held |
| U20 `test_model_context_uses_its_own_backend` | passed | all assertions held |
| U21 `test_adapter_context_uses_its_own_state` | passed | all assertions held |
| U22 `test_temperature_context_uses_its_own_state` | passed | all assertions held |
| U23 `test_tenant_sequence_cache_isolated_by_value_and_type` | passed | all assertions held |
| U24 `test_different_tenants_have_independent_cold_prefix_work` | passed | all assertions held |
| U25 `test_identical_prompts_with_distinct_types_keep_their_own_state` | passed | all assertions held |
| U26 `test_continuation_length_isolated_on_warm_cache` | passed | all assertions held |
| U27 `test_zero_length_request_does_no_model_work` | passed | all assertions held |
| U28 `test_empty_batch_has_no_model_work` | passed | all assertions held |
| U29 `test_eos_included_at_first_eligible_position` | passed | all assertions held |
| U30 `test_early_eos_kept_and_ignored_until_minimum` | passed | all assertions held |
| U31 `test_padded_work_counts_every_submitted_row` | passed | all assertions held |
| U32 `test_packed_work_counts_each_real_effective_token_once` | passed | all assertions held |
| U33 `test_effective_context_is_right_truncated` | passed | all assertions held |
| U34 `test_unicode_and_bytes_prompts_match_the_uncached_backend` | passed | all assertions held |
| U35 `test_scalar_only_backend_receives_the_effective_prompt` | passed | all assertions held |
| U36 `test_rollout_seed_and_shard_isolate_sequence_state` | passed | all assertions held |
| U37 `test_utf8_stop_markers_span_output_tokens` | passed | all assertions held |
| U38 `test_stop_marker_before_minimum_is_retained` | passed | all assertions held |
| U39 `test_packed_cold_192` | passed | all assertions held |
| U40 `test_padded_cold_192` | passed | all assertions held |
| U41 `test_replay_work_budget_192` | passed | all assertions held |
| U42 `test_harness_registry_renders_instances_in_order` | passed | all assertions held |
| U43 `test_harness_stops_at_first_text_match` | passed | all assertions held |
| U44 `test_harness_response_cache_reopens_and_scatters_hits` | passed | all assertions held |
| U45 `test_harness_response_cache_isolates_serving_scopes` | passed | all assertions held |
| U46 `test_harness_partial_checkpoint_uses_response_cache_scope` | passed | all assertions held |
| U47 `test_harness_legacy_response_keys_remain_readable` | passed | all assertions held |
