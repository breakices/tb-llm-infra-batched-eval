# 题面、单测与证据映射

| 单测 | 对应业务要求 | 改动/回归点 | 可观测证据 |
|---|---|---|---|
| U1 | 兼容单请求、输入校验、输出上限、截断和停止条件 | CP1 | Compare the six-field request's complete token list with a separately computed scalar continuation. |
| U2 | 兼容单请求、输入校验、输出上限、截断和停止条件 | CP1 | Compare next_tokens for mixed numeric, text and byte prompts with independently calculated tokens. |
| U3 | 相容工作可复用，运行主体、不同配置及模型变更隔离 | CP3 | Assert changed tokens from a replacement model and inspect that model's measured prefill and generation calls. |
| U4 | 相容工作可复用，运行主体、不同配置及模型变更隔离 | CP3 | After warming and clearing twice, inspect the next generation's tokens, prefill input and model calls. |
| U5 | 相容工作可复用，运行主体、不同配置及模型变更隔离 | CP3 | Mutate the original and then retrieved list; read the stored list after each mutation. |
| U6 | 相容工作可复用，运行主体、不同配置及模型变更隔离 | CP3 | Inspect len(cache) and values after writes to bool/int lookalikes, an overwrite and a clear. |
| U7 | 兼容单请求、输入校验、输出上限、截断和停止条件 | CP1 | Assert KeyError for a later unknown model and zero prefill and forward calls on the valid first row. |
| U8 | 兼容单请求、输入校验、输出上限、截断和停止条件 | CP1 | Check the error type and zero measured work for negative, boolean, fractional and impossible budgets. |
| U9 | 兼容单请求、输入校验、输出上限、截断和停止条件 | CP1 | Inspect the exception and backend history for nonfinite, string and boolean temperatures. |
| U10 | 兼容单请求、输入校验、输出上限、截断和停止条件 | CP1 | Check each unsupported EOS id and assert that no backend was called. |
| U11 | 兼容单请求、输入校验、输出上限、截断和停止条件 | CP1 | Pass an unsupported layout; check the rejection and the absence of model calls. |
| U12 | 批次与逐条生成一致，按输入位置返回，节省模型工作 | CP2 | Compare per-row tokens from separate cold engines against independent expectations including empty prompts and real zeros. |
| U13 | 批次与逐条生成一致，按输入位置返回，节省模型工作 | CP2 | Inspect per-row continuation positions for a mixed-model batch, its permutation and a warm replay. |
| U14 | 批次与逐条生成一致，按输入位置返回，节省模型工作 | CP2 | Compare ordered outputs and row count for repeated request ids with both distinct and duplicate contents. |
| U15 | 相容工作可复用，运行主体、不同配置及模型变更隔离 | CP3 | Mutate both returned rows from a duplicate batch; compare sibling and later cache hit to the original continuation. |
| U16 | 批次与逐条生成一致，按输入位置返回，节省模型工作 | CP2 | Verify the mutable input lists after padded and packed cold/warm runs, including a real trailing zero. |
| U17 | 批次与逐条生成一致，按输入位置返回，节省模型工作 | CP2 | Record the actual backend dispatch count and its prompt rows for compatible distinct requests. |
| U18 | 批次与逐条生成一致，按输入位置返回，节省模型工作、相容工作可复用，运行主体、不同配置及模型变更隔离 | CP2+CP3 | Count actual submitted model rows and compare three scattered duplicate outputs against the scalar oracle. |
| U19 | 相容工作可复用，运行主体、不同配置及模型变更隔离 | CP3 | Warm one request; change id, weight and layout, then check unchanged tokens and zero new model calls. |
| U20 | 相容工作可复用，运行主体、不同配置及模型变更隔离 | CP3 | Compare outputs and inspect the backend calls for two logical models receiving the same prompt. |
| U21 | 相容工作可复用，运行主体、不同配置及模型变更隔离 | CP3 | Change the adapter, verify a cold forward, then replay both requests and require warm hits. |
| U22 | 相容工作可复用，运行主体、不同配置及模型变更隔离 | CP3 | Change the decoding temperature, verify a cold forward, then require hits for both temperatures. |
| U23 | 相容工作可复用，运行主体、不同配置及模型变更隔离 | CP3 | Observe two cold executions and later warm hits for distinct tenant strings and for integer/boolean tenant identities. |
| U24 | 相容工作可复用，运行主体、不同配置及模型变更隔离 | CP3 | Compare backend prefill work on identical prefixes across tenants and a partial prefix within the original tenant. |
| U25 | 相容工作可复用，运行主体、不同配置及模型变更隔离 | CP3 | Submit bool/int lookalike prompts sequentially and observe a cold backend call per type followed by warm hits. |
| U26 | 相容工作可复用，运行主体、不同配置及模型变更隔离 | CP3 | Compare short and long outputs against oracle in both insertion orders to detect stale truncation. |
| U27 | 兼容单请求、输入校验、输出上限、截断和停止条件 | CP1 | Assert empty output and inspect both generation and prefill counts for zero tokens. |
| U28 | 批次与逐条生成一致，按输入位置返回，节省模型工作 | CP2 | Check empty result lists and zero measured backend work for both layouts. |
| U29 | 兼容单请求、输入校验、输出上限、截断和停止条件 | CP1 | Assert the exact short continuation containing eligible EOS and the full unstopped result. |
| U30 | 兼容单请求、输入校验、输出上限、截断和停止条件 | CP1 | Assert the exact full continuation when EOS occurred before the minimum length. |
| U31 | 两种布局按实际输入计量 | CP5 | Check cumulative rectangular input work for two calls including duplicates and cache hits, and unchanged packed count. |
| U32 | 两种布局按实际输入计量 | CP5 | Compare cumulative real-token count after right truncation, duplicates and replay, and unchanged padded count. |
| U33 | 兼容单请求、输入校验、输出上限、截断和停止条件 | CP1 | Compare tokens for limits zero, one, two, beyond prompt, and unlimited in both layouts. |
| U34 | 兼容单请求、输入校验、输出上限、截断和停止条件、批次与逐条生成一致，按输入位置返回，节省模型工作 | CP1+CP2 | Compare text and raw byte inputs with independent uncached arithmetic in both layouts. |
| U35 | 兼容单请求、输入校验、输出上限、截断和停止条件、批次与逐条生成一致，按输入位置返回，节省模型工作 | CP1+CP2 | Record the truncated input delivered to a scalar-only backend and compare its resulting tokens. |
| U36 | 相容工作可复用，运行主体、不同配置及模型变更隔离 | CP3 | Inspect cold work for independent rollout seeds and shards, followed by hits within those namespaces. |
| U37 | 兼容单请求、输入校验、输出上限、截断和停止条件 | CP1 | Compare output tokens retained before a UTF-8 delimiter crossing token boundaries. |
| U38 | 兼容单请求、输入校验、输出上限、截断和停止条件 | CP1 | Check tokens retained from an early delimiter and trim only the later eligible marker. |
| U39 | 批次与逐条生成一致，按输入位置返回，节省模型工作、相容工作可复用，运行主体、不同配置及模型变更隔离 | CP2+CP3 | Compare the observed packed batch output at each seeded row to an independently evaluated scalar reference. |
| U40 | 批次与逐条生成一致，按输入位置返回，节省模型工作、相容工作可复用，运行主体、不同配置及模型变更隔离 | CP2+CP3 | Compare the observed padded batch output at each seeded row to an independently evaluated scalar reference. |
| U41 | 批次与逐条生成一致，按输入位置返回，节省模型工作、相容工作可复用，运行主体、不同配置及模型变更隔离 | CP2+CP3 | Compare cold generation rows and prefilled tokens by model/adapter/temperature to typed per-tenant scopes, bound actual dispatches, and verify zero new work on reordered cross-layout replay. |
| U42 | 注册模型的 generate_until 输入、续写文本与位置契约 | CP4 | Resolve the native registry entry, instantiate through the upstream factory, and compare returned strings in both layouts to independently rendered UTF-8 tokens. |
| U43 | 注册模型的 generate_until 输入、续写文本与位置契约 | CP4 | Give native Instance rows competing scalar/list stop strings and compare the rendered prefix ending before the earliest match. |
| U44 | 进程重启、配置切换、部分结果和旧模型磁盘缓存兼容 | CP6 | Reopen SQLite in two fresh Python processes with different hash seeds; observe output order and the actual cold prompt sent to the second backend. |
| U45 | 进程重启、配置切换、部分结果和旧模型磁盘缓存兼容 | CP6 | Reopen a shared SQLite file across configuration scopes, then replace models in separate engine lifetimes at the same registration generation; verify each actual continuation and warm hit. |
| U46 | 进程重启、配置切换、部分结果和旧模型磁盘缓存兼容 | CP6 | Persist through upstream CacheHook, observe independent work under another tenant and a disk-only hit for the writer. |
| U47 | 进程重启、配置切换、部分结果和旧模型磁盘缓存兼容 | CP6 | Read a pre-populated unscoped v0.4.13 disk key from the native CachingLM with the standard dummy provider. |
