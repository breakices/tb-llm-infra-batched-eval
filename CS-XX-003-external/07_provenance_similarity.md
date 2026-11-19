# CS-XX-003 来源、许可与公开评测相似度

## 业务来源和改造边界

上游是 EleutherAI 的 Language Model Evaluation Harness。Hugging Face 的 Open LLM Leaderboard 用该框架评测模型，并公开了固定模型 revision、批大小和任务组的运行方式。真实维护记录包括合并生成请求的分组与顺序恢复、处理不同种子对分组的影响，以及修复字节或图像参数无法生成响应缓存键的问题。[排行榜运行说明](https://huggingface.co/docs/leaderboards/en/open_llm_leaderboard/about) · [分组和重排 PR #1197](https://github.com/EleutherAI/lm-evaluation-harness/pull/1197) · [分组种子 PR #3047](https://github.com/EleutherAI/lm-evaluation-harness/pull/3047) · [缓存序列化 PR #4040](https://github.com/EleutherAI/lm-evaluation-harness/pull/4040)

本题固定上游 v0.4.13，提交 `ddd67220430a2470529f25fd5c05a576ca1057a0`，Git tree `b2c54aacf87ea3fe55e790cc9ad00cdba925833e`。对未经题包改造的 `lm_eval/` 计数，有 751 个 Python 文件、746 个非空文件。另用 Python AST 筛选实际含类或函数定义的源码，排除测试、模板目录、生成器文件及有自动生成头标的文件，再按全文件内容去重，余 372 份。该口径排除了全部 YAML 任务配置、模型权重与样例数据。[固定源树](https://github.com/EleutherAI/lm-evaluation-harness/tree/ddd67220430a2470529f25fd5c05a576ca1057a0/lm_eval)

供应方新增了离线 CPU 模型、`codesense-batch` 后端、故障起点和验收输入。这些是从真实维护机制衍生的评测改造。该实验使用合成租户场景。上游先通过 model registry 查找后端，启用 `use_cache` 时使用 `CachingLM` 包装，在评测时按 `Instance.request_type` 分发。`generate_until` 是该链路中的实际请求类型。题包的 `tests/test_harness_integration.py` 使用注册后端、原生 `Instance` 和 SQLite 响应缓存，覆盖重新打开数据库和部分结果检查点。[注册与包装](https://github.com/EleutherAI/lm-evaluation-harness/blob/ddd67220430a2470529f25fd5c05a576ca1057a0/lm_eval/evaluator.py#L239-L286) · [批请求分发](https://github.com/EleutherAI/lm-evaluation-harness/blob/ddd67220430a2470529f25fd5c05a576ca1057a0/lm_eval/evaluator.py#L582-L610)

## 许可和数据范围

上游软件的主许可证为 MIT，文件保留 EleutherAI 的版权声明。部分文件另带 Apache-2.0 头部，如 Hugging Face 来源的指标实现和 NVIDIA 的 NeMo 适配器。上游仓库还包含第三方评测任务配置、少样本示例和测试存档。例如，MATH 适配配置中含有公开问题和答案。这些资源沿用各自来源的条款；根许可证不能替外部数据集、模型和权重授权。作者仓通过固定上游版本获取源码，避免将第三方任务数据逐文件重新复制到作者仓的 Git 历史。对外分发完整导出物时应保留上游声明，并独立核对适用的数据条款。[上游 MIT](https://github.com/EleutherAI/lm-evaluation-harness/blob/ddd67220430a2470529f25fd5c05a576ca1057a0/LICENSE.md) · [指标代码声明](https://github.com/EleutherAI/lm-evaluation-harness/blob/ddd67220430a2470529f25fd5c05a576ca1057a0/lm_eval/api/metrics.py) · [NeMo 声明](https://github.com/EleutherAI/lm-evaluation-harness/blob/ddd67220430a2470529f25fd5c05a576ca1057a0/lm_eval/models/nemo_lm.py) · [上游 MATH 资源示例](https://github.com/EleutherAI/lm-evaluation-harness/blob/ddd67220430a2470529f25fd5c05a576ca1057a0/lm_eval/tasks/score/math/to_be_fixed_questions.json)

本题评分使用程序生成的请求、合成租户、确定性的数值模型和临时 SQLite 文件。测试不要求访问客户推理日志、数据集账号、外部模型 API 或模型权重。

## 已知公开重叠与盲测

最接近的公开题是 Terminal-Bench 4.1 的 `batched-eval-parity`。相同概念包括 padded/packed 两种布局、恢复输入顺序、共享前缀、停止条件与逐样本数值对照。Terminal-Bench 题在自己的 `evalbench` CLI 上读 JSONL，覆盖 few-shot 引用、分数归一化、PMI/DC-PMI 和整分片校准，并用专门的 55 秒时限验收共享前缀。`CS-XX-003` 的主要工作是维护 lm-eval 原生生成后端的作用域、模型替换、磁盘响应缓存与进程重开语义，以记录到的模型调用量验收批执行预算。旧 v3 实验含有与 TB 更相似的指标与校准概念，应保留此来源说明。[TB v4.1.0 固定题面](https://github.com/harbor-framework/terminal-bench/blob/4070d6cef56a7d20cb5dc98d5e10ada88a6b9440/tasks/batched-eval-parity/instruction.md)

本仓在 v4 之前已公开过 v3 起点、参考解与私测。即使新模型只在隔离容器里作答，仍可能预先见过公开内容。校准需要记录版本、排除作者材料和联网取答案路径，并分别报告 v4 的新答卷。公开材料不能支持对模型预训练污染作全面排除的结论。

按 TB v4.1.0 提交取出该题 environment、solution 和 tests 下的全部 30 个 Python 文件。用 Python `tokenize` 去除注释、空白和缩进标记，保留标识符、字面量及运算符，得到 21,741 个不同的连续 50-token 窗口。将其与本题起始版及参考版的四个运行时/集成文件（`llm_runtime.py`、后端注册表、`codesense_batch.py`、`lm_eval/api/model.py`），连同本题 tests、evaluator 和反例程序比对，未出现相同窗口。这个检查只覆盖指定实现的连续代码复用，不衡量机制相似程度或改写后的代码。

另按公开 SWE 数据集的 `repo` 字段或完整实例编号核对了以下快照。所列分片均未命中大小写归一化的 `EleutherAI/lm-evaluation-harness`。列举仓库名只能排查同仓实例，不能排除其他项目中存在类似缓存问题。

| 数据快照 | 核对范围 |
|---|---|
| [SWE-bench](https://huggingface.co/datasets/princeton-nlp/SWE-bench/tree/e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6) | train 19,008，dev 225，test 2,294。各分片完整仓库频数之和等于行数 |
| [SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith/tree/ea6d7173829c7ec8fa16c22055699ff2e9188091) 和 [SWE-smith-py](https://huggingface.co/datasets/SWE-bench/SWE-smith-py/tree/77cab9055d42ab4a5c25c89a8f937096db13558e) | 两个 train 分片分别为 59,136、50,908 行。完整仓库频数覆盖全部行 |
| [SWE-bench-extra](https://huggingface.co/datasets/nebius/SWE-bench-extra/tree/11dcbfb30e19552df2a2f8030bd764adc95c92a5/data) | 逐行查询 train 分片全部 6,376 个仓库字段 |
| [SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench/tree/89cdfbab4ab1bd8f5a658bb212d1b63624f4f881/data) 与 [leaderboard](https://huggingface.co/datasets/nebius/SWE-rebench-leaderboard/tree/34d5a58864acf91613740a09ec5d205228dcfa39/data) | 分别查询 test 21,336 行、filtered 6,542 行；leaderboard 的 test 与 15 个按月分片合计 1,720 行，含重复实例 |
| [SWE-rebench-V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2/tree/d7608f017a29ab256400ad4b73d5248d97204966/data) 与 [V2-PRs](https://huggingface.co/datasets/nebius/SWE-rebench-V2-PRs/tree/fbf0ecf50f268d5344149e2f0097db6bede83737/data) | train 分别为 32,079、126,300 行，按仓库或实例编号逐行核对 |
