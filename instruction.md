# 离线生成评测的批处理修复

团队在 Language Model Evaluation Harness 里用 `codesense-batch` 后端回放生成请求。它在 CPU 上运行，供上线前比较调度器与单条模型调用的行为。值班记录里还有一种情况：磁盘缓存已经预热，换实验配置并重启进程后仍出现之前的文本。切换批布局或复用同一个后端进程后，偶尔有请求拿到另一行的续写。带重复数据的评测还发起了过多模型调用。

请修改 `repo/`。使用 harness 登记的模型名调用 `generate_until`，以及直接使用已有 Python 批处理接口时，两者都要符合随仓库提供的后端协议（`docs/model_guide/codesense-batch.md`）。逐条、无缓存的模型调用是续写的数值依据。继续兼容已有请求构造方式、调用接口和结果类型；结果按输入顺序对应，请求标记和指标权重只是评测记录。调用者可以修改自己得到的结果，输入对象和后续请求仍应按原始内容执行。

线上按租户分别核算模型工作，模型替换或清理后要使用当前模型状态。开启 harness 的持久缓存后，同样的生成约定在进程重启、配置切换和部分命中时继续成立，已有其他模型后端的缓存文件仍须可读。两种批布局的计数应反映实际输入量。对支持批调用的模型，一个冷批次应至少达到每组兼容且互异的非空请求每八条一次调用的预算，尾部不足八条可使用一次调用。完整重放相同工作负载时应复用之前的结果。应在本机 CPU 上验证，不需要下载权重或调用模型服务。

同事提议把每个输入的文本作为唯一标识，复用第一次出现时的生成结果，最后按本次请求的长度截短。请判断这份方案能否满足上述要求，直接交付代码。

镜像预装了本题所用的 registry、Instance 和 SQLite 组件。可以在 `/workspace/repo` 下用真实模型接口运行一条请求：

```sh
python - <<'PY'
from lm_eval.api.instance import Instance
from lm_eval.api.registry import get_model

row = Instance("generate_until", {"text": "hello"}, ("hello", {"max_gen_toks": 3}), 0)
model = get_model("codesense-batch")(model="m1", mode="packed")
print(model.generate_until([row]))
PY
```

这次整改仅涉及 CPU 确定性生成；其他评测任务、训练权重、远端服务和第三方 GPU 后端不在验收范围内。
