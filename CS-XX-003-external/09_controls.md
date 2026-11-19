# 参考实现与四个校准样例

`solution/golden/golden.patch` 从可见的起始仓库得到参考实现。`tests/calibration/` 中的 full_score、equivalent、defective、guessed 分别给出完整提交和运行结果单，另两种可接受实现及比较说明在 `solution/variants/`。每份补丁都从起始仓库单独应用。

交付前在同一环境检查 full_score 和 equivalent 全通过，guessed 错在其针对的断言，defective 的每处错误都被对应检查捕获。重复程序自检属于判分稳定性证据。正式模型结果另见 `06_baseline.md`。
