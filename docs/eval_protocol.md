# Unified Evaluation Protocol (Placeholder)

统一评测协议用于保证不同实验之间“可比”：

1. 输入：`run.json`（报告文本 + citations + plan + trace/issues）
2. 输出：`metrics.json` 与 `metrics.jsonl`
3. 字段：遵循 `docs/results_contract.md`
4. 评测原则：
   - unsupported 的定义固定（句子无 citation 或 citation 不支持 slot）
   - correctness 的 slot 抽取器固定版本（后续实现时必须版本化）
   - 时延必须真实计时（后续实现时在 trace/metrics 中落盘）

