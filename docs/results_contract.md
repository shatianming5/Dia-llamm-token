# Results Contract (Placeholder)

本文件定义统一评测与推理产物的字段契约（schema）。字段应“稳定”：允许增字段，不随意删/改名。

## 1) `run.json`（推理闭环产物）

由 `voxtoken.runner.smoke`（或未来真实推理）输出，最小字段：

```json
{
  "report": "string",
  "citations": [{"sent_id": 0, "cited_token_ids": [1, 2]}],
  "plan": {"facts": [], "impression": []},
  "trace": [],
  "issues": []
}
```

允许的扩展字段（只增不删/不随意改名）：

- `budget_B`：推理 token 预算
- `tokens_used`：实际使用 token 数
- `verifier_score`：最终 verifier 分数（与 `metrics.json(l)` 对齐）

## 2) `metrics.json` / `metrics.jsonl`（统一评测输出）

由 `voxtoken.runner.unified_eval` 输出，建议字段：

```json
{
  "case_id": "string",
  "budget_B": 1024,
  "tokens_used": 0,
  "latency_ms": {"total": 0.0},
  "slot_f1": 0.0,
  "unsupported_rate": 0.0,
  "verifier_score": 0.0
}
```
