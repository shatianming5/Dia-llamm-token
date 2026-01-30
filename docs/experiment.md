# Experiment Matrix（baseline-first）

本文件用于固定**可运行**的实验矩阵（baseline-first），并作为 README 的单一指向入口。当前仓库处于 **interfaces-only（M0）**：仅保证 *命令入口 + 产物 schema + 评测 contract* 可运行且稳定；训练/推理主体仍为占位实现。

对应 claims 与证据映射见：`docs/plan.md`（M0 Claims & Evidence Map）。

---

## Baseline-first 规则（强制）

- 任意新增可运行实验前，必须先通过 baseline：
  - `python -m voxtoken.runner.smoke --out artifacts/smoke`
  - `python -m voxtoken.runner.unified_eval --in artifacts/smoke/run.json --out artifacts/eval`

---

## Summary matrix（仅包含当前可运行 EXP）

| EXP-ID | Type | Goal/Claim | Model | Config | Single-GPU | Multi-GPU | Smoke | Full | Results | Runs Path |
|---|---|---|---|---|---|---|---|---|---|---|
| EXP-0000 | baseline | CLAIM-M0-1 | schema-only | - | `python -m voxtoken.runner.smoke --out artifacts/smoke` | - | [x] | [x] | `artifacts/smoke/run.json` | `artifacts/smoke/` |
| EXP-0001 | baseline | CLAIM-M0-2 | schema-only | - | `python -m voxtoken.runner.unified_eval --in artifacts/smoke/run.json --out artifacts/eval` | - | [x] | [x] | `artifacts/eval/metrics.jsonl` | `artifacts/eval/` |


### EXP-0000 — Smoke（artifact contract）

| Field | Value |
|---|---|
| Goal/Claim | CLAIM-M0-1 |
| Baseline? | yes |
| Model name | schema-only（no model） |
| Model weights | - |
| Code paths | `voxtoken/runner/smoke.py`, `voxtoken/schemas.py` |
| Config | - |
| Dataset/Split | - |
| Metrics required | -（只验收 artifacts schema） |
| What it tests | `run.json` / `summary.json` 可生成且字段符合 `docs/results_contract.md` |
| VRAM (per GPU) | - |
| Time per epoch | - |
| Total time | seconds |
| Single-GPU command | `python -m voxtoken.runner.smoke --out artifacts/smoke` |
| Multi-GPU command | - |
| Smoke passed | [x] |
| Full passed | [x]（M0 阶段 smoke==full） |
| Smoke run path | `artifacts/smoke/` |
| Full run path | `artifacts/smoke/` |
| Results summary | `run.json` 生成，包含 `report/citations/plan/trace/issues` |
| Notes | M0 阶段仅验证“产物契约/入口命令” |


### EXP-0001 — Unified Eval（metrics contract）

| Field | Value |
|---|---|
| Goal/Claim | CLAIM-M0-2 |
| Baseline? | yes（依赖 EXP-0000 的 `run.json`） |
| Model name | schema-only（no model） |
| Model weights | - |
| Code paths | `voxtoken/runner/unified_eval.py` |
| Config | - |
| Dataset/Split | - |
| Metrics required | `case_id`, `budget_B`, `tokens_used`, `latency_ms.total`, `slot_f1`, `unsupported_rate`, `verifier_score` |
| What it tests | `metrics.json` / `metrics.jsonl` 可生成且字段稳定（契约见 `docs/results_contract.md`） |
| VRAM (per GPU) | - |
| Time per epoch | - |
| Total time | seconds |
| Single-GPU command | `python -m voxtoken.runner.unified_eval --in artifacts/smoke/run.json --out artifacts/eval` |
| Multi-GPU command | - |
| Smoke passed | [x] |
| Full passed | [x]（M0 阶段 smoke==full） |
| Smoke run path | `artifacts/eval/` |
| Full run path | `artifacts/eval/` |
| Results summary | `unsupported_rate=0.0`（空报告），其余字段为占位 0.0/0 |
| Notes | M0 阶段用于固定 metrics schema；后续实现只允许“加字段/版本化”，禁止随意删改 |

---

## Backlog（未来工作：未进入可运行矩阵）

以下为 proposal 级实验草案（尚不可运行），不应进入上方矩阵，直到：命令/配置/产物契约齐全，且 baseline-first gate 已通过。

- EXP-0100：Fixed-grid tokens baseline
- EXP-0200：Heuristic split（entropy/recon）
- EXP-0300：Learned split policy（bandit/offline RL）
- EXP-0400：No-citation / No-constrained ablation

---

## Audit checklist（当前可运行矩阵必须满足）

- 所有 EXP（上方 Summary matrix）都具备完整字段与可执行命令
- baseline-first gate 已显式存在（EXP-0000/0001）
- 产物/指标字段与 `docs/results_contract.md` 一致
