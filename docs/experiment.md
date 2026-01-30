# Experiment Matrix（baseline-first）

本文件用于固定**可运行**的实验矩阵（baseline-first），并作为 README 的单一指向入口。当前仓库已覆盖 **M0→M1**：M0 保证 *命令入口 + 产物 schema + 评测 contract*；M1 进一步提供最小可运行的推理闭环 `infer_refine`（固定网格 tokenizer + 规则化生成 + verifier gate）。

对应 claims 与证据映射见：`docs/plan.md`（M0/M1 Claims & Evidence Map）。

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
| EXP-0100 | baseline | CLAIM-M1-1/2 | fixed-grid + rules | `voxtoken/configs/inference.yaml` | `python -m voxtoken.runner.infer_refine --out artifacts/infer --budget 16; python -m voxtoken.runner.unified_eval --in artifacts/infer/run.json --out artifacts/infer_eval` | - | [x] | [x] | `artifacts/infer/run.json`, `artifacts/infer_eval/metrics.jsonl` | `artifacts/infer/` |


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


### EXP-0100 — Infer Refine（M1 minimal loop）

| Field | Value |
|---|---|
| Goal/Claim | CLAIM-M1-1/2 |
| Baseline? | yes（依赖 EXP-0000/0001 的 contract 稳定） |
| Model name | fixed-grid tokenizer + rule-based generator + verifier |
| Model weights | -（当前为规则/占位实现） |
| Code paths | `voxtoken/runner/infer_refine.py`, `voxtoken/models/*`, `voxtoken/verify/*` |
| Config | `voxtoken/configs/inference.yaml` |
| Dataset/Split | dummy volume（M1 阶段仅验证闭环与契约） |
| Metrics required | `tokens_used`, `unsupported_rate`, `verifier_score`（其余占位） |
| What it tests | Tokenize→Generate→Verify→(Refine) 的最小可运行闭环 + per-sentence citations gate |
| VRAM (per GPU) | - |
| Time per epoch | - |
| Total time | seconds |
| Single-GPU command | `python -m voxtoken.runner.infer_refine --out artifacts/infer --budget 16; python -m voxtoken.runner.unified_eval --in artifacts/infer/run.json --out artifacts/infer_eval` |
| Multi-GPU command | - |
| Smoke passed | [x] |
| Full passed | [x]（M1 阶段 smoke==full） |
| Smoke run path | `artifacts/infer/` |
| Full run path | `artifacts/infer/` |
| Results summary | `run.json` 含 `tokens_used/budget_B/verifier_score`；`metrics.jsonl` 读取并落盘 |
| Notes | 后续 M2+ 用真实数据/训练替换 dummy，同时保持 results contract 字段稳定 |

---

## Backlog（未来工作：未进入可运行矩阵）

以下为 proposal 级实验草案（尚不可运行），不应进入上方矩阵，直到：命令/配置/产物契约齐全，且 baseline-first gate 已通过。

- EXP-0200：Heuristic split（entropy/recon）
- EXP-0300：Learned split policy（bandit/offline RL）
- EXP-0400：No-citation / No-constrained ablation

---

## Audit checklist（当前可运行矩阵必须满足）

- 所有 EXP（上方 Summary matrix）都具备完整字段与可执行命令
- baseline-first gate 已显式存在（EXP-0000/0001）
- 产物/指标字段与 `docs/results_contract.md` 一致
