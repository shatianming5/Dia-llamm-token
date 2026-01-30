# Experiment Matrix (Placeholder)

本文件用于固定实验矩阵（baseline-first），并作为 README 的单一指向入口。

> 当前仓库仍处于 interfaces-only（M0），因此矩阵先给出占位 EXP-ID 与期望产物/命题。

## Baseline-first 规则

- 任意新增实验前，必须先通过：
  - `python -m voxtoken.runner.smoke --out artifacts/smoke`
  - `python -m voxtoken.runner.unified_eval --in artifacts/smoke/run.json --out artifacts/eval`

## 实验表（占位）

| EXP-ID | 命题/对照 | 命令 | 产物 | 状态 |
|---|---|---|---|---|
| EXP-0000 | 工程冒烟：产物契约打通 | `python -m voxtoken.runner.smoke --out artifacts/smoke` | `artifacts/smoke/run.json` | done |
| EXP-0001 | 统一评测：metrics schema 固定 | `python -m voxtoken.runner.unified_eval --in artifacts/smoke/run.json --out artifacts/eval` | `artifacts/eval/metrics.jsonl` | done |
| EXP-0100 | Fixed-grid tokens baseline | TBD | `runs/EXP-0100/...` | todo |
| EXP-0200 | Heuristic split（entropy/recon） | TBD | `runs/EXP-0200/...` | todo |
| EXP-0300 | Learned split policy（bandit/offline RL） | TBD | `runs/EXP-0300/...` | todo |
| EXP-0400 | No-citation / No-constrained ablation | TBD | `runs/EXP-0400/...` | todo |

