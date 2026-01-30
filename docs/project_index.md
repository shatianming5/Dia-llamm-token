# Project Index（Repo 入口/结构索引）

本索引用于支撑 doc-driven 循环（`docs/plan.md` + `docs/mohu.md` + `docs/experiment.md`）：让任何人都能快速回答“怎么跑 baseline / eval？产物在哪？指标契约是什么？”。

当前仓库状态：**interfaces-only（M0）**。仅保证 *命令入口 + 产物 schema + 评测 contract* 可运行且稳定；训练/推理主体仍为占位实现。

---

## 1) 目录结构（Directory map）

```text
.
├─ README.md
├─ requirements.txt
├─ docs/
│  ├─ plan.md                # proposal +（M0）claims & evidence map
│  ├─ experiment.md          # baseline-first 实验矩阵（当前仅含可运行 EXP）
│  ├─ mohu.md                # Gap/Ambiguity 阻塞清单（收敛后为空）
│  ├─ results_contract.md    # run.json / metrics.json(l) 字段契约
│  ├─ eval_protocol.md       # 统一评测协议（输入/输出/原则）
│  └─ project_index.md       # 本文件
├─ voxtoken/
│  ├─ schemas.py             # 核心数据结构（Token/Evidence/Issue/Trace/Plan...）
│  ├─ torch_compat.py        # torch 可选依赖 shim（无 try/except）
│  ├─ configs/               # 训练/推理 YAML（当前占位）
│  ├─ runner/                # CLI entrypoints（smoke / unified_eval / train_* / infer_refine）
│  ├─ models/                # tokenizer/evidence/generator/policy 接口（占位）
│  ├─ verify/                # verifier/rules/scorer（占位）
│  └─ eval/                  # 指标/可视化（占位）
└─ artifacts/                # 轻量产物输出目录（默认 gitignore）
```

> 说明：`artifacts/` 默认不入库；训练级输出未来建议落到 `runs/<exp_id>/<run_id>/...`。

---

## 2) Entrypoints（怎么跑）

| What | Where | Command | Outputs |
|---|---|---|---|
| Baseline smoke（产物契约） | `voxtoken/runner/smoke.py` | `python -m voxtoken.runner.smoke --out artifacts/smoke` | `artifacts/smoke/run.json`, `artifacts/smoke/summary.json` |
| Unified eval（指标契约） | `voxtoken/runner/unified_eval.py` | `python -m voxtoken.runner.unified_eval --in artifacts/smoke/run.json --out artifacts/eval` | `artifacts/eval/metrics.json`, `artifacts/eval/metrics.jsonl` |
| Train tokenizer（占位） | `voxtoken/runner/train_tokenizer.py` | `python -m voxtoken.runner.train_tokenizer --config voxtoken/configs/train_tokenizer.yaml` | （未来：checkpoints/metrics/logs） |
| Train evidence（占位） | `voxtoken/runner/train_evidence.py` | `python -m voxtoken.runner.train_evidence --config voxtoken/configs/train_evidence.yaml` | （未来：checkpoints/metrics/logs） |
| Train policy（占位） | `voxtoken/runner/train_policy.py` | `python -m voxtoken.runner.train_policy --config voxtoken/configs/train_policy.yaml` | （未来：checkpoints/metrics/logs） |
| Inference refine（占位） | `voxtoken/runner/infer_refine.py` | `python -m voxtoken.runner.infer_refine --config voxtoken/configs/inference.yaml` | （未来：run.json/trace/metrics） |

---

## 3) Configs（source of truth）

当前 YAML 仅用于固定“接口与路径约定”，后续实现将以这些文件为配置入口：

- `voxtoken/configs/train_tokenizer.yaml`
- `voxtoken/configs/train_evidence.yaml`
- `voxtoken/configs/train_policy.yaml`
- `voxtoken/configs/inference.yaml`

---

## 4) Metrics / Artifacts contract（稳定契约）

- Results contract：`docs/results_contract.md`
  - `run.json`：报告文本 + citations + plan + trace/issues
  - `metrics.json` / `metrics.jsonl`：统一评测输出（字段允许增，不随意删/改名）
- Unified eval protocol：`docs/eval_protocol.md`

---

## 5) 快速验收（可回答三个问题）

1) **怎么跑 baseline smoke？**  
   `python -m voxtoken.runner.smoke --out artifacts/smoke`

2) **怎么跑 unified eval？**  
   `python -m voxtoken.runner.unified_eval --in artifacts/smoke/run.json --out artifacts/eval`

3) **产物在哪/长什么样？**  
   见 `docs/results_contract.md`（契约）与 `docs/experiment.md`（实验矩阵与状态）。

