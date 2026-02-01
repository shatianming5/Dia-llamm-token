# Dia-llamm-token / VoxToken++（接口先行 Repo Skeleton）

本仓库是 `docs/plan.md` 中 **VoxToken++：Budgeted Adaptive 3D Tokenization for Proof-Carrying CT Reporting** 的工程化落地骨架：先把**模块边界、数据结构、类/函数接口、产物契约**固定下来，再逐步填充训练/推理实现与实验矩阵。`docs/plan.md` 同时包含 proposal、实现方案（A）与实验方案（B）的最小不可分拆解。

当前状态：**M0→M1（最小推理闭环已打通）**  
已提供可运行的 `smoke` / `unified_eval`（用于固定“产物结构/指标 schema/命令入口”），以及最小可运行的 `infer_refine`（fixed-grid tokenizer + rule-based generator + verifier gate）。训练仍为占位实现（仅落盘 outputs 契约，不做真实学习）。

---

## 1. 项目概述（What/Why）

### 1.1 解决的问题

在 3D CT 报告生成中，我们把“输入体数据 → 输出报告”改写为一个**预算约束的 3D tokenization**问题：在 token 预算 `B` 下动态分配 3D tokens 的密度，并用 **token-citation + constrained decoding + verifier** 让“unsupported claim”从指标变成结构性难发生的性质。

### 1.2 核心方法（高层）

闭环：`Tokenize → Generate → Verify → Refine → Regenerate`

对应模块：

- **Tokenizer3D**：层级 3D tokens（coarse→fine）+ 显式 3D 支持域 `omega_box_mm`
- **EvidenceHead**：tokens → 结构化证据图（EvidenceNode）
- **Planner/Realizer**：证据 → 计划（ReportPlan）→ 文本 + per-sentence citations
- **Verifier**：程序化检查 missing/inconsistency/overclaim/unsupported，并给出 score
- **SplitPolicy**：在预算下决定 refine 哪些 token（learned > heuristic）

### 1.3 阶段性 Claims（M0/M1）

> 可审计的 claims 与证据映射以 `docs/plan.md` 顶部的 `C####` 为准；对应可运行实验见 `docs/experiment.md` 的 `E####`。

- **C0001**（legacy: `CLAIM-M0-1`）：baseline smoke 可运行，并生成符合 results contract 的 `run.json`/`summary.json`
- **C0002**（legacy: `CLAIM-M0-2`）：unified eval 可运行，并生成符合 results contract 的 `metrics.json`/`metrics.jsonl`
- **C0003**（legacy: `CLAIM-M1-1`）：`infer_refine` 可运行，生成非空 report 且每句都有 citation
- **C0004**（legacy: `CLAIM-M1-2`）：`unified_eval` 可读取 `infer_refine` 的 `tokens_used/verifier_score`

（Proposal-level 的研究命题/假设仍在 `docs/plan.md` 的 long-horizon 部分，当前不作为 M0 收敛判据。）

---

## 2. 目录结构说明（Project Layout）

目录树（与当前仓库一致）：

```text
.
├─ plan.md                 # moved: see docs/plan.md
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ docs/
│  ├─ experiment.md
│  ├─ plan.md
│  ├─ mohu.md
│  ├─ results_contract.md
│  ├─ eval_protocol.md
│  └─ project_index.md
└─ voxtoken/
   ├─ schemas.py              # 核心数据结构（Token/Evidence/Plan/Issue/Trace...）
   ├─ torch_compat.py         # torch 可选依赖的兼容层（无 try/except）
   ├─ configs/                # 配置占位（YAML）
   ├─ data/                   # ingest/preprocess（占位）
   ├─ models/                 # tokenizer/evidence/generator/policy（接口）
   ├─ runner/                 # train_* / infer_refine + smoke/unified_eval
   ├─ verify/                 # verifier + rules/scorer（占位）
   └─ eval/                   # metrics/counterfactuals/pareto（占位）
```

入口/结构索引（entrypoints/产物/契约一览）：`docs/project_index.md`

---

## 3. 环境安装（Environment Setup）

### 3.1 版本要求

- Python：建议 `>=3.10`（本仓库已在本地 `3.11` 环境下做过语法检查）
- GPU/CUDA：可选（真正训练/推理实现阶段需要）
- PyTorch：可选（接口层可在不安装 torch 的情况下 import）

### 3.2 安装

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt

# Optional: GPU + paper-grade tracks (torch/numpy/nibabel).
# Install a CUDA-enabled torch wheel appropriate for your system.
pip install -r requirements_gpu.txt
```

---

## 4. 数据准备（Data Preparation）

> 数据 ingest/preprocess 已可运行（repo-skeleton 口径）：用于固定“数据入口/manifest/split 规则/产物路径契约”，不等价于论文级真实训练数据流程。

### 4.1 数据来源/格式（计划）

- CT-RATE：CT volumes + reports（用于 tokenizer/报告学习）
- RadGenome-Chest CT：sentence↔mask grounding 子集（用于硬 grounding 评测）

### 4.2 路径约定（建议）

```text
data/
  raw/
    ct_rate/
    radgenome/
  processed/
    ct_rate/
    radgenome/
```

### 4.3 可运行的数据 ingest/preprocess（示例）

```bash
# 例：CT-RATE valid volumes（当前机器有 38 个可解析 .nii.gz），deterministic 70/30 train/val
python -m voxtoken.data.ingest --config voxtoken/configs/data_ingest_ct_rate_valid_all_e0920.yaml
python -m voxtoken.data.preprocess --config voxtoken/configs/data_preprocess_ct_rate_valid_all_split70_e0921.yaml

# 例：RadGenome-ChestCT（本机路径 /data/tiasha/RadGenome-ChestCT），仅做 volumes 子集 ingest + train/val split
python -m voxtoken.data.ingest --config voxtoken/configs/data_ingest_radgenome_chestct_train_e0930.yaml
python -m voxtoken.data.preprocess --config voxtoken/configs/data_preprocess_radgenome_chestct_train_e0930.yaml

# 例：从 RadGenome anatomy masks 构建 GT-box manifest（会从 train_anatomy_mask_* 里按需解压单个 mask 到 artifacts/）
python -m voxtoken.data.radgenome_mask_manifest \\
  --in artifacts/radgenome_proc_e0930/manifest.jsonl \\
  --out artifacts/radgenome_gt_lung_effusion_e0931 \\
  --config voxtoken/configs/ct_rate_ts_grounding_e0907.yaml \\
  --radgenome-root /data/tiasha/RadGenome-ChestCT \\
  --mask "lung effusion" \\
  --max-cases-train 20 --max-cases-val 5 --max-cases-test 0

# 例：同样方式构建 lung nodule（RadGenome anatomy mask 名称：\"lung nodule\"）
python -m voxtoken.data.radgenome_mask_manifest \\
  --in artifacts/radgenome_proc_e0930/manifest.jsonl \\
  --out artifacts/radgenome_gt_lung_nodule_e0933 \\
  --config voxtoken/configs/ct_rate_ts_grounding_e0907.yaml \\
  --radgenome-root /data/tiasha/RadGenome-ChestCT \\
  --mask "lung nodule" \\
  --max-cases-train 40 --max-cases-val 10 --max-cases-test 0

# 例：一键跑 RadGenome lung nodule 端到端闭环（E0933–E0935）
python -m voxtoken.runner.run_queue --queue .rd_queue/queue_e0933_0935.json --overwrite

# 例：paper-grade RadGenome lung nodule track（E0940–E0944；multi-seed + random/oracle baselines；policy 用 torchrun/DDP）
# 可通过 NPROC_PER_NODE 控制 torchrun 进程数（默认 1）
python -m voxtoken.runner.run_queue --queue .rd_queue/queue_e0940_0944_smoke.json --overwrite
python -m voxtoken.runner.run_queue --queue .rd_queue/queue_e0940_0944_full.json --overwrite

# 例：oracle-imitation policy（E0945–E0948；基于 E0941 的 nodule pseudo-GT 生成 oracle trajectories 并训练 split policy）
python -m voxtoken.runner.run_queue --queue .rd_queue/queue_e0945_0948_smoke.json --overwrite
python -m voxtoken.runner.run_queue --queue .rd_queue/queue_e0945_0948_full.json --overwrite

# 例：full-eligible nodule（E0949–E0953；去掉 E0941 的 max-cases 上限，覆盖所有可解析的 nodule masks）
python -m voxtoken.runner.run_queue --queue .rd_queue/queue_e0949_0953_smoke.json --overwrite
python -m voxtoken.runner.run_queue --queue .rd_queue/queue_e0949_0953_full.json --overwrite
```

paper-track（RadGenome lung nodule）关键产物：

- `artifacts/radgenome_gt_lung_nodule_e0941/summary.json`：GT-box manifest 构建统计（train/val counts、missing/empty masks）
- `outputs/train_policy/E0942_seed{0,1,2}/checkpoint.json` + `model.pt`：learned split policy（torch）
- `artifacts/e0943_benchmark_radgenome_lung_nodule_seed{0,1,2}/summary.json`：multi-seed benchmark 汇总
- `artifacts/paper_e0944_radgenome_lung_nodule/table1_main_ci.csv`：主表（multi-seed bootstrap CI）
- `artifacts/paper_e0948_radgenome_lung_nodule_oracle_policy/table1_main_ci.csv`：oracle-imitation policy 主表（train=400/val=100）
- `artifacts/radgenome_gt_lung_nodule_full_eligible_e0949/summary.json`：full-eligible GT-box manifest 统计（train/val counts）
- `artifacts/paper_e0953_radgenome_lung_nodule_oracle_policy_full_eligible/table1_main_ci.csv`：full-eligible oracle-imitation policy 主表（multi-seed bootstrap CI）

参考结果（RadGenome lung nodule，val n=100，full run 于 2026-01-31 产出）：

- budget_B=32：learned mean IoU=0.1128（CI≈[0.0979, 0.1270]），random=0.1133（≈[0.0987, 0.1278]），heuristic=0.1104（≈[0.0965, 0.1255]），oracle=0.1776（≈[0.1658, 0.1896]）
- 结论：learned 与 random/heuristic **几乎持平**（差异在 1e-3 量级，CI 明显重叠），但与 oracle 仍有显著 gap，说明“token split 选择策略”仍有大量可挖掘空间

参考结果（RadGenome lung nodule full-eligible，val n=126，full run 于 2026-02-01 产出）：

- budget_B=32：learned mean IoU=0.1126（CI≈[0.0987, 0.1274]），random=0.1164（≈[0.1026, 0.1307]），oracle=0.1788（≈[0.1663, 0.1909]）
- 结论：learned 仍未超过 random（差异约 3.8e-3），与 oracle 仍有显著 gap（“split policy 特征/建模”不足仍是主瓶颈）

### 4.4 split 与 seed（对齐 plan）

- patient-level split（避免同一病人泄漏到 train/test）
- 固定 seed 写入 runs/ 或 artifacts/ 元数据（避免“跑不回去”）

---

## 5. 快速开始（Quickstart）

### 5.1 一条命令跑 baseline smoke

```bash
python -m voxtoken.runner.smoke --out artifacts/smoke
```

输出产物：

- `artifacts/smoke/run.json`：最小闭环产物（report/citations/plan/trace/issues）占位
- `artifacts/smoke/summary.json`：环境与运行信息

### 5.2 一条命令跑统一 eval

```bash
python -m voxtoken.runner.unified_eval --in artifacts/smoke/run.json --out artifacts/eval
```

输出产物：

- `artifacts/eval/metrics.json`
- `artifacts/eval/metrics.jsonl`

### 5.3 一条命令跑最小推理闭环（infer_refine）

```bash
python -m voxtoken.runner.infer_refine --out artifacts/infer --budget 16
python -m voxtoken.runner.unified_eval --in artifacts/infer/run.json --out artifacts/infer_eval
```

输出产物：

- `artifacts/infer/run.json`（含 `tokens_used/budget_B/verifier_score` + report/citations/plan/trace/issues）
- `artifacts/infer_eval/metrics.jsonl`（读取并落盘 `tokens_used/verifier_score`）

---

## 6. 训练（Training）

> 训练脚本大多数仍为“repo-skeleton 口径”的最小实现：用于固定 stage 切分与产物路径。  
> 例外：paper-track 的 `voxtoken/runner/train_policy_torch.py`（E0942）会真实训练一个小型 torch policy（落盘 `model.pt`），并在 E0943 基准评测中加载对照 random/oracle。

### 6.1 单卡启动（占位）

```bash
python -m voxtoken.runner.train_tokenizer --config voxtoken/configs/train_tokenizer.yaml
python -m voxtoken.runner.train_evidence  --config voxtoken/configs/train_evidence.yaml
python -m voxtoken.runner.train_policy    --config voxtoken/configs/train_policy.yaml
```

### 6.4 CT-RATE pseudo-GT（effusions）训练闭环（repo-skeleton）

本仓库提供一个“现有数据可跑通”的训练闭环：用 TotalSegmentator 的 effusion masks 作为 pseudo-GT，先在 train split 上拟合 policy，再在 val split 上跑 grounding benchmark（fixed/heuristic/learned 三种方法对照）。

- 实验定义：见 `docs/experiment.md` 的 `E0920–E0927`
- 一键复现（按顺序）：

```bash
for e in E0920 E0921 E0922 E0923 E0924 E0925 E0926 E0927; do
  python -m voxtoken.runner.reproduce --exp "$e"
done
```

输出示例：

- train/val 划分后的 manifest：`artifacts/ct_rate_processed_valid_all_e0921/manifest.jsonl`
- policy checkpoints：`outputs/train_policy/E0926/checkpoint.json`、`outputs/train_policy/E0927/checkpoint.json`
- val benchmark summary：`artifacts/e0926_benchmark_pleural/summary.json`、`artifacts/e0927_benchmark_pericardial/summary.json`

### 6.2 多卡启动（占位）

- 计划使用 torchrun / deepspeed / accelerate（待定，后续在 `docs/experiment.md` 固化）

### 6.3 配置如何选择（configs）

- `voxtoken/configs/train_tokenizer.yaml`：Stage T（tokenizer 自监督）
- `voxtoken/configs/train_evidence.yaml`：Stage E（evidence head）
- `voxtoken/configs/train_policy.yaml`：Stage P（split policy）
- `voxtoken/configs/inference.yaml`：推理/预算/refine 轮数

---

## 7. 统一评测（Unified Evaluation）

### 7.1 评测入口命令

```bash
python -m voxtoken.runner.unified_eval --in artifacts/smoke/run.json --out artifacts/eval
```

### 7.2 指标输出文件 schema（字段说明）

详见 `docs/results_contract.md`。核心字段（当前为占位输出）：

- `verifier_score`：最终 verifier 分数
- `slot_f1`：结构化 slots 的 F1（后续由抽取器/标注对齐实现）
- `unsupported_rate`：缺少 citation 或 citation 不支持 slot 的句子占比
- `tokens_used` / `budget_B`
- `latency_ms.total`

### 7.3 如何复现同一结果

- 固定 `seed`（写入 summary/run 元数据）
- 固定 config（保存到 runs/<exp>/<run>/config.yaml）
- 统一输出 contract（metrics.jsonl 字段不随实验随意变动）

---

## 8. 实验矩阵（Experiment Matrix）

- 实验矩阵与 baseline-first 规则见 `docs/experiment.md`
- 原则：**baseline smoke + unified eval 必须先通过**，再新增 EXP

---

## 9. 结果与可视化（Results & Artifacts）

### 9.1 runs / artifacts 约定

- `artifacts/`：轻量、可缓存的推理/评测产物（默认 gitignore）
- `runs/<exp_id>/<run_id>/`：训练/评测完整产物（后续实现）

### 9.2 如何读输出

- `run.json`：推理闭环的 report/citations/plan/trace/issues
- `metrics.json` / `metrics.jsonl`：统一评测输出（字段契约见 `docs/results_contract.md`）

### 9.3 可视化图片在哪里

- 计划输出到 `runs/<exp>/<run>/plots/`（后续在 `voxtoken/eval/` 落地）

---

## 10. GPU 调度（VRAM Packing / 沾满策略）

> 目前为设计约束说明，调度器实现待补齐。

- 策略：**先把一张卡“塞满到 headroom”，再用下一张卡**
- headroom：保留显存余量避免碎片化/峰值 OOM（例如 5%–10%）
- 同卡并行：同 GPU 上并行作业需保证峰值显存不会叠加超过 headroom

---

## 11. 工程规范（Engineering Rules）

- **禁止 try/except**：用显式返回值/错误码/契约校验解决问题，不用吞异常隐藏失败
- **接口先行**：任何新增模块先写 schema + typing + 产物 contract，再写实现
- **配置统一**：训练/推理/评测均由配置驱动（configs/ + runs/ 备份）
- **结果契约优先**：metrics/trace/artifacts 字段稳定，不随实验随意改名

---

## 12. 常见问题与排障（Troubleshooting）

- OOM：优先通过降低 `budget_B`、减少 refine 轮数、降低 batch size、或按“沾满策略”重排作业；不要用 try/except 绕过
- 指标缺失：先检查 `docs/results_contract.md` 对应字段是否生成，再定位上游产物（run.json/trace）
- 分布式：确保每个 run 输出目录独立、seed 固定、配置快照落盘

---

## 13. 复现实验（Reproduce）

### 13.1 指定 E-ID 一键复现

```bash
python -m voxtoken.runner.reproduce --exp E0000
# 也兼容 legacy: EXP-0000 / 0000
```

### 13.2 批量跑队列（Queue Runner）

本仓库在 `.rd_queue/` 下提供了可复用的队列文件（例如 `queue_e0920_0927.json`）。你可以用内置 runner 一次性跑完，并把 stdout/stderr 与结果摘要落到 `.rd_queue/logs/`、`.rd_queue/results/`。

```bash
python -m voxtoken.runner.run_queue --queue .rd_queue/queue_e0920_0927.json
python -m voxtoken.runner.run_queue --queue .rd_queue/queue_papertrack_nodules_smoke.json
```

只跑某些实验 / stage：

```bash
python -m voxtoken.runner.run_queue --queue .rd_queue/queue_e0920_0927.json --only-id E0926 --only-stage smoke
```

### 13.3 如何定位 runs/<exp>/<run>

- 计划约定：`runs/<exp_id>/<run_id>/` 下包含 `config.yaml`、checkpoints、logs、metrics.jsonl、plots/

---

## 14. 贡献指南（Contributing）

推荐流程：

1. 在 `docs/experiment.md` 新增 EXP（写清命题/对照/命令/产物）
2. 新增/更新 `voxtoken/configs/` 配置
3. 保持 `docs/results_contract.md` 字段稳定（只增不删，必要时做版本化）
4. 提交 PR 前先跑：
   - `python -m voxtoken.runner.smoke --out artifacts/smoke`
   - `python -m voxtoken.runner.unified_eval --in artifacts/smoke/run.json --out artifacts/eval`
