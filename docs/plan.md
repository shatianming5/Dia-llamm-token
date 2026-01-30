---

# Repo Plan（M0）& Proposal（Long-horizon）

本仓库当前阶段：**interfaces-only（M0）**。为了让 doc-driven 循环可终止、可审计，本文件在顶部新增 **M0 Claims & Evidence Map**：只有该小节中的 claims 会被视为“必须证明”的工程承诺；其余大段内容是 long-horizon proposal / 设计笔记（为后续研究与实现服务），**不作为当前收敛的阻塞项**。

## M0 Claims & Evidence Map（收敛判据）

| Claim ID | Claim（可验证陈述） | Evidence（在哪看） | Verify（命令） | Status |
|---|---|---|---|---|
| CLAIM-M0-1 | baseline smoke 可运行，并生成符合 results contract 的 `run.json`/`summary.json` | `docs/experiment.md` → EXP-0000；产物：`artifacts/smoke/` | `python -m voxtoken.runner.smoke --out artifacts/smoke` | PROVED |
| CLAIM-M0-2 | unified eval 可运行，并生成符合 results contract 的 `metrics.json`/`metrics.jsonl` | `docs/experiment.md` → EXP-0001；产物：`artifacts/eval/` | `python -m voxtoken.runner.unified_eval --in artifacts/smoke/run.json --out artifacts/eval` | PROVED |

## M0 Contracts（契约/协议）

- Results contract：`docs/results_contract.md`
- Unified evaluation protocol：`docs/eval_protocol.md`
- Experiment matrix（baseline-first）：`docs/experiment.md`
- Project index（入口/结构）：`docs/project_index.md`

## Change Log（before/after + rationale）

- 2026-01-30：新增 M0 Claims & Evidence Map；补齐 `docs/project_index.md`；将 `docs/experiment.md` 从占位表修复为可审计矩阵；同步 README（对齐 claims/矩阵/入口命令）。

---

本文档给出一份**按 NeurIPS Oral 的“硬标准”反推设计**的完整 proposal（以 **3D tokenization** 为主线），目标是将 reviewer 常见质疑点——**novelty、heuristic、grounding 真实性、active 是否真的 active、指标是否硬、因果性是否成立**——转化为论文的“不可替代贡献”。

本文档不对“中稿概率”作承诺，但结构将**按“面向 Oral 级投稿所需证据链”**组织：按该证据链实施，可显著降低因“像拼装系统”而被一票否决的风险。

---

# VoxToken++：Budgeted Adaptive 3D Tokenization for Proof-Carrying CT Reporting

## 0. TL;DR（投 Oral 的一句话）

我们把 3D CT 报告生成改写为一个**预算约束的 3D tokenization 问题**：

> 在 token 预算 (B) 下，通过**可学习的层级 3D token 分配（octree / hierarchical VQ tokens）**主动细化信息密度，并用**token-citation + 受限解码（constrained decoding）**生成“携带证据”的报告，使 **unsupported claim 机制性趋近 0**，同时在 **correctness–latency Pareto** 上显著优于固定 tokenization / 切片 / ROI 裁块方法。

---

## 1. 研究问题（Problem Statement）

给定 3D CT 体数据 (V) 与 token 预算 (B)（约束计算/显存/时延），学习一个 tokenization 函数与生成器，使得：

* **Tokenize**：输出可变长度 token 集合
  [
  T = {(t_i, \Omega_i)}_{i=1}^{|T|},\quad |T|\le B
  ]
  其中 (t_i) 是 token 表示（离散 id 或连续 embedding），(\Omega_i\subset \mathbb{R}^3) 是其**明确的 3D 支持域（box/voxel set）**；
* **Generate**：生成报告 (y)，并且每条关键医学断言都必须附带 token 引用集合 (C_k\subseteq {1,\dots,|T|})（token-citation）；
* **Guarantee**：报告中的结构化临床事实（finding/side/location/size/negation/uncertainty）必须从 token 证据中“可回溯”，并能用 verifier 程序化检查。

---

## 2. 核心贡献（Contributions，按 Oral 口味写成 3 条“不可替代”）

### C1. **Learned Budgeted Adaptive 3D Tokenization（不是 heuristic crop）**

提出 **VoxToken++**：一个**层级离散 3D tokenizer**（octree + VQ/RVQ），在推理时以 **split 决策**动态分配 token 密度，实现可变长度 token 序列，显式优化 correctness–cost。

### C2. **Proof-Carrying Generation via Token-Citation + Constrained Decoding（机制性消灭 unsupported）**

提出 **token-citation 受限生成**：关键事实的词元只能从证据图中取值，并强制输出引用 token 集合；由此使 unsupported claim 从“评测指标”变成“结构性很难发生”的性质（并提供可验证的 gate）。

### C3. **Verifier-as-Metric-and-Reward（把自检变成算法，而不是 prompt 技巧）**

设计一个可复现 verifier：输出可枚举 issues（missing-slot / inconsistency / overclaim / unsupported），既作为主评测指标，也作为 split policy 的训练 reward，实现 tokenize→generate→verify→refine 的闭环学习。

---

## 3. 方法（Method）：最小不可分闭环 + 每一步都能“被评审承认是方法”

### 3.1 组件总览

* **Tokenizer** ( \mathcal{T}_\phi )：层级离散 tokens（coarse→fine），每个 token 带 3D 支持域 (\Omega)
* **Split Policy** ( \pi_\psi )：决定对哪些 token 做 split（细化）
* **Evidence Head** ( \mathcal{E}_\eta )：从 tokens 提取结构化证据（finding/属性）+（关键子集）mask/measurement
* **Generator** ( \mathcal{G}_\theta )：基于证据图受限生成 + token-citation
* **Verifier** ( \mathcal{V} )：程序化验证与计分（也是 reward）

最终推理：**Tokenize → Generate → Verify → Refine → Regenerate**，直到边际收益不足或预算耗尽。

---

## 3.2 Tokenizer：层级离散 3D tokens（真正的“tokenize”）

### Token 表示

我们采用 **Residual VQ（RVQ）/ VQ-MAE** 风格的 3D tokenizer：

* 3D encoder (f_\phi) 将体数据编码为多尺度特征金字塔：({F^{(0)},F^{(1)},...,F^{(L)}})
* 每层通过 codebook 量化得到离散 token id：
  [
  t^{(\ell)}*{u} = \mathrm{VQ}(F^{(\ell)}*{u}) \in {1,\dots,K_\ell}
  ]
* 每个 token 对应一个明确支持域 (\Omega^{(\ell)}_u)（一个 3D box 或 voxel block）

### 层级（Octree）结构

* Level-0：粗网格 tokens 覆盖全体积（低成本、全局上下文）
* Level-1..L：仅对被选择 split 的节点生成更细 tokens（局部高信息密度）

> **关键点**：本方法不是“固定 grid tokens”，而是“层级 tokens + 可学习 split”。这就是 Oral 级 novelty 的落脚点。

---

## 3.3 Split 决策：显式目标函数 + 可学习策略（回应“heuristic”质疑）

我们把“细化哪些 token”形式化为一个预算约束优化：

### 目标函数（核心）

在第 (k) 次 refinement，我们选择一组 split 操作 (A_k)，最大化**期望验证收益**与**证据覆盖**，并惩罚 token 成本：

[
\max_{A_k} \ \mathbb{E}\big[\Delta \mathcal{V}(y_{k+1},T_{k+1}) \mid T_k, A_k\big] - \lambda \cdot \Delta|T|
]

其中 (\Delta \mathcal{V}) 来自 verifier 的分数提升（见 3.6），(\Delta|T|) 是 split 增加的 token 数。

### 可训练 Split Policy（主线，面向 Oral 级投稿必备）

定义每个候选 token 的状态特征 (x_i)：

* token-level **uncertainty**（证据头输出熵）
* token-level **reconstruction error**（VQ/MAE 残差）
* token-level **citation pressure**（生成时 attention/citation 频次）
* anatomy prior（若有器官 mask，可作为 token 的器官分布）
* history（该 token 是否已 split、split 深度）

策略：
[
\pi_\psi(\text{split}\mid x_i) \in [0,1]
]

训练方式（回应“RL 不稳定”的常见质疑）：

* 先用启发式/teacher 产生 logged data（entropy/recon）
* 再做 **contextual bandit / offline RL**：奖励 = verifier 分数提升 − λ token 成本
* 最终展示：**learned policy > heuristic**（这条是 Oral 关键）

---

## 3.4 Evidence Graph：token → 结构化证据（让引用不是装饰）

我们构建证据图 (G)：

* 节点：token 或 token 组（含 (\Omega)、embedding/id）
* 节点属性：结构化预测

  * finding（present/absent/uncertain）
  * laterality、location、size bins、negation/uncertainty
* 关键 finding 子集：输出局部 mask/weak box，并计算 measurement（diameter/extent/volume）

**为什么这一步是“不可分”？**
因为它把 token-citation 变成“引用证据节点”，而不是“attention 可视化”。如果没有结构化证据，citation 很容易被 reviewer 说成“装饰”。

---

## 3.5 受限生成：Token-citation + Constrained Decoding（机制性 unsupported≈0）

报告生成分两步：

### Step 1：生成结构化报告计划（Plan）

从证据图中抽取一个 report plan（器官→findings→impression），每条事实是一个 slot tuple：
[
(\text{finding_type}, \text{side}, \text{location}, \text{size}, \text{certainty})
]
并附带支持 token 集合 (C_k)。

### Step 2：自然语言表述（Surface Realization）

用 LLM 只负责把 slot tuple 转写成自然语言，但关键 token（finding/side/location/size/certainty）**必须来自 slot**。
每句输出格式：

* sentence text
* cited token ids：`[tok:12, tok:57,…]`

**Constrained decoding**：如果模型试图生成不在 slot 中的 finding/侧别/大小，解码直接禁止或替换为不确定模板。

> 这能把 unsupported claim 从“需要 critic 才能发现”变成“生成机制阻止它出现”。

---

## 3.6 Verifier：指标也是 reward（让“反思”成为算法）

Verifier 输出 issues（程序化可复现）：

* missing-slot：出现 finding 但缺 side/location/size
* inconsistency：左右、否定、数量冲突
* overclaim：证据不确定却用确定语气
* unsupported：无 citation 或 citation token 不包含对应 slot 证据

并输出总分：
[
\mathcal{V} = \text{ClinicalCorrectness} - \alpha #missing - \beta #inconsistency - \gamma #overclaim - \delta #unsupported
]

**训练 split policy** 的 reward：
[
R = \Delta \mathcal{V} - \lambda \Delta|T|
]

---

## 3.7 推理算法（Oral 级必须能写成 Algorithm 1）

**Algorithm: VoxToken++ Inference**

1. 初始化 coarse tokens (T_0)（Level-0）
2. 生成 (y_0)（带 citations）
3. verifier 得到 issues 与 (\mathcal{V}(y_0,T_0))
4. for k = 0..K-1:

   * 根据 (\pi_\psi) 选择要 split 的 token 集合 (A_k)
   * 得到 refined tokens (T_{k+1})
   * 生成 (y_{k+1})，验证得分 (\mathcal{V}(y_{k+1},T_{k+1}))
   * 若边际收益 ( \Delta\mathcal{V} / \Delta|T| < \tau) 或预算耗尽，stop
5. 输出 final report + citations + token supports + issues（应接近 0）

---

# 4. 理论/可解释性（Oral 常见加分点：哪怕弱也要有）

不必做大理论，但至少给出一个**可写进 paper 的“合理性保证”**：

1. **边际收益递减假设**：refinement 使得证据不确定性下降通常呈递减（coarse→fine），因此 (\Delta\mathcal{V}) 在 token 数上呈 diminishing returns（经验上可验证）。
2. **Stop rule 的合理性**：当 (\Delta\mathcal{V} / \Delta|T| < \tau) 时停止，等价于在 correctness–cost Pareto 上选择膝点。
3. **Unsupported 的结构性上界**：在 constrained decoding + verifier gate 下，unsupported claim rate 被强约束到接近 0（可作为“系统性质”陈述）。

---

# 5. 实验设计（按 Oral 反推：必须给出的图/表）

## 5.1 主指标（拒绝只卷 ROUGE）

主结果必须围绕四件事：

1. **Clinical correctness**：结构化事实 F1（finding + side + location + size bins + negation/uncertainty）
2. **Grounding**：sentence→token→GT region 命中（hit-rate / IoU / Dice）
3. **Unsupported claim rate**：应显著低于 baselines（最好接近 0）
4. **Efficiency**：#tokens、latency（wall-clock）
   并给出 **Pareto 曲线**：correctness vs #tokens / latency。

## 5.2 Baselines（防“没对齐 baseline”）

最小不可分 baseline 列表（必须全做）：

* **Fixed-grid 3D tokens**（同 encoder、同生成器，只禁用 adaptive split）
* **2D slice tokens**（uniform slice sampling / 2.5D）
* **ROI crop pipeline**（经典 coarse-to-fine/region-guided）
* **Heuristic split**（entropy / recon error）vs **learned split policy（本文方法）**
* **No-citation**（去掉 constrained/citation）
* **No-refine**（只做一次 tokenization）
* **Citation=attention top-k**（弱基线）vs **证据图 constrained citation（本文方法）**（证明不是可视化）

## 5.3 反事实/因果实验（Oral 级关键）

这是面向 Oral 级投稿的关键证据链，必须写进 proposal：

* **Permutation test**：随机打乱 token 支持域 (\Omega_i)（保持 token embedding 不变）→ grounding 与 correctness 显著下降
* **Citation swap**：交换 cited token ids（保持句子不变）→ verifier 立即报 unsupported
* **Mask-level sanity**：对有 GT mask 的 finding，细化 tokens 应更集中覆盖 lesion（统计 token 密度 vs lesion overlap）

这些实验是用来正面击穿“citation 只是 attention 装饰”的质疑。

## 5.4 关键可视化（Oral 需要的 3 张主图）

* **Fig1**：VoxToken++ 框图（层级 tokens + refine loop + constrained generation）
* **Fig2**：Pareto 曲线（相对于 fixed-grid & ROI crop 形成优势/dominance）
* **Fig3**：Grounding 可视化 + 反事实实验结果（强说服力）

---

# 6. 数据与落地策略（保证可实现“硬 grounding”）

需要确保至少一部分数据具备 **region-level ground truth**，否则 Oral 很难。

最小策略：

1. 主数据集：大规模 CT + 报告（用于 token→text 学习）
2. Grounding 子集：带 lesion mask/box 的 CT 子集（哪怕规模小）用于 sentence→token→region 的硬评测与训练（关键 finding 头）

如果现成数据 grounding 不够：

* 可构建一个**小规模标注子集**（几十到几百例），只标 2–3 个高价值 finding（结节/积液/气胸），但要做到高质量。Oral 认可“少而硬”。

---

# 7. 里程碑（保证是“最小不可分、可执行”的计划）

**M0（2周）**：固定 grid tokens + token-citation + verifier gate 跑通（unsupported 明显下降）
**M1（4周）**：实现 octree adaptive split（先 heuristic），做 Pareto 曲线
**M2（6–8周）**：训练 learned split policy，证明 > heuristic（关键）
**M3（8–10周）**：grounding 子集 + 反事实实验（关键）
**M4（10–12周）**：跨数据集/外部验证（锦上添花但对 Oral 很重要）

---

# 8. 风险与对策（提前写给 reviewer 看）

1. **citation 漂移** → 用证据图 slot 约束 + permutation/counterfactual 证明因果
2. **token 太粗导致伪 grounding** → 强制关键 finding 使用 mask-head；refine 优先 lesion-prone 区域
3. **RL 不稳定** → 先 offline bandit（logged data），再小步 fine-tune；主结果先用 learned policy>heuristic
4. **过度保守（只写 uncertain）** → verifier 同时惩罚 recall/漏报；加入 coverage reward
5. **计算太重** → 以 token budget 为第一约束；报告 wall-clock 与吞吐

---

# 9. 核心卖点（面向 Oral 级投稿；可写在 Introduction 的最后一段）

* 我们不是在做“更强的报告生成器”，而是在做**可解释、可验证、预算可控的 3D tokenization**。
* 通过 **learned adaptive tokens**，我们实现 **同等正确性更低 token/时延** 或 **同等预算更高正确性** 的 Pareto 优势。
* 通过 **token-citation + constrained decoding**，我们把 unsupported claim 从“难测的主观问题”变成“可被机制性约束的系统性质”。
* 通过 **verifier-as-reward**，我们把“慢思考/反思”落到可训练的决策闭环。

---

## 下一步优先事项（两件；决定是否具备 Oral 级证据链）

1. **把 split 决策写成一个可以实现的目标函数 + learned policy**（这是 novelty 的锚）
2. **准备 grounding 子集 + 做 3 个反事实实验**（这是说服力的锚）

---

可选补充（非主线）：

* **NeurIPS 论文大纲（逐段落写什么）**
* **Algorithm 1/2 的伪代码（tokenize/refine/constrained decode/verifier）**
* **主实验表格模板（每一列对应 reviewer 质疑点）**

实现版本选择：更倾向 **离散 VQ tokens 主线**，或先用 **连续 tokens MVP** 并写成“可插拔 VQ”的版本。

下面给出两套内容，**都按“最小不可分（atomic）”**来写： 

* **A. 实现方案（Implementation Plan）**：每个模块的**接口、落地实现、训练/推理闭环、产物**一次性写全。
* **B. 实验方案（Experiment Plan）**：按 NeurIPS Oral 反推的**主指标、baseline、消融、反事实因果实验、图表清单**一次性写全。

本文档将把“能跑通的最小 MVP”与“面向 Oral 级投稿的必备增强”写成同一条链路，确保每一步都有**验收检查（acceptance check）**，避免出现“做了很多但无法证明”的情况。

---

# A. 实现方案（最小不可分，端到端闭环）

## A0. 统一协议与产物（先固定，避免后续接口/评测不一致）

### A0.1 统一坐标与支持域

* 所有 token 必须带**空间支持域** (\Omega_i)：推荐用**原始高分辨率体坐标系的 3D box**（`x0,x1,y0,y1,z0,z1`），必要时再附 voxel mask（仅关键 finding）。
* 必须提供 `low↔high` 映射（若有重采样），保证 token box 可回溯到原 CT。

### A0.2 统一 Schema（JSON 可序列化）

最小对象集合（每个都必须落盘）：

* `Token`: `{token_id, level, code, embedding_ref, omega_box, parent_id, children_ids}`
* `Citation`: `{sent_id, cited_token_ids:[...]}`
* `EvidenceNode`: `{eid, type, attrs{side,location,size,certainty,negation}, supported_token_ids:[...], optional_mask_ref}`
* `Issue`: `{type, span, reason, related_eids/tokens}`
* `Trace`: 每轮 refine 的决策与耗时：`{k, B_used, split_tokens, added_tokens, verifier_before/after, latency}`

### A0.3 端到端输出三件套（不可少）

* `final_report.txt`（含 token-citation）
* `evidence_graph.json`
* `trace.jsonl`（每轮一条）

**验收检查**：任意 case 必须能生成这三件套，即使模型很弱也要跑通。

---

## A1. 数据管线（Ingest & Preprocess）

### A1.1 输入与规范化（MVP）

* 读入 DICOM/NIfTI → HU（CT）→ 重采样到 `target_spacing`（如 1–2mm，按任务定）
* windowing：至少两路（骨窗/软组织窗或肺窗/纵隔窗），拼成 `C=2` 通道
* 归一化：clamp + z-score/min-max
* 输出：`V ∈ R^{C×D×H×W}`

### A1.2 Grounding 子集（硬要求）

为了 Oral，你必须有一部分带 **region GT**（mask/box）。最小策略：

* 主数据：CT + report（规模大，用于 token→text）
* 子集：2–3 个高价值 finding 的 lesion mask/box（规模可以小，但要硬）

**验收检查**：子集里每个样本必须有 `{mask 或 box}`，能计算 IoU/Dice/hit-rate。

---

## A2. Tokenizer（层级离散 3D tokens：VQ/RVQ + octree）

你要的“tokenize”必须是**离散 tokens**，否则 reviewer 会说你只是 patch embedding。

### A2.1 模型结构（最小可跑 + 具备层级）

* 3D Encoder (f_\phi)：输出多尺度特征金字塔 ({F^{(0)},...,F^{(L)}})

  * 实现：3D Conv / 3D Swin / UNETR encoder 都行，但要能输出多尺度。
* 每层 quantizer：RVQ / VQ（每层一个 codebook 或共享 codebook）

  * 输出 token id：(t_u^{(\ell)} = \text{VQ}(F_u^{(\ell)}))

### A2.2 支持域 (\Omega) 的实现（关键）

* Level-(\ell) 的 token 网格大小是 ((D_\ell,H_\ell,W_\ell))
* token (u=(z,y,x)) 的支持域 box 由 stride 与 spacing 决定：

  * `omega = grid_cell_to_box(u, level, spacing, origin)`
* 存 `parent_id`：(\ell) → (\ell+1) 8 个子格（octree）

### A2.3 两种实现路径（推荐先做“选择式”，再做“按需式”）

**路径 A（最省工程，先冲主结果）— 选择式层级 tokenization**

* 一次 forward 得到所有层 tokens（coarse+fine）
* refine 时不重新编码，只是**从更细层“挑更多 tokens”**（split=选择子 token）
* 成本主要体现在生成器 cross-attn token 数增加（#tokens/latency）

**路径 B（更强但更难）— 按需式 tokenization**

* 初始只算 Level-0
* split 某 token 时，对该 (\Omega) 区域 crop 高分块 → 编码 → quantize 得到子 tokens
* 优点：encoder 计算也随预算增长，Pareto 更漂亮（更像“预算化计算”）

**验收检查**：路径 A 先跑通即可；路径 B 作为 Oral 强化项。

### A2.4 Tokenizer 训练（Stage-1：自监督）

* 目标：VQ-MAE / reconstruction

  * loss = 重建 loss + commitment loss + codebook usage 正则（避免 codebook collapse）
* 输出：每层 codebook 使用率、重建误差分布（作为后续 split feature）

**验收检查**：codebook perplexity 不塌；重建误差能区分复杂区域/简单区域（用于 split 特征）。

---

## A3. Evidence Head（token → 结构化证据 +（关键子集）mask/measurement）

这是让 citation “不是装饰”的关键模块。

### A3.1 输入输出

**Input**：选中的 tokens（含 level、embedding、(\Omega)）
**Output**：Evidence nodes（结构化 slots）+ optional mask/measurement（只对关键 finding）

### A3.2 结构（最小可跑）

* 对每个 token embedding 做 multi-head：

  * finding（present/absent/uncertain 或概率）
  * side / location（可先粗粒度，如 left/right/unknown）
  * size bins（先离散 bins，稳定）
  * negation/uncertainty（可并入 certainty）
* 对关键 finding 子集（例如结节/积液/气胸）：

  * **mask head**：在 (\Omega) crop 内做轻量 3D U-Net 输出 mask（或弱 box）
  * measurement：diameter/extent/volume

### A3.3 Evidence Graph 构建（merge/prune）

* merge：同一 finding、空间重叠的 token 合并为一个 evidence node
* prune：每类最多保留 N 条（按 certainty + 是否有 mask 优先）

**验收检查**：Graph 节点数受控；每个 evidence node 都能回溯到 token ids 与 (\Omega)。

---

## A4. Generator（Proof-carrying：token-citation + constrained decoding）

把 unsupported 从“指标”变成“机制性难发生”。

### A4.1 两阶段生成（最小可跑且可控）

**Stage-1：Plan 构建（结构化）**

* 输入：Evidence Graph
* 输出：slot tuples 列表（每条事实绑定 supporting token ids）

  * 最小实现：**非 LLM**（规则/排序）

    * 按器官分组 → 每组取 top evidence → 形成 findings + impression
  * Oral 强化：用小模型/LLM 生成 plan，但仍要 verifier 校验

**Stage-2：Surface Realization（表述）**

* 最小实现：**模板+fill**（保证约束）
* 强化：LLM 文本润色，但关键字段来自 slot，不允许自由编造

### A4.2 Constrained decoding（硬约束）

* finding/side/location/size/certainty 的 token 只能从 plan 的离散集合里取
* 每句必须输出 `cited_token_ids`

**验收检查**：生成文本中所有关键句都有 citation；去掉 citation 直接判 fail。

---

## A5. Verifier（指标 + reward，程序化可复现）

### A5.1 Issue taxonomy（最小集合）

* `missing_slot`
* `inconsistency`
* `overclaim`
* `unsupported`（无 citation / citation 不支持该 slot）

### A5.2 Verifier score

[
\mathcal{V} = \text{ClinicalCorrectness} - \alpha#missing - \beta#inconsistency - \gamma#overclaim - \delta#unsupported
]

* ClinicalCorrectness 的最小实现：结构化 slots 与 GT（或 labeler 抽取）做 F1（见实验部分）

**验收检查**：同一输入多次运行 verifier 输出稳定；issues 能定位到句子 span 与 tokens/evidence。

---

## A6. Split Policy（learned budgeted refinement：bandit/offline RL）

这是“主动”与“非 heuristic”的核心。

### A6.1 状态特征 (x_i)（每个候选 token 一条）

最小必备：

* evidence uncertainty entropy
* tokenizer recon error（VQ residual）
* citation pressure（被引用频次/attention mass）
* depth/history（split 深度、是否已 split）

### A6.2 训练数据（logged trajectories）

用 heuristic policy（entropy+recon）跑一批 case，记录：

* 在 step k 选择 split token set (A_k)
* refine 后 (\Delta \mathcal{V}) 与 (\Delta|T|)
* reward (R=\Delta\mathcal{V}-\lambda\Delta|T|)

### A6.3 学习方式（稳定优先）

* **contextual bandit**（推荐先做）：

  * 训练 (\hat{Q}(x)\approx \mathbb{E}[R|x])，选择 top-m token split
* 进阶：offline RL（CQL/IQL 类）在 bandit 之上再做

**验收检查**：learned policy 在 offline evaluation（IPS/DR 或回放）优于 heuristic；在线小规模验证能提升 Pareto。

---

## A7. 推理闭环（Algorithm 1：Tokenize→Generate→Verify→Refine→Regenerate）

实现上就是一个可复现 runner：

```text
T0 = coarse_tokens(V)
y0, G0 = generate(T0)
V0, issues0 = verify(y0, G0)

for k in 0..K-1:
  candidates = expandable_tokens(Tk)
  Ak = policy_select(candidates, features, budget_left)
  Tk+1 = refine(Tk, Ak)      # select children (path A) or on-demand encode (path B)
  yk+1, Gk+1 = generate(Tk+1)
  Vk+1, issuesk+1 = verify(yk+1, Gk+1)
  if (Vk+1-Vk)/( |Tk+1|-|Tk| ) < tau: break
return y*, G*, trace
```

**验收检查**：每轮 trace 写清：split 哪些 token、增加多少 token、latency、verifier 提升。

---

## A8. 代码结构（最小可落地 repo skeleton）

```
voxtoken/
  configs/
    train_tokenizer.yaml
    train_evidence.yaml
    train_policy.yaml
    inference.yaml
  data/
    ingest.py
    preprocess.py
    splits/
  models/
    encoder3d.py
    vq.py              # VQ/RVQ quantizer
    tokenizer.py       # hierarchical tokens + omega mapping
    evidence_head.py
    generator/
      planner.py       # rule/LLM plan
      realize.py       # template or LLM
      constrained.py   # constraints
    policy.py
  verify/
    rules.py
    scorer.py
    extract_slots.py   # from GT or labeler
  runner/
    train_tokenizer.py
    train_evidence.py
    train_policy.py
    infer_refine.py
  eval/
    metrics.py
    counterfactuals.py
    pareto.py
  artifacts/           # outputs
```

---

# B. 实验方案（最小不可分，Oral 反推）

## B0. 需要证明的 4 个核心命题（每个对应一组实验）

1. **Adaptive tokenization 的 Pareto 优势**：同等 correctness 更少 tokens/latency，或同等预算更高 correctness
2. **token-citation 的因果性**：citation 不是 attention 装饰（反事实实验必须成立）
3. **unsupported claim 机制性趋近 0**：constrained + verifier gate 显著压低 unsupported
4. **learned split policy > heuristic**：主动策略带来额外 Pareto 改善

---

## B1. 数据与切分（必须能支持 grounding）

### B1.1 数据组成（最小）

* **主数据集**：CT + report（用于 token→text）
* **grounding 子集**：带 lesion mask/box（用于 sentence→token→region 硬评测）

### B1.2 切分

* train/val/test：按 patient-level split
* grounding 子集必须覆盖 test（否则容易被质疑为“只做训练不做评测”）

---

## B2. 指标（主指标必须硬）

### B2.1 Clinical correctness（主）

* 从生成 report 中抽取结构化 slots（finding/side/location/size/negation/uncertainty）
* 与 GT slots 做 micro/macro F1
* 若 GT 无结构化 slots：用固定抽取器（规则或医学 IE 模型）做一致评测（关键是可复现）

### B2.2 Grounding（主）

* sentence → cited tokens → (\Omega) 与 GT mask/box 的关系

  * hit-rate：是否命中（IoU>0 或 Dice>0）
  * IoU/Dice（可选）
* token density vs lesion overlap（refine 后是否更集中）

### B2.3 Unsupported claim rate（主）

* `unsupported = 句子无 citation 或 citation 不支持其 slot`
* 目标：本文方法接近 0，baseline 明显更高

### B2.4 Efficiency（主）

* #tokens（每轮、最终）
* wall-clock latency（必须真实计时）
* GPU memory peak（可选）

### B2.5 Pareto（必须图）

* correctness vs #tokens
* correctness vs latency

---

## B3. Baselines（最小不可分清单，必须齐）

1. **Fixed-grid 3D tokens**（同 encoder + 同 generator，只禁用 adaptive split）
2. **2D slice tokens / 2.5D**（uniform sampling）
3. **ROI crop pipeline**（传统 coarse-to-fine，非 token）
4. **Heuristic split**（entropy/recon）
5. **Learned split policy（本文方法）**
6. **No-citation**（去掉 citation/constrained）
7. **No-refine**（只用 level-0 或一次性 tokens）
8. **Citation = attention top-k**（弱引用） vs **evidence-graph constrained citation（本文方法）**（强引用）

---

## B4. 消融（每条都要能回答一个 reviewer 问题）

* A1：adaptive split vs fixed tokens（证明 tokenization 贡献）
* A2：learned split vs heuristic split（证明非 heuristic）
* A3：constrained vs unconstrained（证明 unsupported 机制性下降）
* A4：mask-head on/off（证明 grounding 不是伪）
* A5：path A（选择式）vs path B（按需式）（若实现 B，这是 Oral 强加分）

---

## B5. 反事实/因果实验（Oral 级关键；最小三条必须做）

1. **Permutation test**：随机打乱 token 支持域 (\Omega_i)（保持 embedding 不变）

   * 预期：grounding 与 correctness 明显下降
2. **Citation swap**：交换 cited token ids（句子不变）

   * 预期：verifier 立即报 unsupported，上升显著
3. **Mask-level sanity**：对有 GT mask 的 finding

   * 统计 refine 后新增 tokens 的 (\Omega) 与 lesion overlap 分布显著上升（相对随机/heuristic）

> 这三条是用来击穿“citation 只是 attention 可视化”的。

---

## B6. 结果呈现（Oral 必备 3 张主图 + 2 张主表）

### 主图（必须）

* **Fig1**：方法框图（层级 tokens + refine loop + constrained generation）
* **Fig2**：Pareto（correctness–#tokens / correctness–latency）
* **Fig3**：Grounding 可视化 + 反事实结果（bar/curve）

### 主表（必须）

* **Table1**：主指标对比（correctness、grounding、unsupported、latency）
* **Table2**：消融（split policy、constrained、mask head）

---

## B7. 训练/评测协议（可复现）

* 固定随机种子（≥3 seeds）
* 同等 token budget 下比较（避免“我 token 更多所以更好”的喷点）
* 同等 backbone 下比较（只换 tokenization/策略/约束）
* 所有输出 artifacts（report/evidence/trace）写入 `artifacts/run_id/`

---

# C. 最小可执行里程碑（每一步都有“必须验收”的标准）

## M0（跑通闭环）

* fixed tokens（level-0）+ template generator + verifier + citations
* **验收**：unsupported rate 显著低于 no-citation；trace 完整

## M1（层级 tokens + refine）

* 选择式 split（从 level-1/2 挑子 tokens）
* **验收**：Pareto 曲线出现边际收益递减；refine 有效

## M2（grounding 子集）

* mask head 只做 2–3 类 finding
* **验收**：Fig3 能画出 token box/mask overlay + grounding 指标

## M3（learned split policy）

* contextual bandit 训练 (\hat{Q}(x)) 并在线验证
* **验收**：learned policy > heuristic（Pareto dominate 或在固定预算上提升显著）

## M4（反事实因果）

* permutation/citation swap/mask sanity 全部跑通
* **验收**：反事实显著劣化（统计显著），否则 Oral 很难

---

## 第一周任务清单（可直接开工，无需额外前置信息）

1. 把 Schema + trace runner 写出来（A0/A7）
2. 实现 fixed-grid tokens（先用连续 embedding 也行，但接口按离散 token 设计）
3. 实现 rule-based planner + template realization + citations（A4）
4. 实现 verifier（A5）
5. 跑 50 个样本输出 artifacts，确保三件套全有（验收 M0）

---

可选：将上述内容进一步工程化到可直接写代码的粒度：

* **每个模块的函数签名**（Python typing）
* **4 个 config 模板**（tokenizer/evidence/policy/infer）
* **Table1/Table2/Fig2/Fig3 的具体列名与作图脚本接口**（确保结果格式不会在写论文时崩）。

下面给出一份**“最小不可分（atomic）可落地”**的完整方案：将 A/B 进一步**工程化到“可直接开 repo 写代码/跑实验”的粒度**——包含：

* **模块边界与接口（Python typing 级别）**
* **训练/推理闭环（Algorithm 1/2 可直接进论文）**
* **4 份 YAML 配置模板**
* **评测脚本与产物格式**
* **Table/Figure 的列名与作图输入约定**
* **反事实实验的可执行实现方式**

论文主线采用**离散 VQ tokens**（面向 Oral 级表述），同时提供一个**连续 tokens 的 MVP 替身**（保持接口不变，便于先跑通再替换为 VQ）。

---

## 0. 两条原则（不写清楚会导致后续接口/评测不一致）

### 0.1 “Token = 表示 + 3D 支持域 + 可追溯父子关系”

无论先用连续还是离散，**Token 数据结构必须固定**，并且每个 token 必须具备：

* `omega_box_mm = (x0,x1,y0,y1,z0,z1)`：在**原始物理坐标（mm）**的 3D box（不是 resample 后 index）
* `level`、`parent_id`、`children_ids`：支持层级 split
* `code`：离散 VQ 时为 `int`，连续 MVP 时可为 `None`
* `embed_ref`：embedding 存储索引（避免 json 塞大向量）

### 0.2 “三件套产物（任何样本、任何 baseline 都必须输出）”

每个 case 必须落盘：

1. `final_report.txt`（每句带 citations）
2. `evidence_graph.json`（结构化 slots + token 支持）
3. `trace.jsonl`（每轮 refine：split、预算、时延、verifier 前后）

这是后续做 Pareto、做反事实、做可视化的**唯一可信证据链**。

---

## 1. 数据选择（让 grounding 变硬，不靠嘴）

至少需要一个“报告生成主数据”+ 一个“硬 grounding 子集”。

推荐组合（公开可复现、论文可引用）：

* **CT-RATE**：大规模胸部 CT 体数据 + 放射科报告（用于 report learning / 自监督 tokenizer）([arXiv][1])
* **RadGenome-Chest CT**：在 CT-RATE 基础上扩展的**region-guided**数据：提供大量**句子↔分割 mask**级别 grounding（用于最关键的“硬 grounding”主评测）([Nature][2])
* （可选）**RAD-ChestCT**：大规模多异常/位置标签胸 CT（用于外部泛化或弱监督证据头）([Zenodo][3])
* （可选）**LIDC-IDRI**：肺结节 CT 数据（可用来做结节 mask/box 级 sanity 或训练 mask-head）([癌症影像库][4])

> 关键点：RadGenome 提供 “**sentence→mask**” 的硬评测抓手；无需一开始就拿到所有 lesion mask，也可以先把“citation/因果”部分做成硬结论。

---

## 2. 代码级实现方案（Atomic 模块 + 接口）

下面给出**可直接照着写**的模块边界与 typing（建议用 `dataclasses` + `pydantic` 管 json）。

### 2.1 核心数据结构（必须固定）

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Literal, Any

BoxMM = Tuple[float, float, float, float, float, float]  # x0,x1,y0,y1,z0,z1

@dataclass
class Token:
    token_id: int
    level: int
    omega_box_mm: BoxMM
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)

    # representation
    code: Optional[int] = None          # VQ token id (discrete); None for continuous MVP
    embed_ref: Optional[int] = None     # index into a tensor bank on disk (np.memmap/pt)

@dataclass
class Citation:
    sent_id: int
    cited_token_ids: List[int]

@dataclass
class EvidenceNode:
    eid: str
    finding_type: str                   # e.g., nodule/effusion/atelectasis/normal...
    attrs: Dict[str, Any]               # side, location, size_bin, certainty, negation...
    supported_token_ids: List[int]
    optional_mask_ref: Optional[str] = None  # path to mask in npy/nii.gz
    optional_measure: Optional[Dict[str, float]] = None  # diameter/volume...

IssueType = Literal["missing_slot","inconsistency","overclaim","unsupported"]

@dataclass
class Issue:
    type: IssueType
    span: Tuple[int, int]               # char span in report, or (sent_id, token_span)
    reason: str
    related_eids: List[str] = field(default_factory=list)
    related_tokens: List[int] = field(default_factory=list)

@dataclass
class TraceStep:
    k: int
    budget_total: int
    budget_used: int
    split_token_ids: List[int]
    added_token_ids: List[int]
    verifier_score_before: float
    verifier_score_after: float
    latency_ms: Dict[str, float]        # tokenize/generate/verify/total
```

---

## 2.2 Tokenizer（离散主线 + 连续 MVP 兼容）

### 2.2.1 Tokenizer 接口（保持不变）

```python
import torch
from typing import NamedTuple

class TokenPyramid(NamedTuple):
    # tokens_by_level[l] is a list of Token (with omega) and a tensor bank ref
    tokens_by_level: Dict[int, List[Token]]
    # embed_bank_path points to saved tensor bank on disk for each level
    embed_bank_path: Dict[int, str]

class Tokenizer3D(torch.nn.Module):
    def __init__(self, cfg: Dict[str, Any]): ...
    @torch.no_grad()
    def build_pyramid(self, volume: torch.Tensor) -> TokenPyramid:
        """
        volume: (C, D, H, W) float32
        returns: multi-level tokens, each token has omega_box_mm.
        """
        ...

    @torch.no_grad()
    def select_tokens(self, pyramid: TokenPyramid, active_nodes: List[int], budget_B: int) -> List[Token]:
        """
        Path A (选择式)：从 pyramid 中选 token（包含 coarse + 被 split 的 fine tokens）
        """
        ...
```

### 2.2.2 两种 tokenizer 落地路径（先 A 后 B）

* **Path A（推荐先做，工程最省）**：一次前向拿到全部 level 的 token embedding / code，然后 refine 就是“**选择更多细层 token**”。
* **Path B（Oral 强化，真正 budget 化计算）**：只对被 split 的节点 crop→encode→quantize（encoder 计算也随预算走）。

论文写法：**方法定义支持 Path B**，实现先用 A，后面补 B（有时间就上）。

---

## 2.3 Evidence Head（让 citation 有“可检验语义”）

### 2.3.1 接口

```python
class EvidenceHead(torch.nn.Module):
    def __init__(self, cfg: Dict[str, Any]): ...

    def forward(self, tokens: List[Token], embed_bank: torch.Tensor) -> List[EvidenceNode]:
        """
        tokens: selected tokens
        embed_bank: (N, d) embeddings aligned with tokens
        returns: evidence nodes with structured attrs & supported_token_ids
        """
        ...
```

### 2.3.2 最小可跑实现（M0/M1 必须能做）

* **每 token 多头分类**（即使很弱也要跑通）：

  * `finding_type`（多标签或 top-k）
  * `certainty/negation`
  * `laterality`（left/right/na）
  * `coarse_location`（lung/pleura/mediastinum/other…，先粗粒度）

* **Graph merge**：空间重叠 + finding_type 相同 → 合并为一个 EvidenceNode（supported_token_ids 合并）

### 2.3.3 Oral 强化（M2/M3 才上）

* 对 2–3 个高价值 finding（例如 nodule/effusion/pneumothorax）加：

  * `mask_head(omega crop) -> lesion mask`
  * `measurement`（diameter/extent/volume）

> 可用 RadGenome 的句子↔mask 做“器官/区域 grounding”的硬评测，再用少量 lesion 标注把 mask-head 做硬 sanity（两条线并行）。

---

## 2.4 Planner + Realizer（两阶段生成，天然可约束）

### 2.4.1 Plan 的结构（slot tuple + token 支持）

```python
@dataclass
class FactSlot:
    finding_type: str
    side: str
    location: str
    size_bin: str
    certainty: str
    supported_token_ids: List[int]

@dataclass
class ReportPlan:
    facts: List[FactSlot]
    impression: List[FactSlot]  # or derived summary facts
```

### 2.4.2 Planner（M0 用规则就够；M2 再换 LLM）

```python
class Planner:
    def __init__(self, cfg: Dict[str, Any]): ...

    def build_plan(self, evidence: List[EvidenceNode]) -> ReportPlan:
        """
        Deterministic: sort by severity/certainty, group by anatomy, cap per organ.
        """
        ...
```

### 2.4.3 Realizer（M0 模板填充；M2 加 LLM 润色但不放权）

```python
class Realizer:
    def __init__(self, cfg: Dict[str, Any]): ...

    def realize(self, plan: ReportPlan) -> Tuple[str, List[Citation]]:
        """
        Returns report text and per-sentence citations.
        """
        ...
```

---

## 2.5 Constrained Decoding（把 unsupported 变成“结构性难发生”）

即使使用 LLM，也要把约束写成**可执行 gate**：

* **关键词集合约束**：finding/side/location/size/certainty 只能来自 plan
* **每句必须输出 citation**：没有 citation → verifier 判 unsupported（不可协商）

最小实现：模板天然满足；LLM 强化实现：先生成草稿，再做**slot 对齐与替换**（不通过就 fallback 到模板句）。

---

## 2.6 Verifier（指标=reward=可复现实验开关）

### 2.6.1 Verifier 接口

```python
class Verifier:
    def __init__(self, cfg: Dict[str, Any]): ...

    def verify(self, report_text: str, citations: List[Citation], plan: ReportPlan) -> Tuple[float, List[Issue]]:
        """
        1) parse report -> slots (or use the plan as canonical)
        2) check missing_slot / inconsistency / overclaim / unsupported
        returns score and issues
        """
        ...
```

### 2.6.2 评分（建议固定，别每次改）

[
\mathcal{V} = \text{SlotF1} - \alpha #missing - \beta #inconsistency - \gamma #overclaim - \delta #unsupported
]

* **M0**：SlotF1 可以先用“对 reference report 的 slot 抽取”（silver），哪怕抽取器简单也行，但必须固定可复现。
* **M2**：RadGenome 的 grounded sentence/region 可提供硬对齐的部分（location/organ 维度）([Nature][2])

---

## 2.7 Split Policy（learned active 的最短路径：contextual bandit）

### 2.7.1 特征与动作

```python
@dataclass
class TokenFeatures:
    token_id: int
    level: int
    recon_error: float
    evidence_entropy: float
    citation_pressure: float
    history_splits: int

class SplitPolicy(torch.nn.Module):
    def __init__(self, cfg: Dict[str, Any]): ...

    def score(self, feats: List[TokenFeatures]) -> List[Tuple[int, float]]:
        """returns (token_id, score)"""
        ...
```

### 2.7.2 训练（离线 bandit，稳）

* 用 heuristic（entropy+recon）跑出 logged trajectories（每一步 refine 后的 (\Delta \mathcal{V})）
* 训练一个 ( \hat{Q}(x)\approx \mathbb{E}[\Delta \mathcal{V}-\lambda\Delta|T| \mid x] )
* 推理时选 top-m split（受 budget 约束）

> 这样可以非常清楚地在论文中陈述：**learned policy > heuristic**，且训练稳定。

---

## 2.8 推理 Runner（Algorithm 1 可直接进 paper）

```python
class RefineRunner:
    def __init__(self, tokenizer, evidence_head, planner, realizer, verifier, policy, cfg): ...

    @torch.no_grad()
    def run_case(self, volume, budget_B: int) -> Dict[str, Any]:
        pyramid = self.tokenizer.build_pyramid(volume)          # Path A
        Tk = self.tokenizer.select_tokens(pyramid, active_nodes=[], budget_B=budget_B)

        plan, report, cites, score, issues = self._gen_verify(Tk, pyramid)
        trace = []

        for k in range(self.cfg["refine"]["max_rounds"]):
            feats = self._featurize_tokens(Tk, pyramid, cites, issues)
            split_ids = self._select_splits(feats, budget_left=budget_B-len(Tk))
            Tk2 = self._refine_select(pyramid, Tk, split_ids, budget_B)

            plan2, report2, cites2, score2, issues2 = self._gen_verify(Tk2, pyramid)
            trace.append(TraceStep(...))

            if self._stop(score, score2, len(Tk), len(Tk2)): 
                break
            Tk, plan, report, cites, score, issues = Tk2, plan2, report2, cites2, score2, issues2

        return self._dump_artifacts(report, cites, plan, trace, issues)
```

---

# 3. 训练方案（按“最小不可分”拆成 3 个 stage）

## Stage T（Tokenizer 自监督：VQ-MAE / VQ-VAE）

**目标**：得到离散 codebook + 多尺度 tokens；并产出 recon error 特征。

* 输入：CT-RATE volumes（无需报告）([arXiv][1])
* loss：recon + commitment + codebook usage regularization
* 产物：`tokenizer.ckpt` + 每层 `codebook.pt` + `recon_error_stats.json`

**验收**：

* codebook perplexity 不塌
* recon error 能区分“复杂区域/简单区域”（用于 split feature）

---

## Stage E（Evidence Head：结构化 slots + 可选 mask-head）

**目标**：token→finding/attrs，能构 evidence graph。

* 输入：

  * CT-RATE（弱监督：全局 abnormality labels）
  * RadGenome（强监督：句子↔mask 对齐，用来训练 location/organ 相关与 grounding）([Nature][2])
* 产物：`evidence_head.ckpt`

**验收**：

* evidence graph 节点数受控（每 organ cap N）
* 每个 evidence node 都能追溯 `supported_token_ids + omega_box_mm`

---

## Stage P（Policy：offline contextual bandit）

**目标**：learned split > heuristic split。

* 数据：用 heuristic 跑出来的 `trace.jsonl`（logged trajectories）
* 训练：回归 ( \hat{Q}(x) ) 或 pairwise ranking（谁 split 更赚）
* 产物：`policy.ckpt`

**验收**：

* 在离线回放/重放上，learned 策略在相同预算下 (\mathcal{V}) 更高或 Pareto 更优

---

# 4. 配置模板（4 份 YAML，可直接照抄改路径）

## 4.1 `configs/train_tokenizer.yaml`

```yaml
data:
  dataset: ct_rate
  root: /data/ct_rate
  spacing_mm: [1.0, 1.0, 3.0]
  windows:
    - {center: -600, width: 1500}   # lung
    - {center: 40, width: 400}      # mediastinum
model:
  encoder: conv3d_unet
  levels: [0,1,2]
  level_grids:
    - [8, 8, 8]
    - [16,16,16]
    - [32,32,32]
  quantizer:
    type: rvq
    codebook_size: [1024, 1024, 1024]
    embed_dim: 256
train:
  objective: vq_mae
  mask_ratio: 0.6
  batch_size: 2
  lr: 2e-4
  epochs: 50
  save_dir: artifacts/tokenizer
```

## 4.2 `configs/train_evidence.yaml`

```yaml
data:
  datasets:
    - {name: radgenome, root: /data/radgenome_chestct, use_grounded_sentences: true}
    - {name: ct_rate, root: /data/ct_rate, use_abnormality_labels: true}
model:
  evidence_head:
    embed_dim: 256
    heads:
      finding_type: {num_classes: 20}
      laterality: {num_classes: 3}
      location: {num_classes: 12}
      certainty: {num_classes: 3}
    mask_head:
      enabled: false
train:
  batch_size: 4
  lr: 1e-4
  epochs: 30
  save_dir: artifacts/evidence
```

## 4.3 `configs/train_policy.yaml`

```yaml
data:
  traces_glob: artifacts/runs/*/trace.jsonl
features:
  use_recon_error: true
  use_evidence_entropy: true
  use_citation_pressure: true
  use_history: true
model:
  type: mlp_regressor
  hidden: [128, 64]
train:
  lr: 1e-3
  epochs: 20
  save_dir: artifacts/policy
```

## 4.4 `configs/inference.yaml`

```yaml
runtime:
  device: cuda
  seed: 0
refine:
  budget_B: 512
  max_rounds: 5
  tau: 1e-3
generation:
  mode: template   # template | llm_rewrite
  require_citation: true
verifier:
  alpha: 0.5
  beta: 1.0
  gamma: 1.0
  delta: 5.0
output:
  save_dir: artifacts/runs
```

---

# 5. 实验方案（按 Oral 反推：4 个命题 → 4 组结果）

## 命题 H1：Adaptive tokenization 给出 correctness–cost Pareto 优势

**做法**：在多个 budget B（如 128/256/512/1024）下画 Pareto。

* X 轴：`#tokens` 或 `latency_ms`
* Y 轴：`Slot-F1`（主）+ `Grounding hit-rate`（主）

**关键对比**：

* Fixed-grid（同 tokenizer/同生成器，只禁用 split）
* ROI crop pipeline（传统 coarse-to-fine）
* 2D/2.5D slice tokens

---

## 命题 H2：Citation 不是 attention 装饰（因果反事实必须成立）

最小三件套（proposal 里那三条，必须可执行）：

1. **Permutation((\Omega))**：打乱 token 的 `omega_box_mm`（embedding 不变）
   预期：grounding 与 correctness 显著下降（统计显著）。

2. **Citation swap**：交换句子引用的 token ids（句子文本不变）
   预期：verifier 的 `unsupported` 立刻暴涨。

3. **Mask-level sanity**（在 RadGenome 的 region mask 上做也行）：
   统计 refine 后新增 tokens 与目标 mask 的 overlap 分布显著上升（相对随机/heuristic）。

RadGenome 之所以适合：它有**句子↔mask**规模化数据，能把上述反事实做成“硬结论”。([Nature][2])

---

## 命题 H3：Unsupported claim rate 机制性趋近 0（不是靠 prompt）

对比：

* No-citation（去掉 citation/约束）
* Attention-topk citation（弱引用）
* 本文方法：EvidenceGraph + constrained generation + verifier gate（强引用）

**主指标**：

* `Unsupported (%)`：句子无 citation 或 citation 不支持 slot
* `Overclaim (%)`：证据不确定但用确定语气

目标：本文方法接近 0，baseline 显著更高。

---

## 命题 H4：Learned split policy > heuristic split（“active 真 active”）

对比：

* heuristic split（entropy / recon error）
* learned bandit policy（本文方法）
* random split（下界）

呈现方式：

* 固定预算 B：比较 (\mathcal{V})、Slot-F1、Grounding、latency
* Pareto：learned 曲线应整体优于 heuristic 或至少在关键 budget 上显著更好

---

# 6. 指标定义（论文里需要写得像“硬标准”，而不是 NLP 指标）

## 6.1 Clinical correctness（Slot-F1）

slot 维度建议固定为：

* finding_type
* laterality
* coarse_location
* size_bin（M0 可先不做；M2 再加）
* certainty/negation

计算：micro/macro F1（建议 micro 为主，macro 辅助）

## 6.2 Grounding

给定一句话的 citations → token boxes → 与 GT mask/box 的关系：

* `Hit@τ`：是否存在 cited token 与 GT mask overlap > τ
* `Mean IoU` 或 `Dice`（有 mask 时）

RadGenome 提供“句子对应 region mask”的直接评测通道。([Nature][2])

## 6.3 Unsupported / Overclaim（Verifier 输出）

* unsupported：缺 citation / citation 不支持 slot
* overclaim：certainty mismatch（如 evidence=uncertain 但文本=definite）

## 6.4 Efficiency

* `#tokens`
* `latency_ms`（tokenize / gen / verify / total）
* （可选）`peak_mem`

---

# 7. 表格与作图模板（可直接按列落 CSV）

## 7.1 Table 1（主表：对齐 reviewer 所有硬喷点）

`table_main.csv` 列名建议：

* `method`
* `budget_B`
* `tokens_final`
* `lat_total_ms`
* `slot_f1_micro`
* `ground_hit@0.0`（overlap>0）
* `ground_hit@0.1`
* `unsupported_sent_pct`
* `overclaim_sent_pct`
* `missing_slot_per_report`
* `inconsistency_per_report`

## 7.2 Table 2（消融表）

`table_ablation.csv` 列名建议：

* `variant`（no_refine / no_citation / no_constrained / heuristic_policy / learned_policy / no_mask_head …）
* `budget_B`
* `slot_f1_micro`
* `ground_hit@0.1`
* `unsupported_sent_pct`
* `lat_total_ms`

## 7.3 Fig 2（Pareto）

输入格式：每条 run 输出 `metrics.json`，包含：

* `budget_B`
* `tokens_final`
* `lat_total_ms`
* `slot_f1_micro`

作图脚本契约：

* `eval/pareto.py --glob "artifacts/runs/*/metrics.json" --x tokens_final --y slot_f1_micro`

## 7.4 Fig 3（Grounding + 反事实）

输入：`counterfactual.csv`

* `cf_type`（permute_omega / swap_citation / random_split …）
* `slot_f1_micro`
* `ground_hit@0.1`
* `unsupported_sent_pct`

---

# 8. 反事实实验怎么“可执行”（可直接写进 eval/counterfactuals.py）

提供三个开关（跑出来就是论文结论）：

1. `permute_omega(tokens)`：对 tokens 的 `omega_box_mm` 做随机置换（保持 embedding 顺序不变）
2. `swap_citations(report)`：在同一 report 内随机交换句子 citations
3. `mask_sanity(new_tokens, gt_mask)`：统计新增 tokens 与 gt_mask overlap 的分位数（p50/p75/p90）

所有反事实都必须落盘：`artifacts/runs/<id>/counterfactual/*.json`

---

# 9. 里程碑（每一步都“验收即有论文证据”）

## M0：闭环跑通（1 周内必须做到）

* fixed tokens（level-0）+ template + citations + verifier
* 输出三件套
* **验收**：`unsupported` 相对 no-citation 显著下降

## M1：层级 tokens + refine（2–3 周）

* Path A 选择式 split
* 画 Pareto（正确性 vs tokens/latency）
* **验收**：边际收益递减（曲线有膝点）

## M2：RadGenome grounding 主评测（并行推进）

* 用句子↔mask 做 grounding hit-rate
* 做 permutation/citation swap
* **验收**：反事实显著劣化（因果链成立）([Nature][2])

## M3：learned split policy（bandit）

* learned > heuristic
* **验收**：相同预算下 (\mathcal{V})/Slot-F1 提升或 Pareto dominate

## M4：外部验证（加分但别拖主线）

* RAD-ChestCT 或别的中心数据做 robustness（可选）([CVIT - Center for Virtual Imaging Trials][5])

---

## 10. 可直接照着写的“最小 repo 骨架”

（与当前 draft 类似，但脚本接口固定为“产物驱动评测”）

```
voxtoken/
  configs/
  data/
    adapters/ct_rate.py
    adapters/radgenome.py
    preprocess.py
  core/
    schemas.py          # Token/Evidence/Trace dataclasses
    geometry.py         # grid<->box(mm) mapping
  models/
    tokenizer/
      encoder3d.py
      quantizer.py
      tokenizer3d.py
    evidence_head.py
    policy.py
  generation/
    planner.py
    realizer.py
    constrained.py
  verify/
    verifier.py
    slot_extract.py
  runner/
    train_tokenizer.py
    train_evidence.py
    train_policy.py
    infer_refine.py
  eval/
    metrics.py
    pareto.py
    counterfactuals.py
    visualize_grounding.py
```

---

## 最关键的实现取舍（默认决策）

1. **论文主线：离散 VQ/RVQ tokens（VoxToken++ 名字才站得住）**
2. **工程启动：允许连续 tokens MVP，但接口必须按离散 token 设计**（`code=None`，其余完全一致）
3. **先用 Path A（选择式 split）拿 Pareto/反事实/learned policy 结果**，再决定是否上 Path B（按需编码）去拿更漂亮的 latency 叙事。

---

按本方案实施，将形成一条非常“NeurIPS reviewer 友好”的证据链：

* **tokenization 有 budget 的形式化定义**（不是 crop heuristics）
* **active split 有 learned policy + reward（verifier）**（不是“主观认为这里重要”）
* **grounding 有句子↔mask 的硬评测 + 反事实因果实验**（不是 attention 可视化）
* **unsupported 被 constrained+verifier 机制性压低**（不是靠评测后验抓）

下一步最优先的第一步不是训练大模型，而是：
**先把 schemas + runner + verifier + 三件套产物固定**，跑 50 个样本把 trace/metrics 全打通——这一步一旦完成，后续模块替换（VQ、更强 evidence、更强 policy）都不会“做了很多但无法证明”。

[1]: https://arxiv.org/abs/2403.17834 "[2403.17834] Developing Generalist Foundation Models ..."
[2]: https://www.nature.com/articles/s41597-025-05922-9 "Development of a large-scale grounded vision language dataset for chest CT analysis | Scientific Data"
[3]: https://zenodo.org/records/6406114 "RAD-ChestCT Dataset"
[4]: https://www.cancerimagingarchive.net/collection/lidc-idri/?utm_source=chatgpt.com "LIDC-IDRI - The Cancer Imaging Archive (TCIA)"
[5]: https://cvit.duke.edu/resource/rad-chestct-dataset/ "RAD-ChestCT Dataset - CVIT - Center for Virtual Imaging Trials"
