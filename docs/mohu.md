# mohu.md（Gaps & Ambiguities）

本文件用于记录两类阻塞项，并驱动“实现→验证→再修复”的闭环迭代。

收敛条件：当 **Gap** 与 **Ambiguity** 两部分都为空时，进入实验定义与运行阶段（`docs/experiment.md`）。

---

## 1. 当前没有实现的点（Gap）

> 规则：逐条实现；每条必须“实现→验证通过→才能进入下一条”；验证失败必须回到该条继续修复。

（空表示当前无 Gap）

---

## 2. 当前 `docs/plan.md` 与实现之间的模糊点（Ambiguity）

> 规则：逐条澄清为可执行规格（指标/协议/保存物/脚本/验收标准）；必要时修订 `docs/plan.md`（保留 before/after）；并完成实现与验证。

（空表示当前无 Ambiguity）

