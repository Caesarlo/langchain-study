# 18 · Agent Architecture Design（智能体架构设计 + Capstone）

> 横向工程能力的**综合课**，也是整门课的 capstone。把前面所有原语（loop / 工具 /
> 记忆 / RAG / 规划 / 反思 / 多 agent / 工作流）和三门工程学科（prompt / context /
> harness）组装成**一个真实 agent 系统**。核心是一套"从需求到架构"的决策方法论。
> 建议在 `08` 之后建立判断力，全部学完后用它做 capstone。

## 学习目标

- 掌握一套**架构决策框架**：给定任务，推导出该用什么结构
- 学会"从失败模式倒推架构"——先问最容易怎么错，再据此设计
- 会写一份 **Agent Design Doc**（目标/约束/失败模式/工具/记忆/拓扑/护栏/eval/成本）
- 能在 workflow↔agent、单 agent↔多 agent 之间做有依据的取舍
- 完成一个端到端 capstone：设计 → 实现 → eval → 迭代 → 复盘

## 核心概念

### 1. 从失败模式倒推（设计的起点）
- 别先想"用什么框架"，先问：**这个任务最可能怎么失败？**
  （查错？步数失控？工具选错？上下文爆？幻觉？越权？）
- 每个高频失败 → 对应一个架构对策（护栏 / 子 agent / 工作流 / HITL / eval）

### 2. 核心决策树
- **workflow vs agent**（呼应 `08`）：能用确定性工作流就别上 autonomous agent
- **单 agent + 多工具 vs 多 agent**（呼应 `07`）：
  默认单 agent；只有当职责/上下文确实冲突、可并行时才拆
  —— Cognition _Don't Build Multi-Agents_ 的告诫：多 agent 共享上下文难、易脆 ⭐
- **拓扑选择**：supervisor / hierarchical / handoff / network（呼应 `07`）
- **同步 vs 异步、单轮 vs 长程**：决定记忆与 harness 复杂度

### 3. 状态与记忆架构（呼应 `03` / `16`）
- 信息放哪（上下文 / scratchpad / 向量库 / DB）、存多久、谁可见
- 单一可信源（避免 `16` 的 clash/poisoning）

### 4. 工具面设计（呼应 `02` / `17`）
- 工具粒度与数量：太多 → 选择困难 → 分组或拆 agent
- 命名空间、ACI 友好、危险工具的审批边界

### 5. 上下文架构（呼应 `16`）
- 每个 agent / 每一步**应该看到什么**——这是多 agent 设计的真正难点
- 子 agent 隔离 vs 共享上下文的取舍

### 6. 人在环路与护栏（呼应 `13`）
- 在哪设审批点（高风险动作前）、哪些输出要校验、注入与越权怎么防

### 7. 成本 / 延迟架构（呼应 `13`）
- 模型分级（简单步小模型）、prompt caching、并行、预算上限

### 8. 可演进性（呼应 `08` / `12`）
- **从最简方案开始**，每加一层复杂度都要有 eval 兜底证明值得
- 架构是迭代出来的，不是一次设计对的

### 9. 安全作为架构关注点（贯穿）
- **"致命三要素"**（lethal trifecta）：私有数据 + 不可信内容 + 对外通信，三者同现即高危
- 设计时就隔离这三者，而不是事后打补丁（与 `13` guardrails、`17` 审批协同）
- 注：独立的安全/对抗专题计划补到 `19-security`（见根 README 路线）

## 代表性参考

- **Anthropic _Building Effective Agents_**（选型与"别过度工程化"）
- **Anthropic _How we built our multi-agent research system_**（2025，真实架构权衡）⭐
- **Cognition _Don't Build Multi-Agents_**（Walden Yan, 2025）⭐ 反方必读
- OpenAI _A Practical Guide to Building Agents_（设计清单）
- _12-Factor Agents_（HumanLayer，架构/工程原则，呼应 `17`）
- Simon Willison _The lethal trifecta_（2025，安全三要素）
- 你自己的 `langgraph-study/workflow-agent/`（对照真实工作流实现）

## 手撕任务（Capstone）

> 不是新骨架，而是**用前面所有手撕件，做一个真实的 agent 系统**。

1. [ ] 选一个**有真实价值的任务**（如：代码仓库问答助手 / 资料调研 agent /
   自动化运维助手），范围适中、能跑通端到端
2. [ ] 写一份 **Agent Design Doc**：
   - 目标与成功标准、约束（成本/延迟/安全）
   - **失败模式清单** → 每条的对策
   - 工具面、记忆/上下文架构、拓扑（单 or 多 agent，为什么）
   - 护栏与 HITL 点、成本/延迟方案
   - eval 方案（接 `12`）
3. [ ] 用 `01`–`17` 的手撕件**组装实现**（能复用框架的地方注明取舍）
4. [ ] 接 `12` 的 eval harness，跑分；接 `17` 的 replay 调试
5. [ ] 至少**迭代一轮**：根据 eval 结果改架构/prompt/context，记录前后对比
6. [ ] 写一页**复盘**：架构选择对不对、最大瓶颈是什么、重来会怎么改

## 完成标准

- [ ] 有一份能说服别人的 Agent Design Doc（尤其是"为什么这个拓扑"）
- [ ] capstone 能端到端跑通一个真实任务
- [ ] 有 eval 数字，且能展示"迭代一轮后变好"
- [ ] 能为每个架构决策给出基于**失败模式**的理由
- [ ] 能讲清何时**不**该多 agent、致命三要素怎么隔离

## 课程意义（全课终点 🎉）
走完横向各课（`15`–`20`）并做完本模块的 capstone，你不只会"手撕原语"，更能
**像架构师一样设计 agent 系统**：从需求和失败模式出发，在 workflow/agent、
单/多 agent、上下文、harness、安全与 reasoning 之间做出有数据、有理由的取舍——
这正是"对 agent 细节技术足够把握"的标志。

> 这里是整门课（纵向 `00`–`14` + 横向 `15`–`20`）的终点。回头看，你已从
> "会用框架"走到"懂底层、会工程、能设计、敢上线"。接下来就是用真实项目持续打磨。
