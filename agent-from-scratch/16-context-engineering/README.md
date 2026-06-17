# 16 · Context Engineering（上下文工程）⭐ 前沿核心

> 横向工程能力之一，也是 2024–2025 业界的**范式转移**：从"写好一段 prompt"到
> "在每一步动态决定把什么放进有限上下文窗口"。`03-memory` 解决了"不爆窗口"，
> 本模块把它升级成一门系统学科。建议在 `03` / `04` 之后穿插，长程任务前必学。

## 学习目标

- 理解为什么领域重心从 prompt engineering 移到 **context engineering**
- 把上下文窗口当**有限预算**来管理（不是越多越好——有 context rot）
- 拆解上下文的解剖结构与四种典型失效模式
- 掌握四类核心策略：**压缩 / 写出 / 选择 / 隔离**（compress / write / select / isolate）
- 学会工具结果治理与结构化记笔记（scratchpad）
- 为 `14` 的长程任务打底：compaction + 外部记忆 + 子 agent

## 核心概念

### 1. 从 prompt 到 context 的转移
- prompt engineering：优化"写死的一段话"
- context engineering：agent 是**多轮、长程**的，每一步的上下文是**动态组装**的
  ——要工程化的是"组装什么"，而不只是"措辞"

### 2. 上下文窗口是预算，且会"腐烂"
- 有限 + 烧钱 + **注意力稀释**：塞得越多，关键信息越容易被淹没
- **context rot**：输入越长，模型有效利用能力越下降（不是线性，是退化）
- 结论：**最小充分上下文**——只放"这一步真正需要的"，而非"能塞的都塞"

### 3. 上下文的解剖
system prompt / 工具定义 / 对话历史 / 检索内容 / scratchpad 笔记 / 子任务结果。
每一类都要问：这一步**必须**有它吗？能压缩 / 外置 / 延后取吗？

### 4. 四种失效模式（来自 "How contexts fail"）
- **污染 poisoning**：错误信息进了上下文，后续被反复当真
- **分心 distraction**：无关内容挤占注意力，模型跑偏
- **混淆 confusion**：冗余/相似内容让模型分不清
- **冲突 clash**：上下文里自相矛盾的信息

### 5. 四类核心策略 ⭐ 本模块主撕
- **压缩 compress**：摘要历史、压缩工具结果（深化 `03` 的 buffer+summary，
  加上对**工具输出**的压缩——往往是 token 大头）
- **写出 write / offload**：把信息写到上下文**之外**（scratchpad / 文件系统 / DB），
  需要时再读回（呼应 `17` 文件系统作状态、`11` 沙箱）
- **选择 select / just-in-time**：**按需检索**而非一次性预载（呼应 `04`）；
  给模型"检索工具"，让它在需要时才取，而不是开局塞满
- **隔离 isolate**：用子 agent / 分区让每个上下文**聚焦**（呼应 `07`；
  Anthropic 多 agent research 系统就是靠子 agent 各自独立上下文扩容）

### 6. 工具结果治理（context engineering 的高频战场）
- 大结果（网页全文、长 JSON、文件内容）**别原样回填**
- 截断 / 分页 / 只回相关字段 / 存文件只回引用+摘要（与 `17-harness` 协同）

### 7. 结构化记笔记（structured note-taking）
- 让 agent 主动把"已知事实 / 待办 / 决定"写进**持久 scratchpad**
- 下一步从笔记读回要点，而不是重读整段历史——长程任务的关键技巧

### 8. 长程任务的上下文策略（呼应 `14`）
- compaction（到阈值就把历史压成结构化摘要）+ 外部记忆 + 子 agent 隔离
- 目标：跑数百步也不偏题、不爆预算

## 代表性参考

- **Anthropic _Effective context engineering for AI agents_**（2025，engineering 博客）⭐
  本模块"圣经"
- **Anthropic _How we built our multi-agent research system_**（2025，子 agent 上下文隔离的实战）
- LangChain _Context Engineering for Agents_（write / select / compress / isolate 框架）
- Chroma _Context Rot: How Increasing Input Tokens Impacts LLM Performance_（2025，技术报告）
- Drew Breunig _How Long Contexts Fail_ + _How to Fix Your Context_（2025，失效模式来源）
- **论文** _MemGPT: Towards LLMs as Operating Systems_（2023）`arXiv:2310.08560`（上下文分页）

## 手撕任务

> 在你 `03` 的记忆系统上继续加工。

1. [ ] 给 agent 加一个 **scratchpad**：可写/读的持久笔记（文件或内存）
2. [ ] 实现**工具结果治理**：大结果存文件，上下文只放摘要 + 引用 id
3. [ ] 实现 **compaction**：历史到阈值就压成结构化摘要（区别于纯滑窗）
4. [ ] 做一个 **context 组装器**：给定预算，从各源（历史/检索/笔记）按优先级选片段
5. [ ] 跑一个较长任务，对比"全量塞" vs "context-engineered"的 **token 数与成功率**
6. [ ]（选做）用子 agent 隔离一个子任务，观察主上下文是否更干净

## 完成标准

- [ ] 能讲清 context rot 与"最小充分上下文"原则
- [ ] 能说出四种失效模式，并在自己 agent 里指出至少一处风险
- [ ] 四类策略（压缩/写出/选择/隔离）各能举出你代码里的一个落点
- [ ] 有数据证明 context engineering 降了 token / 提了成功率
- [ ] scratchpad + compaction 让一个长任务跑得更稳

## 下一步
→ `17-harness-engineering`：上下文里"放什么"解决了，接下来是模型之外那层
**运行壳**——loop、工具运行时、文件系统、错误恢复，决定 agent 能不能真正干活。
