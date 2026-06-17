# 07 · Multi-Agent（多智能体）

> 单 agent 塞太多职责会变笨。把任务拆给多个**专职 agent** 协作。本模块手撕
> supervisor + workers 架构，并理解各种拓扑与通信/交接（handoff）机制。

## 学习目标

- 多智能体架构：supervisor / hierarchical / network / swarm
- agent 间通信：消息传递、共享状态、handoff（控制权交接）
- 角色分工设计（planner / coder / reviewer 等）
- 路由：supervisor 如何决定派给谁
- 终止与汇总：何时结束、谁来收尾
- 多 agent 的代价（成本、延迟、失控风险）

## 核心概念

### 1. 为什么要多 agent
- 单 agent 工具/职责过多 → 选择困难、prompt 臃肿
- 专职化：每个 agent 上下文聚焦、prompt 简单、更可靠

### 2. 常见拓扑
- **Supervisor**：一个主管派活给 workers，收结果（最常用）⭐ 主撕
- **Hierarchical**：主管下面还有子主管（树形，适合大任务）
- **Network**：agent 之间自由通信（灵活但易乱）
- **Swarm / handoff**：agent 之间直接交接控制权（OpenAI Swarm 风格）

### 3. 通信机制
- 共享 messages / 共享状态对象
- handoff：A 决定"这事该 B 干"，把控制权和上下文交给 B
- 结构化交接（传什么上下文、传多少）

### 4. 路由决策
- supervisor 用 LLM 判断下一步派给哪个 worker（本质是带特殊工具的 agent）
- 可结合 `02` 的 tool：每个 worker 暴露成一个"工具"

### 5. 代价与边界
- 多 agent ≠ 更好：先问"单 agent + 多工具"是否够
- token 成倍增长、延迟叠加、调试更难

## 代表性参考

- Anthropic _Building Effective Agents_ 的 orchestrator-workers 模式
- **OpenAI Swarm / Agents SDK**：handoff 范式的代表
  `https://github.com/openai/swarm`
- LangGraph multi-agent 教程（supervisor / hierarchical / network 三种模板）
- **CrewAI**：角色式多智能体框架（role/goal/backstory 抽象）
- 你已有的 `langgraph-study/workflow-agent/Orchestrator-worker.py`（对照）

## 手撕任务

> 骨架 `multi_agent.py`，核心留 TODO：

1. [ ] 定义若干 worker agent（各自专职 + 各自工具），复用 `02` 的 agent
2. [ ] 写 `supervisor`：根据任务和已完成情况，决定派给哪个 worker（或结束）
3. [ ] 设计共享上下文/消息传递机制
4. [ ] 主循环：supervisor 路由 → worker 执行 → 回报 → 直到完成
5. [ ] 跑一个真实分工任务（如 planner+coder+reviewer 写一段代码）
6. [ ]（选做）实现 handoff 风格：worker 之间直接交接

## 框架对照

对比 LangGraph supervisor 模板 / OpenAI Agents SDK 的 handoff。理解框架的
"把 agent 当工具调"和你手撕的路由是同一回事。

## 完成标准

- [ ] supervisor 能正确路由并在完成时终止
- [ ] 多个专职 agent 协作产出比单 agent 更好的结果
- [ ] 能讲清 4 种拓扑的差异与适用场景
- [ ] 能说出多 agent 的代价、什么时候**不**该用

## 下一步
→ `08-workflows`：回头审视——很多任务根本不需要 autonomous agent，固定工作流更好。
