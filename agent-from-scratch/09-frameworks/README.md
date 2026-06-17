# 09 · Frameworks（主流框架对照）

> 底层撕透了，现在学框架**最快**——因为你知道每个抽象背后是什么。本模块用主流
> 框架重写前面的手撕产物，建立"何时用哪个框架"的判断。

## 学习目标

- 掌握主流 agent 框架的定位与差异
- 用框架重写 `01`/`07` 的手撕产物，对比代码量与控制力
- 理解每个框架的核心抽象（图 / handoff / role / code-agent）
- 框架选型判断力

## 核心概念 / 框架地图

### 1. LangChain `create_agent`（LangChain 1.x）
- 最新 LangChain agents 的核心是一个可配置 harness：model + tools + prompt + middleware
- agents 底层建在 LangGraph 上，继承 durable execution / HITL / persistence 等能力
- 适合：想快速搭 agent loop，又希望用 middleware 控制上下文、工具、模型、guardrails

### 2. LangGraph（你已有基础）
- 图（节点 + 边 + 状态）抽象，最主流、最灵活
- checkpointer 持久化、human-in-the-loop、streaming
- 适合：复杂、可控、生产级编排

### 3. Deep Agents
- LangChain 官方 batteries-included agent：planning、subagents、虚拟文件系统、
  context compression 等能力开箱即用
- 适合：对照 `16-context` / `17-harness` / `18-architecture`，看工业 harness 如何组合这些能力

### 4. OpenAI Responses API vs Agents SDK
- Responses API：平台托管工具与内置能力更强，适合快速接入 OpenAI 托管工具
- Agents SDK：服务端自己掌控 orchestration、tool execution、state、approval、MCP、tracing
- 学习重点：理解“平台托管编排”与“自己拥有 harness”的边界

### 5. OpenAI Agents SDK（原 Swarm 的正式版）
- 轻量，核心是 handoff + guardrails + sessions
- 适合：快速搭多 agent、OpenAI 生态

### 6. CrewAI
- 角色式（role / goal / backstory / task）
- 适合：人类直觉的"团队分工"建模

### 7. LlamaIndex
- RAG / 数据接入最强，也有 agent 能力
- 适合：知识密集型应用

### 8. smolagents（HuggingFace）
- 极简、code-agent 优先（让 LLM 写代码而非 JSON 工具调用）
- 适合：读源码学原理、轻量 code agent

### 9. AutoGen / AG2
- 多 agent 对话框架，研究氛围浓

### 10. 选型维度
- 控制力 vs 易用性、生态、可观测性、生产成熟度

## 代表性参考

- LangGraph 官方文档与 tutorials
- LangChain 1.x `create_agent` / middleware / context engineering 文档
- Deep Agents 文档与源码（根 `pyproject.toml` 已包含 `deepagents`）
- OpenAI 平台 Agents 指南（Responses API）与 OpenAI Agents SDK 文档
- OpenAI Agents SDK 文档：`https://github.com/openai/openai-agents-python`
- CrewAI：`https://github.com/crewAIInc/crewAI`
- LlamaIndex：`https://docs.llamaindex.ai/`
- smolagents：`https://github.com/huggingface/smolagents`
- LangChain Academy _Intro to LangGraph_（免费课）

## 手撕/实践任务

> 本模块是"对照"，重点在对比而非从零撕：

1. [ ] 用 LangGraph 重写 `01` 的 ReAct agent，对比你手写的循环
2. [ ] 用 LangChain `create_agent` + middleware 重写 `01`，对比手写 harness 的控制点
3. [ ] 用 Deep Agents 跑一个带文件系统 / 子 agent / context compression 的任务
4. [ ] 用 OpenAI Agents SDK（或 CrewAI）重写 `07` 的多 agent
5. [ ] 用 smolagents 跑一个 code agent，对比"写代码"vs"调 JSON 工具"
6. [ ] 写一份框架选型对照表（维度 × 框架）
7. [ ]（选做）同一任务三框架各实现一遍，比代码量/可控性/调试体验

## 完成标准

- [ ] 能用至少 2 个框架重写手撕产物
- [ ] 能对每个框架说出核心抽象和适用场景
- [ ] 面对新需求能合理选型并说明理由
- [ ] 体会到"懂底层后，框架只是加速器"

## 下一步
→ `10-mcp`：学习工具/数据接入的标准协议 MCP（Claude Code 本身就靠它）。
