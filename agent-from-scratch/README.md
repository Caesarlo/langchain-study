# Agent 技术系统化学习（手撕版）

> 从零手撕每一个 agent 核心概念，再用主流框架对照。**先理解原语，再上工程化。**

---

## 这份课程是什么

一套**编号化、自包含**的 agent 学习路线。每个模块一个目录、一份 `README.md`。
课程分两层：**纵向主线 `00 → 14`**（手撕原语，按序推进）+ **横向工程能力 `15 → 20`**
（prompt / context / harness 工程、架构、安全、reasoning 设计，建议穿插学，见下方映射表）。

> **2026 校准版总控图**：见 [`LEARNING_PLAN.md`](LEARNING_PLAN.md)。根 README 负责模块地图；
> `LEARNING_PLAN.md` 负责每阶段的必做代码产物、评估指标、框架映射、capstone 和最新资料来源。
> 如果目标是“对 agent 细节技术足够把握”，请按总控图的验收包推进，而不是只读完各模块。

你可以随时关掉对话，下次打开新对话时，只要说

> “我们继续 `03-memory`” 或 “开始 `05-planning` 的第 2 节”

我就能从对应目录的 `README.md` 读到完整上下文，无缝接着教。

## 技术栈（已锁定）

| 项 | 选择 |
|---|---|
| 语言 | Python 3.13（`uv` 管理） |
| 模型 | DeepSeek 官方接口（`deepseek-chat`，OpenAI 兼容） |
| 底层 SDK | 官方 `openai` SDK —— 手撕阶段**不依赖任何 agent 框架** |
| 框架（后期对照） | LangChain `create_agent` / LangGraph / Deep Agents / OpenAI Agents SDK 等 |
| 共享代码 | `_shared/common.py`（最小 LLM client，所有模块复用） |

`.env` 需要：`MODEL_API_KEY` / `MODEL_BASE_URL` / `MODEL_NAME` / `MODEL_TEMPERATURE`

> **开始前先做环境准备**：见 [`SETUP.md`](SETUP.md)（安装依赖、配 `.env`、冒烟测试、
> 分模块额外依赖与坑）。变量模板见 [`.env.example`](.env.example)。
> 一句话上手：`uv add openai` → `cp agent-from-scratch/.env.example .env` 填 key →
> `python agent-from-scratch/_shared/common.py` 跑通即可开始 `00`。

## 教学法（每个模块都遵循）

1. **概念** —— 我讲清楚“这是什么、为什么需要、不用它会怎样”
2. **代表性参考** —— 给公认最经典的论文 / 博客 / 开源实现（附可搜索的名字与链接）
3. **我给骨架** —— 写好脚手架和工具，**核心逻辑留 `TODO` 给你**
4. **你手撕** —— 你来填核心实现
5. **我 review** —— 指出问题、对比框架做法、给出改进
6. **框架对照**（部分模块）—— 用 LangGraph 等重写，体会取舍

## 你的起点（重要）

你**已经**用 LangGraph 做过：5 种 workflow 模式、core components（agents/models/messages）。
所以你的短板不是“会不会用框架”，而是**框架帮你藏起来的底层循环**。
因此：`00` 极简带过，**真正从 `01-react-agent`（手撕 agent loop）开始发力**。

---

## 纵向主线（`00`–`14` · 手撕原语，按序推进）

| 编号 | 模块 | 一句话 | 手撕产出 |
|---|---|---|---|
| 00 | foundations | LLM API 本质、采样、结构化输出 | 最小 LLM client（已给） |
| 01 | react-agent | **从零手撕 ReAct 循环** ⭐ | 不依赖框架的 agent loop |
| 02 | tool-use | function calling、schema、并行调用 | `@tool` 装饰器 + 注册表 |
| 03 | memory | 短期/长期记忆、上下文压缩 | buffer + summary 混合记忆 |
| 04 | rag | embedding、向量库、Agentic RAG | RAG pipeline → retrieval tool |
| 05 | planning | 任务分解、Plan-and-Execute、ToT | Plan-and-Execute agent |
| 06 | reflection | self-critique、Reflexion | generate→critique→refine 回路 |
| 07 | multi-agent | supervisor/swarm、handoff | supervisor + workers 系统 |
| 08 | workflows | Anthropic 5 种模式、何时**不**用 agent | 5 种模式各撕一遍 |
| 09 | frameworks | LangGraph / OpenAI Agents SDK 等 | 用框架重写前面的手撕 |
| 10 | mcp | Model Context Protocol | MCP server + client |
| 11 | advanced-agents | code agent、computer/browser use、沙箱 | 带沙箱的 code agent |
| 12 | eval-observability | trajectory eval、LLM-as-judge、tracing | eval harness + tracing |
| 13 | production | streaming、HITL、guardrails、成本优化 | production-ready agent API |
| 14 | frontier | agentic RL、self-improving、long-horizon | 选读 + 小实验 |

## 横向工程能力（`15`–`20` · 前沿工程学科，建议穿插）

> **为什么单列**：主线 `00`–`14` 教你**手撕每个原语**；但"对 agent 细节技术足够把握"
> 还需要几门**跨模块的工程学科**——它们原先零散分布在各模块（如工具描述、上下文压缩、
> loop 防失控、护栏），现在抽出来系统化深讲。编号只是 ID，**不代表难度顺序**，按下表穿插学最佳。

| 编号 | 模块 | 一句话 | 手撕产出 |
|---|---|---|---|
| 15 | prompt-engineering | agent 的 system/工具/输出 prompt + eval 驱动迭代 | 结构化 prompt + 小 eval 集对比 |
| 16 | context-engineering ⭐ | 有限上下文里"每步放什么"：压缩/写出/选择/隔离 | scratchpad + compaction + 组装器 |
| 17 | harness-engineering | 模型之外的运行壳：ACI、控制循环、工具运行时、replay | 生产级 agent harness |
| 18 | architecture-design | 从失败模式倒推架构的决策法 + **capstone** | Agent Design Doc + 端到端系统 |
| 19 | security | 注入/越权/外泄/对抗 + 致命三要素 + 设计层防御 | agent 攻防 + 出站白名单 |
| 20 | reasoning-agents | reasoning 模型（R1/o 系列）如何改变循环/prompt/规划 | 难度路由 + interleaved thinking |

**推荐穿插顺序**（在主线哪一步插入哪门横向课）：

| 学完主线 | 就插入横向 | 原因 |
|---|---|---|
| `02-tool-use` | → `15-prompt-engineering` | 工具描述就是 prompt 工程，趁热打铁 |
| `03-memory` / `04-rag` | → `16-context-engineering` | 记忆/检索的上层是上下文工程 |
| `11-advanced-agents` | → `17-harness-engineering` | 撕过沙箱/code agent 后，理解 harness 最深 |
| `13-production` | → `19-security` | 有了护栏与生产视角，正好做攻防 |
| `05`/`06` + `15`（有对比基础） | → `20-reasoning-agents` | 体会"换模型即换设计"，收尾横向课 |
| `08-workflows`（建立判断力）→ 全部学完 | → `18-architecture-design` | 综合 + capstone，收尾整门课 |

> **缺口分析（模块层已增补）**：原 `00`–`14` 在"手撕原语"上已很完整，但 prompt /
> context / harness 工程、架构设计、安全与 reasoning 时代设计只是零散提及。
> 现已全部补成独立模块：✅ `15`–`18`（v2）、✅ `19-security`（v3）、
> ✅ `20-reasoning-agents`（v4）。模块地图覆盖
> 原语 → 工程（prompt/context/harness）→ 架构 → 安全 → reasoning 全链路。
> 但要达到“细节足够把握”，仍需补齐每节的**代码产物、测试/eval、运行轨迹、失败复盘**；
> 这些统一放在 [`LEARNING_PLAN.md`](LEARNING_PLAN.md)。

## 进度追踪

> 每完成一个模块，把状态改成 ✅；进行中改 🔄。

- [x] 00 · foundations ✅
- [x] 01 · react-agent ⭐ ✅
- [ ] 02 · tool-use 🔄（@tool 装饰器已撕，registry + agent 集成进行中）
- [ ] 03 · memory
- [ ] 04 · rag
- [ ] 05 · planning
- [ ] 06 · reflection
- [ ] 07 · multi-agent
- [ ] 08 · workflows
- [ ] 09 · frameworks
- [ ] 10 · mcp
- [ ] 11 · advanced-agents
- [ ] 12 · eval-observability
- [ ] 13 · production
- [ ] 14 · frontier

**横向工程能力（穿插）**

- [ ] 15 · prompt-engineering（建议 `02` 后）
- [ ] 16 · context-engineering ⭐（建议 `03`/`04` 后）
- [ ] 17 · harness-engineering（建议 `11` 后）
- [ ] 19 · security（建议 `13` 后）
- [ ] 20 · reasoning-agents（建议 `05`/`06`/`15` 后）
- [ ] 18 · architecture-design + capstone（建议最后做，收尾整门课）

---

## 必读三篇（开始前先建立全局观）

1. **Lilian Weng — _LLM Powered Autonomous Agents_**（2023）
   agent 领域最经典综述。`https://lilianweng.github.io/posts/2023-06-23-agent/`
2. **Anthropic — _Building Effective Agents_**（2024）
   workflow vs agent 的权威划分。`https://www.anthropic.com/engineering/building-effective-agents`
3. **OpenAI — _A Practical Guide to Building Agents_**（PDF，搜标题即得）

## 前沿工程必读（横向工程课的核心来源，新增）

4. **Anthropic — _Effective context engineering for AI agents_**（2025）
   上下文工程的范式之作，`16` 的圣经。
5. **Anthropic — _How we built our multi-agent research system_**（2025）
   子 agent 上下文隔离 + 多 agent 架构的实战权衡（喂 `16`/`18`）。
6. **Cognition — _Don't Build Multi-Agents_**（Walden Yan, 2025）
   多 agent 的反方，架构设计必须听的另一面（`18`）。
7. **_SWE-agent: Agent–Computer Interfaces…_**（`arXiv:2405.15793`）
   "harness/ACI 决定能力上限"的实证（`17`）。
8. **_12-Factor Agents_**（HumanLayer）`github.com/humanlayer/12-factor-agents`
   把 agent 当软件工程的 12 条原则（`17`/`18`）。
9. **Simon Willison — _The lethal trifecta for AI agents_**（2025）
   私有数据 + 不可信内容 + 对外通信 = 高危，agent 安全的核心心智模型（`19`）。

## 如何使用本课程（给未来的自己 / 给 AI 的提示）

> 新对话开场白模板：
> “我在学 `agent-from-scratch` 课程，技术栈见根 README。
> 现在继续 `<模块编号>`，请读该模块的 README.md 接着教。”
>
> 模块编号可以是纵向主线（`00`–`14`）或横向工程课（`15`–`20`）。
> 不确定下一步学哪个时，让我看根 README 的**穿插顺序表**帮你定位。
