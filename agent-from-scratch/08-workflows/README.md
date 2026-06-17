# 08 · Workflows vs Agents（工作流，以及何时别用 agent）

> 关键认知课：**大多数"agent"任务其实该用确定性工作流**。Anthropic 把可控的
> workflow 和 autonomous agent 明确分开。你在 LangGraph 做过这 5 种模式，本模块
> **脱离框架重撕一遍**，并建立"何时该用哪种"的判断力。

## 学习目标

- 分清 workflow（预定义路径）与 agent（模型自主决定路径）
- 手撕 Anthropic 的 5 种 workflow 模式
- 建立选型判断：什么任务用工作流、什么任务才真要 agent
- 理解"加 agent = 加不确定性 + 加成本"的工程权衡

## 核心概念

### 1. Workflow vs Agent（核心区分）
- **Workflow**：流程由你写死，LLM 只填空 → 可预测、可调试、便宜
- **Agent**：流程由模型动态决定 → 灵活但不可控、贵、难调
- **原则**：能用 workflow 解决就别上 agent

### 2. 五种 workflow 模式（逐个手撕）
1. **Prompt Chaining**：拆成串行步骤，前一步输出喂下一步
2. **Routing**：分类后分派到不同处理分支
3. **Parallelization**：并行跑多个子任务再聚合（sectioning / voting）
4. **Orchestrator-Workers**：协调者动态拆任务给工人（与 `07` 呼应）
5. **Evaluator-Optimizer**：生成-评估-改进回路（与 `06` 呼应）

### 3. 何时才升级到 agent
- 任务步骤数/路径**无法预先确定**
- 需要模型在开放环境中自主决策、动态使用工具
- 能接受更高成本和不确定性，且有护栏

### 4. 工程权衡
- 可预测性、成本、延迟、可调试性 vs 灵活性
- 先用最简方案，不够再加复杂度

## 代表性参考

- **Anthropic _Building Effective Agents_**（本模块的"圣经"，务必精读）
  `https://www.anthropic.com/engineering/building-effective-agents`
- 你已有的整套 `langgraph-study/workflow-agent/`：
  `prompt_chain.py` / `routing.py` / `parallelization.py` /
  `Orchestrator-worker.py` / `Evaluator-optimizer.py`（直接对照）

## 手撕任务

> 骨架 `workflows/`（5 个文件），核心留 TODO。**脱离 LangGraph，纯函数实现**：

1. [ ] `chaining.py`：串行链
2. [ ] `routing.py`：分类 + 分派
3. [ ] `parallel.py`：并发子任务 + 聚合（用 `asyncio` 或线程池）
4. [ ] `orchestrator.py`：动态拆分 + 分发（对照 `07`）
5. [ ] `evaluator.py`：生成-评估-改进（对照 `06`）
6. [ ] 写一份"选型决策清单"：给定任务 → 该用哪种模式 / 要不要 agent

## 框架对照

把每个手撕版和你 `langgraph-study/workflow-agent/` 里的 LangGraph 版并排看。
体会：框架给了图结构和状态管理，但本质逻辑就是你这几十行纯函数。

## 完成标准

- [ ] 5 种模式都有可运行的纯函数实现
- [ ] 能对一个新任务正确判断"用哪种模式 / 要不要 agent"并说明理由
- [ ] 能复述 Anthropic"别过度工程化、能简单就简单"的核心主张

## 下一步
→ `09-frameworks`：底层都撕透了，现在用主流框架重写，体会工程化加速。
