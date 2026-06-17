# 12 · Evaluation & Observability（评估与可观测性）

> "测不准就改不动。" agent 是非确定性的，必须有评估和追踪才能迭代、防回归。
> 本模块给 agent 加 tracing + 一套 eval harness。

## 学习目标

- 为什么 agent 评估比普通模型难（多步、路径多样、非确定）
- trajectory evaluation：评估整个执行轨迹，不只看最终答案
- LLM-as-judge：用强模型当裁判，及其偏差与缓解
- tracing / observability：把每一步 LLM 调用和工具调用记录下来
- 离线评估集 + 回归测试

## 核心概念

### 1. 评估难在哪
- 同一任务有多条正确路径 → 不能只对答案
- 非确定性 → 要多次跑看稳定性
- 多步 → 中间步错了也可能蒙对，需看过程

### 2. 评估维度
- **最终结果**：对不对（有标准答案时）
- **轨迹**：步骤是否合理、有无多余/危险调用
- **工具使用**：调对没、参数对没
- **效率/成本**：步数、token、延迟

### 3. LLM-as-judge
- 用强模型按 rubric 打分
- 偏差：位置偏好、冗长偏好、自我偏好
- 缓解：明确 rubric、给参考答案、成对比较、多次投票

### 4. Tracing / Observability
- 记录每次 LLM 调用（输入/输出/token/耗时）+ 每次工具调用
- 工具：LangSmith、OpenTelemetry、Langfuse
- 手撕一个最小 tracer：装饰器记录每步，输出可读轨迹

### 5. 评估流程
- 攒一个测试集（输入 + 期望）→ 批量跑 → 打分 → 看回归

## 代表性参考

- LangSmith 文档（evaluation + tracing 一体）
- **OpenTelemetry**（通用可观测标准）
- Langfuse：`https://langfuse.com/`（开源 LLM 可观测）
- DeepEval / Ragas（评估框架，RAG 评估常用）
- OpenAI Evals 思路

## 手撕任务

> 骨架 `tracer.py` + `eval_harness.py`，核心留 TODO：

1. [ ] 写一个 tracer（装饰器/上下文管理器）：记录每步 LLM + 工具调用
2. [ ] 给 `01`/`05` 的 agent 接上 tracer，输出可读的执行轨迹
3. [ ] 攒一个小评估集（5-10 个任务 + 期望）
4. [ ] 写 `llm_judge(task, output, reference)`：按 rubric 打分
5. [ ] 写 eval harness：批量跑 agent → judge → 汇总报告
6. [ ]（选做）做 trajectory 评估：判断步骤是否合理而非只看答案

## 完成标准

- [ ] 能完整追踪一次 agent 运行的每一步
- [ ] eval harness 能批量评估并给出分数/报告
- [ ] 理解 LLM-as-judge 的偏差与缓解手段
- [ ] 能用评估集发现一次回归（改坏了能测出来）

## 前沿实践包（frontier_lab/）

> 把最新论文机制嫁接到本模块。总览见根目录 [`PAPERS_INTEGRATION.md`](../PAPERS_INTEGRATION.md)。

- **Agentic-Skills 方法论 + AgentFairBench**（2026, arXiv）→ `frontier_lab/skill_eval.py`
  按**技能粒度**聚合成功率（定位 agent 弱在哪类技能）+ **配置矩阵**对照（量化每次改动收益）
  + **反事实公平**子集（同任务换人口学属性看决策差异）。直接补"评估要贯穿、别等到 12"的短板。
  笔记 `frontier_lab/paper_notes.md`，验证集 `frontier_lab/eval_cases.jsonl`。

## 下一步
→ `13-production`：把 agent 做成真正能上线的服务。
