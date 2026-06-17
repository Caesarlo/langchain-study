# 05 · Planning（规划）

> ReAct 是"走一步看一步"，复杂任务会迷路。本模块让 agent **先规划再执行**，
> 并了解更高级的搜索式推理（ToT/GoT）。

## 学习目标

- 任务分解（task decomposition）：大任务拆成有序子任务
- Plan-and-Execute 模式：先出计划，再逐步执行 ⭐ 主撕
- ReWOO：把推理与观测解耦，减少 LLM 调用
- LLM Compiler：并行化子任务的 DAG 调度
- Tree of Thoughts / Graph of Thoughts：搜索式多路径推理
- self-consistency：多次采样投票

## 核心概念

### 1. 为什么 ReAct 不够
- 走一步看一步 → 长任务容易跑偏、重复、忘目标
- 没有全局规划 → 难以并行、难以回溯

### 2. Plan-and-Execute
- **Planner**：先把任务拆成步骤列表（plan）
- **Executor**：逐步执行，每步可调工具（复用 `01` 的循环）
- 可选 **Replan**：执行中发现计划不对，重新规划
- 对比 ReAct：规划与执行**分离**，更可控

### 3. ReWOO（Reasoning WithOut Observation）
- 一次性生成完整计划（含变量占位），再统一执行
- 减少 LLM 往返、省 token

### 4. Tree/Graph of Thoughts
- ToT：把推理展开成树，对多个分支评分、剪枝、回溯
- GoT：推广成图，支持合并不同思路
- 适合解谜、规划类硬问题，成本高

### 5. self-consistency
- 同一问题多次采样，对答案投票，提升稳定性

## 代表性参考

- **论文**：_Plan-and-Solve Prompting_（Wang et al., 2023）`arXiv:2305.04091`
- **论文**：_ReWOO_（Xu et al., 2023）`arXiv:2305.18323`
- **论文**：_Tree of Thoughts_（Yao et al., 2023）`arXiv:2305.10601`
- **论文**：_Graph of Thoughts_（Besta et al., 2023）`arXiv:2308.09687`
- **论文**：_LLM Compiler_（Kim et al., 2023）`arXiv:2312.04511`
- LangGraph 官方 `plan-and-execute` / `rewoo` / `llm-compiler` 教程

## 手撕任务

> 骨架 `plan_and_execute.py`，核心留 TODO：

1. [ ] 写 `planner(task)`：让 LLM 输出结构化步骤列表（JSON）
2. [ ] 写 `executor(step, context)`：执行单步（可调工具），返回结果
3. [ ] 主循环：按 plan 逐步执行，累积上下文
4. [ ] 加 `replan(task, done, remaining)`：执行偏离时重新规划
5. [ ] 对比同一任务用 `01` 的纯 ReAct vs Plan-and-Execute 的表现
6. [ ]（选做）实现一个最小 ToT 解决某个搜索类小问题

## 框架对照

对比 LangGraph 的 plan-and-execute 模板。你会看到它的 plan 节点、执行子图、
replan 条件边就是你手撕的三块。

## 完成标准

- [ ] Plan-and-Execute 能完成一个多步任务并展示中间计划
- [ ] replan 在计划失效时能触发
- [ ] 能讲清 ReAct / Plan-Execute / ReWOO 各自适用场景
- [ ] 理解 ToT 的搜索-评分-回溯思想（不要求完整实现）

## 下一步
→ `06-reflection`：让 agent 学会自我批评和修正，质量再上一个台阶。
