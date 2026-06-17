# 06 · Reflection（反思与自我纠正）

> 让 agent 产出后**自己批评、再改进**，而不是一次输出就交付。这是 evaluator-optimizer
> 回路的核心，也是提升输出质量最便宜的手段之一。

## 学习目标

- self-critique：模型评判自己的输出
- self-refine：基于批评迭代改进
- Reflexion：把失败经验转成"语言记忆"指导下次尝试
- evaluator-optimizer 回路（你在 LangGraph 做过 workflow 版，这次撕底层）
- 停止条件：什么时候"够好了"

## 核心概念

### 1. 为什么需要反思
- 一次生成常有疏漏，但模型**有能力发现自己的错**
- 让它"换个角色当审稿人"往往能挑出问题

### 2. Generate → Critique → Refine 回路
- **Generate**：产出初稿
- **Critique**：用（同一个或另一个）LLM 评估，给出具体问题
- **Refine**：根据批评修改
- 循环直到通过或达上限 ⭐ 本模块主撕

### 3. Reflexion
- 任务失败后，让 agent 写一段"反思"（为什么失败、下次怎么改）
- 把反思存进记忆，下次尝试时注入 → 跨尝试学习
- 和 `03-memory` 的 episodic memory 联动

### 4. Evaluator-Optimizer
- 评估者与优化者分离（可用不同 prompt/模型）
- 评估要**可执行的标准**（rubric / 测试 / 约束），否则空转

### 5. 停止条件
- 评估通过、改进收益递减、达到 max 轮数

## 代表性参考

- **论文**：_Self-Refine: Iterative Refinement with Self-Feedback_
  （Madaan et al., 2023）`arXiv:2303.17651`
- **论文**：_Reflexion: Language Agents with Verbal Reinforcement Learning_
  （Shinn et al., 2023）`arXiv:2303.11366`
- Anthropic _Building Effective Agents_ 的 evaluator-optimizer 模式
- 你已有的 `langgraph-study/workflow-agent/Evaluator-optimizer.py`（对照）

## 手撕任务

> 骨架 `reflection.py`，核心留 TODO：

1. [ ] 写 `generate(task)`：产出初稿
2. [ ] 写 `critique(task, draft)`：返回结构化评价（是否通过 + 具体问题）
3. [ ] 写 `refine(task, draft, critique)`：改进
4. [ ] 主循环：generate→critique→refine，直到通过或达上限
5. [ ] 选一个能客观评判的任务（如"写函数并通过给定测试"）
6. [ ]（选做）Reflexion：失败反思写入记忆，跨尝试复用

## 框架对照

对比你自己写的 LangGraph `Evaluator-optimizer.py`：手撕版让你看清那条
"条件边回到 generate"本质就是这个 while 循环 + 通过判断。

## 完成标准

- [ ] 反思回路能让输出质量可见地提升（给出前后对比）
- [ ] 有明确的、可执行的评估标准（不是模糊的"好不好"）
- [ ] 正确处理停止条件，不无限循环
- [ ] 能讲清 self-refine 与 Reflexion 的区别

## 下一步
→ `07-multi-agent`：单个 agent 能力到顶后，用多个分工协作的 agent。
