# 20 · Reasoning Agents（推理模型时代的 agent 设计）

> 横向能力的收官一课，也是 2025 的**大变量**。reasoning 模型（DeepSeek-R1、OpenAI o
> 系列、Claude extended thinking、Gemini thinking）自带"思考"，重塑了 agent 的 prompt、
> 循环、规划与成本结构。本模块讲：它如何改变设计、何时用 / 不用、怎么用好。
> 建议主线学完、有了对比基础后学（呼应 `05` 规划、`06` 反思、`15` prompt）。

## 学习目标

- 理解 reasoning 模型 vs 普通模型的本质区别（test-time compute / 内化的 CoT）
- 看清它如何改变 prompt 策略（呼应 `15`：别再硬塞"一步步想"）
- 理解它对 planning（`05`）、reflection（`06`）的冲击——部分脚手架被**内化**
- 掌握 reasoning 模型的循环：interleaved thinking、思考预算、旧思考的上下文清理
- 算清成本/延迟新账：reasoning token 又多又贵又慢，按难度路由
- 能判断一个任务**该不该**上 reasoning 模型

## 核心概念

### 1. 什么是 reasoning 模型
- 训练（常用 RL，呼应 `14` agentic RL）让模型在答前生成长链"思考"
- **test-time compute**：花更多推理算力换更好答案（不止靠参数量）
- 代表：DeepSeek-R1、OpenAI o1/o3、Claude extended thinking、Gemini thinking
- 与手动 CoT 的区别：思考是**内化**的、更长更结构化，不靠你 prompt 触发

### 2. 对 prompt 的影响（呼应 `15`）⭐
- **别再说"一步步想"**：模型自带，硬塞反而干扰它自己的推理
- 给**目标、约束、成功标准**，**少干预过程**——像给一个聪明下属交代任务
- few-shot 收益变弱甚至有害；倾向 zero-shot + 清晰指令
- 这正是 `15` 里"reasoning 模型要换 prompt 策略"的展开

### 3. 对 planning 的影响（呼应 `05`）
- 一些显式规划脚手架（Plan-and-Execute）被模型**内化**，简单多步可直接交给它想
- 但**长程、多步仍需外置规划**——reasoning ≠ 无限上下文 / 无限可靠
- 取舍：简单交给模型自规划；复杂长程仍要显式分解 + 外部记忆（呼应 `16`）

### 4. 对 reflection 的影响（呼应 `06`）
- 模型在思考链里已做了部分 self-check，外置反思的边际收益下降
- 仍有用，但可从"每步都反思"改为"只在关键节点反思"

### 5. reasoning 模型的循环设计
- **interleaved thinking**：在工具调用之间也思考（思考→调工具→看结果→再思考）
- **思考预算**（reasoning effort / thinking budget）：可调，质量 vs 成本/延迟
- **旧思考的处理**：通常对用户隐藏；上下文里一般**丢弃历史思考**以省 token
  （呼应 `16` 上下文治理）——保留全部思考会迅速爆预算

### 6. 成本 / 延迟新权衡（呼应 `13`）
- reasoning token 多、贵、慢，首 token 延迟高
- **模型分级升级版**：简单步普通模型，难步 reasoning 模型
- 别给所有步都开 reasoning——按**难度路由**（呼应 `08` routing）

### 7. 何时用 / 不用
- **用**：复杂推理、数学/代码、多约束权衡、难规划、容忍延迟
- **不用**：简单抽取/分类/格式化、延迟敏感、需要稳定可控流程
  （这些用 workflow + 普通模型，呼应 `08`）

### 8. 与前面模块的关系
- reasoning 不是取代脚手架，而是**移动了"哪些外置、哪些内化"的边界**
- harness（`17`）、context（`16`）、架构（`18`）都要据此重新权衡

## 代表性参考

- **论文** _DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL_（2025）`arXiv:2501.12948` ⭐
- OpenAI _Reasoning_ 文档（platform docs：reasoning effort、最佳实践、何时用 o 系列）
- Anthropic _Extended thinking_ + _Interleaved thinking_ 文档（工具间思考、思考预算）
- **论文** _Scaling LLM Test-Time Compute Optimally…_（Snell et al., 2024）`arXiv:2408.03314`
- 呼应 `14` agentic RL（reasoning 模型多用 RL 训练）

## 手撕任务

> 用你已有的 agent 做对照实验，体会"换模型即换设计"。

1. [ ] 同一任务：**普通模型 + 手写 CoT** vs **reasoning 模型**，对比质量/成本/延迟
2. [ ] 改造 `15` 的 prompt：去掉"一步步想"等冗余引导，适配 reasoning 模型，重测
3. [ ] 实现**按难度路由**（呼应 `08`）：简单步普通模型、难步 reasoning 模型
4. [ ] 循环里试 **interleaved thinking**，并做**旧思考的上下文清理**（呼应 `16`）
5. [ ]（选做）调 thinking budget，画一条"质量 vs 成本"曲线

## 完成标准

- [ ] 能讲清 reasoning 模型与"普通模型 + CoT"的本质区别
- [ ] 能说出它如何改变 `15`/`05`/`06` 的设计
- [ ] 实现过按难度路由，并有成本/延迟数据支撑取舍
- [ ] 理解 interleaved thinking 与思考的上下文管理
- [ ] 能对一个新任务判断"要不要 reasoning 模型"并说明理由

## 下一步
→ `18-architecture-design`（横向课的最后一站 + **capstone**）：模型、prompt、
context、harness、安全、reasoning 都齐了，最后把这一切组装成一个真实 agent 系统。
走到这里，你已拥有**从原语到前沿、从工程到架构到安全、再到 reasoning 时代设计**
的完整地图——这正是"对 agent 细节技术足够把握"的标志。
