# 14 · Frontier（前沿方向，选读）

> 主线学完后的延伸。这些方向变动快、偏研究，**按兴趣挑 1-2 个深入**即可，
> 不必全做。重点是看懂趋势、能读论文、做小实验。

## 学习目标

- 了解 agent 领域的活跃前沿
- 能读懂相关论文、判断哪些值得跟进
- 在选定方向上做一个小实验

## 前沿方向

### 1. Agentic RL（用强化学习训练 agent）
- 不再只靠 prompt，而是用 RL 训练模型的 agent 行为
- 关键词：RLHF 之后的 agent 训练、工具使用的 RL、过程奖励
- 代表：各家 reasoning/agent 模型的训练方法（DeepSeek-R1 等的思路）

### 2. Self-Improving Agents（自我改进）
- agent 改自己的 prompt / 工具 / 策略
- 关键词：自动 prompt 优化（DSPy）、自生成训练数据、自我对弈
- 代表：DSPy `https://github.com/stanfordnlp/dspy`

### 3. Long-Horizon Agents（长程任务）
- 跨小时/天、数百步的任务（写完整项目、长流程自动化）
- 难点：记忆、纠错累积、不偏离目标、成本控制
- 关键词：任务分解 + 持久记忆 + 自我监督

### 4. 进阶记忆系统
- 分层/可学习记忆，超越 `03` 的 buffer+summary
- 代表：MemGPT、Mem0、Generative Agents 的 memory stream + reflection

### 5. 多 agent 涌现与协议
- 大规模 agent 协作、agent 间标准通信协议
- agent 经济 / agent 市场等设想

### 6. 评估前沿
- agent benchmark：SWE-bench、WebArena、GAIA、AgentBench 等
- 用真实任务衡量 agent 能力

## 代表性参考

- **Benchmarks**：SWE-bench、GAIA、WebArena、AgentBench（搜名字即得论文/榜单）
- DSPy（自我优化）：`https://github.com/stanfordnlp/dspy`
- MemGPT / Mem0（进阶记忆）
- _Generative Agents_（2304.03442，记忆+反思的经典）
- 关注：Anthropic / OpenAI / DeepSeek 的 agent 相关研究博客

## 实践任务（选做）

1. [ ] 挑一个方向，精读 1-2 篇代表论文，写一页笔记
2. [ ] 在某个 benchmark（如 SWE-bench Lite 子集）上跑你的 agent，看分数
3. [ ]（选做）用 DSPy 自动优化前面某个模块的 prompt，对比手写效果
4. [ ]（选做）用 Mem0 等替换 `03` 的记忆，对比长程表现

## 完成标准

- [ ] 能讲清至少 2 个前沿方向在解决什么问题
- [ ] 在一个方向上做过小实验或精读过论文
- [ ] 对"agent 下一步往哪走"有自己的判断

## 纵向主线完结 🎉

走完 `00 → 14`（纵向主线），你将拥有：
- 不依赖框架、从零手撕每个核心概念的能力
- 对主流框架的底层理解和选型判断
- 一个 production-ready 的 agent 服务
- 读懂前沿、持续跟进的基础

## 下一步
→ **横向工程能力 `15`–`20`**：把"会手撕"升级成"会工程化、会设计、会防护"。
若你已按根 README 的穿插表边学主线边插横向课，这里只剩没插完的几门；
否则现在依次补：`15` prompt → `16` context → `17` harness → `19` security →
`20` reasoning →（最后）`18` architecture + **capstone**。整门课的真正终点是 `18` 的 capstone。
