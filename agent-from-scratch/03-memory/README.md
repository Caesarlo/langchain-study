# 03 · Memory（记忆）

> 你的 agent 现在是"金鱼"——每次对话从零开始，且对话一长就撑爆上下文窗口。
> 本模块给它装短期 + 长期记忆，并解决上下文超长问题。

## 学习目标

- 区分记忆类型：working / episodic / semantic（工作 / 情景 / 语义）
- 短期记忆：对话历史管理、上下文窗口预算
- 上下文压缩：滑动窗口、摘要（summary）、混合策略
- 长期记忆：向量检索召回（为 `04-rag` 铺路）
- token 预算管理：什么时候该裁剪、该摘要

## 核心概念

### 1. 记忆分类（借认知科学的说法）
- **working memory**：当前任务的临时上下文（就是 messages）
- **episodic memory**：过往交互记录（"上次你说过…"）
- **semantic memory**：抽象知识/事实（存向量库，按需召回）

### 2. 短期记忆与上下文窗口
- 模型上下文有上限，对话越长 token 越多 → 必须管理
- 朴素方案：全量发 → 会爆 + 越来越贵

### 3. 压缩策略
- **滑动窗口**：只留最近 N 轮（简单但丢早期信息）
- **摘要**：把旧对话用 LLM 压成一段 summary
- **混合（buffer + summary）**：近期保原文 + 远期存摘要 ⭐ 本模块主撕
- 触发时机：按 token 数 / 轮数阈值

### 4. 长期记忆（引子）
- 把重要信息写入向量库，下次按相似度召回
- 与短期记忆的协作：先查长期，注入到当前上下文
- 细节在 `04-rag` 展开

## 代表性参考

- Lilian Weng _Agent_ 博客的 "Memory" 一节（working/long-term 分类）
- **论文**：_Generative Agents: Interactive Simulacra of Human Behavior_
  （Park et al., 2023）`arXiv:2304.03442` —— memory stream + 反思 + 检索经典
- LangChain `ConversationSummaryBufferMemory`（混合策略的工业实现）
- `MemGPT` / `Mem0` 项目（分层记忆，进阶可看）

## 手撕任务

> 骨架 `memory.py`，核心留 TODO：

1. [ ] 写 `ConversationBuffer`：维护 messages，按 token/轮数判断是否超限
2. [ ] 写 `summarize(old_messages)`：用 LLM 把旧对话压成摘要
3. [ ] 写 `HybridMemory`：近期原文 + 远期摘要，组装成发给模型的 messages
4. [ ] 接到 `02` 的 agent 上，验证长对话不爆窗口且记得早期信息
5. [ ]（选做）极简长期记忆：dict 存储 + 关键词召回（向量版留给 `04`）

## 框架对照

对比 LangGraph 的 `checkpointer`（持久化状态）和 LangChain memory 类。你会理解
"短期记忆 = 状态持久化 + 上下文裁剪"这件事框架是怎么封装的。

## 完成标准

- [ ] 长对话（超过窗口）能自动压缩、不报错
- [ ] 压缩后 agent 仍记得早期关键信息（摘要生效）
- [ ] 能讲清三种记忆类型的区别和各自存哪
- [ ] 理解 token 预算与压缩触发时机

## 下一步
→ `04-rag`：把长期记忆做成真正的向量检索，并升级为 Agentic RAG。
