# 04 · RAG（检索增强生成）

> 让 agent 能查外部知识。从零搭 RAG pipeline，再把它**封装成一个 retrieval 工具**
> 挂给 agent —— 即 Agentic RAG（agent 自己决定何时检索、检索什么）。

## 学习目标

- 理解 embedding：文本 → 向量 → 语义相似度
- 向量库基本操作（FAISS / Chroma）
- chunking（切块）策略与权衡
- retrieval + rerank 两段式召回
- **Agentic RAG**：检索作为工具，由 agent 自主决策（对比传统固定 pipeline）
- 评估检索质量（召回了没用的 / 漏了有用的）

## 核心概念

### 1. Embedding 原理
- 把文本映射到高维向量，语义近 = 距离近（余弦相似度）
- embedding 模型选型（DeepSeek/OpenAI/开源 bge 等）

### 2. 向量库
- 建索引、增删、top-k 相似检索
- FAISS（本地、快）vs Chroma（带元数据、易用）

### 3. Chunking 策略
- 固定长度 / 按句子 / 按段落 / 重叠窗口
- chunk 太大→噪声多，太小→丢上下文
- 元数据（来源、标题）的作用

### 4. 检索 pipeline
- query → embed → 向量检索 top-k → （rerank）→ 拼进 prompt
- rerank：用更强的模型对初筛结果二次排序

### 5. Agentic RAG（本模块重点）
- 传统 RAG：固定"先检索后生成"，不管问题需不需要
- Agentic RAG：把 `retrieve(query)` 做成**工具**，agent 自己判断
  - 要不要查、查几次、换关键词重查、查完够不够
- 这正好复用 `01/02` 的 agent loop + tool

## 代表性参考

- **论文**：_Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks_
  （Lewis et al., 2020）`arXiv:2005.11401`
- **论文**：_Self-RAG_（Asai et al., 2023）`arXiv:2310.11511` —— agent 自评检索
- LlamaIndex 文档（RAG 工程化最全）`https://docs.llamaindex.ai/`
- LangChain RAG tutorial、DeepLearning.AI _Agentic RAG_ 短课

## 手撕任务

> 骨架 `rag.py` + `agentic_rag.py`，核心留 TODO：

1. [ ] 准备语料，写 chunking 函数
2. [ ] 调 embedding API，建向量索引（FAISS 或 numpy 手算余弦也行）
3. [ ] 写 `retrieve(query, k)`：返回 top-k 文本块
4. [ ] **传统 RAG**：query → retrieve → 拼 prompt → 生成
5. [ ] **Agentic RAG**：把 `retrieve` 注册成工具，挂到 `02` 的 agent 上，
       让 agent 自主决定检索，对比两者差异
6. [ ]（选做）加 rerank

## 框架对照

对比 LangChain `VectorStoreRetriever` + LangGraph 的 retrieval agent。理解框架的
`Retriever` 抽象就是你这个 `retrieve` 函数 + 向量库的封装。

## 完成标准

- [ ] 传统 RAG 能基于语料正确回答
- [ ] Agentic RAG 能自主决定"是否检索/重查"，并解释比固定 pipeline 强在哪
- [ ] 能讲清 chunking 大小、top-k 对效果的影响

## 下一步
→ `05-planning`：复杂任务需要先规划再执行，而不是走一步看一步。
