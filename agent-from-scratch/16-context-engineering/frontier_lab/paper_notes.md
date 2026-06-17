# TokenPilot — Cache-Efficient Context Management for LLM Agents

> 落到 `16-context-engineering`。本课程取其**两个可手撕机制**：双粒度上下文 + prompt-cache 友好的组装顺序。

## 原文

- 标题：*TokenPilot: Cache-Efficient Context Management for LLM Agents*（2026-06-15, arXiv）
- 作者：Buqiang Xu, Zirui Xue, Dianmou Chen, … Ningyu Zhang 等
- 一句话：双粒度上下文框架，在降低推理成本的同时**保持 prompt cache 连续性**。

## 核心机制（我们取这两点）

### 1. 双粒度上下文（dual-granularity）

每条历史信息保留两种粒度：

- **细粒度（full）**：原始完整内容（工具原始输出、完整对话轮）。
- **粗粒度（digest）**：压缩后的摘要 / 关键字段 / 引用 id。

组装上下文时，**近期 N 步用 full，更早的用 digest**。这区别于 `03` 的纯滑窗
（直接丢弃旧轮）和纯 summary（全部压成一段，丢失可回溯性）——双粒度让旧信息
"还在、但便宜"，需要时能按 id 取回 full。

### 2. prompt-cache 友好的组装顺序 ⭐ 本课程重点

DeepSeek / OpenAI 兼容接口都有 **prefix cache**：只要 messages 的**前缀逐字节相同**，
命中缓存的 token 大幅降价、降延迟。Cache 在第一个**变化的 token** 处断裂。

> 推论：把上下文里**最稳定的部分放最前**，把**每步都变的部分放最后**。

错误顺序（每步都让 cache 失效）：
```
[system] [今天的动态待办] [工具定义] [历史] ...   # 待办一变，后面全部 miss
```

正确顺序（前缀稳定，cache 命中率最大化）：
```
[system 固定] [工具定义 固定] [稳定的早期历史 digest] [近期 full 历史] [本步新增]
```

## 我们不取的部分

- 论文里与 KV-cache 底层调度 / 自建 serving 相关的工程，超出 DeepSeek 托管 API 能控的范围——
  我们只在**应用层**控制"组装顺序 + 双粒度"，命中率提升靠接口自带的 prefix cache。

## 手撕目标（见 `cache_context_builder.py`）

1. [ ] `digest()`：把一条 full 消息压成粗粒度（先用规则截断，再选做 LLM 摘要）。
2. [ ] `build_context()`：按 `[稳定前缀] + [digest 旧] + [full 近期] + [本步]` 组装。
3. [ ] `cache_prefix_len()`：估算与上一次组装结果共享的前缀长度（cache 命中近似指标）。
4. [ ] eval：对比"朴素全量 full + 不稳定顺序" vs "双粒度 + cache 友好顺序"的
   token 数、估算 cache 命中率、答案成功率。

## 验收

- [ ] 能解释 prefix cache 为什么在第一个变化 token 处断裂。
- [ ] 双粒度组装下，旧信息仍可按 id 取回 full。
- [ ] 有数据：cache 友好顺序把"估算共享前缀"显著拉长。
