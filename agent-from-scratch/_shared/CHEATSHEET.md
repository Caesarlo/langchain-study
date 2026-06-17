# 速查 CHEATSHEET（跨模块技术参考）

> 不是教程，是**随手查**。细节以各模块 `README.md` 为准；本页只放最常忘、最易错的点。

## 1. 消息协议（messages[]，所有 agent 的地基）

```jsonc
[
  {"role": "system",    "content": "行为设定"},
  {"role": "user",      "content": "用户输入"},
  {"role": "assistant", "content": null,
   "tool_calls": [{"id": "call_1", "type": "function",
                   "function": {"name": "get_weather", "arguments": "{\"city\":\"北京\"}"}}]},
  {"role": "tool",      "tool_call_id": "call_1", "content": "北京 30°C"}
]
```

**铁律**：
- assistant 带 `tool_calls` 的那条消息要**原样 append** 回 `messages`
- 每个工具结果用 `role:tool` + **对应的 `tool_call_id`** 回填，一一对应
- 模型**无状态**：记忆全靠每轮把历史 `messages` 重新发过去
- `arguments` 是**字符串化的 JSON**，要 `json.loads` 再用

## 2. function calling / 工具 schema

```jsonc
tools = [{"type": "function", "function": {
  "name": "get_weather",
  "description": "查某城市天气。何时用：用户问天气。何时不用：非天气问题。",
  "parameters": {"type": "object",
    "properties": {"city": {"type": "string", "description": "城市名"}},
    "required": ["city"]}}}]
```
- 一轮 response 可含**多个** `tool_calls`（并行调用）→ 全部执行并回填后再进下一轮
- **description 质量 = 调用准确率**（→ `02`/`15`）

## 3. agent loop 骨架（→ `01`）

```python
while step < max_iterations:           # 兜底防失控
    msg = chat(messages, tools=tools)
    messages.append(msg)               # 原样回填
    if not msg.tool_calls:             # 退出条件：模型不再要工具
        return msg.content
    for tc in msg.tool_calls:          # 可能多个
        result = registry.execute(tc.function.name, json.loads(tc.function.arguments))
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
```

## 4. 决策表（选型）

| 问题 | 用 A | 用 B | 判据 |
|---|---|---|---|
| 流程 | **workflow**（写死路径） | **agent**（模型决定路径） | 步数/路径能否预先确定（→ `08`） |
| 结构 | 单 agent + 多工具 | 多 agent | 职责/上下文是否冲突、能否并行（→ `07`/`18`；默认单 agent） |
| 模型 | 普通模型 + CoT | reasoning 模型 | 是否复杂推理/可容忍延迟（→ `20`） |
| 检索 | 传统 RAG（固定先检索） | Agentic RAG（按需检索） | 是否每次都该检索（→ `04`） |
| 风险动作 | 直接执行 | HITL 审批 | 是否不可逆/高危（→ `13`/`19`） |

## 5. 常见失败模式 → 修法

| 症状 | 根因 | 修 |
|---|---|---|
| 循环停不下来/烧钱 | 无停止条件 | `max_iterations` + 预算 + 防循环检测（→ `17`） |
| 模型"对不上"工具结果 | `tool_call_id` 没对应 | 原样回填 + id 一一对应（→ `01`） |
| 上下文爆/越来越笨 | context rot | 压缩/写出/选择/隔离（→ `16`） |
| 工具乱调/选错 | description 差、工具太多 | 改描述、减少/分组工具、拆 agent（→ `02`/`07`） |
| 长任务跑偏 | 无规划/无记忆 | 规划（`05`）+ 记笔记（`16`）+ 子 agent（`07`） |
| 被"数据"里的指令带走 | 间接注入 | 隔离不可信内容、拆致命三要素（→ `19`） |
| 改了 prompt 没把握变好没 | 没度量 | 小 eval 集驱动（→ `12`/`15`） |

## 6. 采样 / 输出 / 流式速记（→ `00`）

- `temperature` 0~0.3 求稳定；**reasoning 模型忽略它**（→ `20`）
- 结构化输出：`response_format={"type":"json_object"}`；解析要容错
- 流式：`stream=True` 返回 chunk 迭代器（→ `13`）

## 7. 成本 / 延迟杠杆（→ `13`）

prompt caching（缓存稳定前缀）· 模型分级（简单步小模型）· 按难度路由（→ `08`/`20`）· 并发 · 超时/重试/限流

## 8. 术语表

| 术语 | 一句话 |
|---|---|
| ReAct | Reason+Act 交错的 agent 循环（`01`） |
| Plan-and-Execute | 先出计划再逐步执行（`05`） |
| ReWOO | 推理与观测解耦，省 LLM 往返（`05`） |
| ToT / GoT | 树/图式搜索推理，评分+回溯（`05`） |
| RAG / Agentic RAG | 检索增强；后者把检索做成工具由 agent 自主决定（`04`） |
| ACI | Agent–Computer Interface，为模型设计的动作接口（`17`） |
| HITL | Human-in-the-Loop，关键步人工审批（`13`） |
| MCP | Model Context Protocol，工具/数据接入标准（`10`） |
| CoT | Chain-of-Thought，链式思考（`15`/`20`） |
| context rot | 上下文越长有效利用越差（`16`） |
| 致命三要素 | 私有数据+不可信内容+对外通信 同现即高危（`19`） |
| CodeAct | 用写代码代替逐个 JSON 工具调用（`11`/`17`） |
| compaction | 历史到阈值压成结构化摘要（`16`） |
| handoff / supervisor / swarm | 多 agent 的交接/主管/蜂群拓扑（`07`） |
| LLM-as-judge | 用强模型按 rubric 打分（`12`） |
| trajectory eval | 评整个执行轨迹而非只看答案（`12`） |

## 9. 模块速览（NN → 一句话）

`00` 地基 · `01` ReAct 循环 ⭐ · `02` 工具 · `03` 记忆 · `04` RAG · `05` 规划 ·
`06` 反思 · `07` 多 agent · `08` 工作流 · `09` 框架 · `10` MCP · `11` 高级形态 ·
`12` 评估观测 · `13` 生产化 · `14` 前沿
**横向**：`15` prompt · `16` context ⭐ · `17` harness · `18` 架构+capstone · `19` 安全 · `20` reasoning
