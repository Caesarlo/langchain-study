# 01 · ReAct Agent（从零手撕 agent 循环）⭐

> **整套课程最重要的一节。** 框架（LangGraph 的 `create_agent`）把这个循环藏起来了，
> 你用过但没亲手写过。本模块不碰任何框架，纯 `openai` SDK 把它撕开。

## 学习目标

- 理解 `Agent = LLM + 循环 + 工具 + 停止条件`
- 看清你旧代码 `5-1-tool-execution-loop.py` 为什么**不是** agent（只调一轮就结束）
- 手写 agent loop：think → act → observe → repeat
- 掌握**退出条件**：`message.tool_calls` 为空 = 模型认为任务完成
- 理解 ReAct 论文的 Reason + Act 交错思想

## 核心概念

### 1. 为什么需要"循环"
- 单次 function call vs 不定轮数任务（"北京和上海哪个热？"要查两次再对比）
- 你无法预知步数 → 必须 `while` → 让**模型自己决定何时停**

### 2. ReAct 模式（Reason + Act）
```
循环 {
    Reason  —— 模型思考（推理文字，可选）
    Act     —— 模型决定调哪个工具（tool_calls）
    Observe —— 你执行工具，把结果以 role=tool 喂回去
}
直到模型不再要求工具（tool_calls 为空）→ content 即最终答案 → 退出
```

### 3. 消息回填的协议（容易错的地方）
- assistant 的 `tool_calls` 消息要**原样 append** 回 messages
- 每个 tool 结果用 `{"role": "tool", "tool_call_id": ..., "content": ...}` 回填
- `tool_call_id` 必须一一对应，否则模型对不上号

### 4. 防失控
- `max_iterations` 上限（防止模型无限调工具）
- 工具异常要捕获并作为 observation 回填（让模型有机会自我纠正）

## 代表性参考

- **论文**：_ReAct: Synergizing Reasoning and Acting in Language Models_
  （Yao et al., 2022）`arXiv:2210.03629`
- **博客**：Lilian Weng _LLM Powered Autonomous Agents_ 的 "Agent loop" 部分
- **极简实现参考**：HuggingFace `smolagents` 源码（几百行看懂一个 agent）
  `https://github.com/huggingface/smolagents`
- Anthropic _Building Effective Agents_ 中 "autonomous agent" 一节

## 手撕任务

> 我会给你骨架 `react_agent.py`，**核心 `while` 循环留 TODO**，你来填：

1. 准备 2-3 个工具（如 `get_weather`、`calculator`）
2. 写 `run_agent(user_input)`：
   - [ ] 维护 messages 列表
   - [ ] `while` 循环里调用 `chat(messages, tools=...)`
   - [ ] 判断有无 `tool_calls`：有就执行并回填，无就返回 `content`
   - [ ] 正确处理 `tool_call_id` 对应
   - [ ] 加 `max_iterations` 防失控
   - [ ] 工具异常捕获并回填
3. 测试用例：单步任务 + **需要多步的任务**（验证循环真的转了多轮）

## 框架对照

完成手撕后，对比 LangGraph `create_agent` —— 它的 `tools_condition` 边、
`ToolNode`、状态图本质上就是你手写的这个循环。你会突然看懂它在干嘛。

## 完成标准

- [ ] 手写的 agent 能跑通**多步**任务（循环 ≥2 轮）
- [ ] 能讲清退出条件、`tool_call_id` 的作用
- [ ] 能指出旧代码 `5-1-tool-execution-loop.py` 的局限并说明你的版本如何解决
- [ ] 理解 `max_iterations` 为何必要

## 下一步
→ `02-tool-use`：把"手动写工具 schema"自动化，做一个 `@tool` 装饰器。
