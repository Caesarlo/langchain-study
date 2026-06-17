# 00 · Foundations（地基）

> 你已有 LLM API 基础，本模块**极简带过**，只做一个热身、统一术语。重点确认
> `_shared/common.py` 能跑通，然后直奔 `01`。

## 学习目标

- 看清 chat completion 的本质：`messages[]` 进，一个 `message` 出
- 理解 `role`（system / user / assistant / tool）的语义与顺序约束
- 掌握采样参数（temperature / top_p / max_tokens）对 agent 的影响
- 结构化输出（JSON mode / response_format）—— agent 解析的基础
- 流式（streaming）的本质：SSE 分块

## 核心概念

### 1. Messages 与 Roles
- `system`：行为设定，放最前
- `user` / `assistant`：对话交替
- `tool`：工具执行结果回填（`01` 开始大量用）
- **关键认知**：模型是无状态的，"记忆"全靠你每轮把历史 `messages` 重新发过去

### 2. 采样参数
- `temperature`：agent 任务通常调低（0~0.3）求稳定；创意任务调高
- `max_tokens`、`top_p`、`stop`：各自作用与坑

### 3. 结构化输出
- `response_format={"type": "json_object"}` 与 schema 约束
- 为什么 agent 需要它：**把自然语言变成可程序化解析的字段**
- 与 function calling 的关系（`02` 展开）

### 4. 流式
- `stream=True` 返回 chunk 迭代器
- 对 agent UX 的意义（`13-production` 再深入）

## 代表性参考

- OpenAI 官方 _Chat Completions_ / _Text generation_ 文档
- DeepSeek 官方 API 文档：`https://api-docs.deepseek.com/`
- Anthropic _Prompt Engineering_ 指南（概念通用）

## 手撕任务

> 本模块的 client 我已写好（`_shared/common.py`），你只需：

1. 配好 `.env`（DeepSeek 官方 key），跑通 `python agent-from-scratch/_shared/common.py`
2. **热身练习**：基于 `common.py` 的 `chat()`，自己写一个 10 行的多轮对话循环
   （命令行里能连续问答，体会"手动维护 messages 历史"）

## 完成标准

- [ ] `common.py` 冒烟测试通过
- [ ] 能解释"为什么模型无状态、记忆靠重发历史"
- [ ] 手写过一个维护 `messages` 的多轮对话循环

## 下一步
→ `01-react-agent`：把"多轮对话"升级成"会自己调工具、自己决定何时停"的 agent。
