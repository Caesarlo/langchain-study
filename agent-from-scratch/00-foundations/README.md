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

## 手撕产出

- `run.py`：手写的多轮对话循环（命令行连续问答，输入 `exit` 退出）。
  - 骨架就三步，**记死它**——后面所有 agent 都是它的变体：
    ```python
    messages = [system]
    while True:
        messages.append(user)        # 1. 拿到新输入
        reply = chat(messages)       # 2. 整包历史 → 模型
        messages.append(assistant)   # 3. 把模型输出存回历史
    ```
  - 验证记忆生效的方法：第二句故意用指代词（"它和压电材料的区别？"），
    模型若能接住，说明"重发历史"机制真的把上下文续上了。

### 手撕时踩过的坑（都是 role / 类型边界问题）

- `role` 必须精确匹配 `system/user/assistant/tool`，`assitant` 少个 `s` 会在**第二轮**才报错。
- `input()` 返回的是**字符串**，判断退出要用它；塞进 `messages` 前才包成 dict。
- `chat()` 返回的是 `message` 对象，存回历史要取 `reply.content` 再组装成
  `{"role":"assistant","content":...}`，别直接 append 对象或裸字符串。

## 完成标准

- [x] `common.py` 冒烟测试通过
- [x] 能解释"为什么模型无状态、记忆靠重发历史"
- [x] 手写过一个维护 `messages` 的多轮对话循环

## 下一步
→ `01-react-agent`：把"多轮对话"升级成"会自己调工具、自己决定何时停"的 agent。
本质就是在上面三步循环的第 2、3 步之间，塞一个"模型自己决定要不要调工具"的判断，
并引入第四种 role：`tool`。
