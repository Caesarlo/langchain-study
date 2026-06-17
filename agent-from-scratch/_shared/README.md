# \_shared（跨模块共享代码）

所有手撕模块复用这里的代码，避免每课重复造轮子。

## 文件

- `common.py` —— 最小 LLM client（只依赖官方 `openai` SDK）
  - `client`：全局 OpenAI client（读 `.env` 的 `MODEL_*` 配置）
  - `chat(messages, tools=None, **kwargs)`：发一轮对话，**返回整个 message 对象**
    （含 `.content` 和 `.tool_calls`）——这是 agent 能判断"要不要调工具"的关键

## 使用

各模块代码里这样导入（按你的运行方式调整路径/sys.path）：

```python
from _shared.common import chat, client
```

## .env 约定

```
MODEL_API_KEY=sk-xxx                       # DeepSeek 官方 key
MODEL_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
MODEL_TEMPERATURE=0.2
```

## 冒烟测试

```
python agent-from-scratch/_shared/common.py
```

打印出模型回答即说明 client 通了。
