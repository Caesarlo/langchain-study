"""
共享的最小 LLM client —— 所有手撕课程复用这一个文件。

设计原则:
- 只依赖官方 openai SDK,不依赖任何 agent 框架(langchain/langgraph 一概不用)。
- DeepSeek / 通义 / SiliconFlow 等都兼容 OpenAI 接口,改 .env 即可切换。
- 这一层故意做得"薄",这样后面每一课的 agent 逻辑都暴露在你眼前,没有黑箱。

.env 需要:
    MODEL_API_KEY=sk-xxx
    MODEL_BASE_URL=https://api.deepseek.com
    MODEL_NAME=deepseek-chat
    MODEL_TEMPERATURE=0.2

完整环境准备见 ../SETUP.md；变量模板见 ../.env.example。
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_API_KEY = os.getenv("MODEL_API_KEY")
_BASE_URL = os.getenv("MODEL_BASE_URL")
_MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")
_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.2"))

if not _API_KEY or not _BASE_URL:
    raise ValueError("请在 .env 中设置 MODEL_API_KEY 和 MODEL_BASE_URL")

# 全局唯一的 client
client = OpenAI(api_key=_API_KEY, base_url=_BASE_URL)


def chat(messages: list[dict], tools: list[dict] | None = None, **kwargs):
    """
    最小封装:发一轮对话,拿回完整的 message 对象(不是只拿 .content)。

    为什么返回整个 message 而不是 message.content?
    —— 因为 agent 需要看 message.tool_calls 来决定"要不要调工具"。
    这是 agent 和普通 chatbot 的第一个分水岭。

    返回: openai 的 ChatCompletionMessage 对象,含 .content 和 .tool_calls
    """
    resp = client.chat.completions.create(
        model=kwargs.pop("model", _MODEL_NAME),
        messages=messages,
        tools=tools,
        temperature=kwargs.pop("temperature", _TEMPERATURE),
        **kwargs,
    )
    return resp.choices[0].message


if __name__ == "__main__":
    # 冒烟测试:确认 .env 配好了、模型能通
    msg = chat([{"role": "user", "content": "用一句话回答:你是谁?"}])
    print("模型响应:", msg.content)
