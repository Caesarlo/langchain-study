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


def chat_stream(messages: list[dict], **kwargs):
    """
    chat() 的流式版:一个 generator,逐块 yield 文本增量(delta)。

    和 chat() 的本质区别:
    - chat() 等模型【全部生成完】才一次性返回整个 message 对象。
    - chat_stream() 底层走 SSE,模型每生成一小块就 yield 一块文本,
      调用方能边收边显示("打字机效果")。整段回答 = 所有 delta 拼起来。

    为什么这里【不】print,而是 yield?
    —— 公共地基不该假设调用方一定想打印到终端。把"怎么消费"的权力交出去:
       调用方自己 for delta in chat_stream(...),决定是打印、写文件还是转发前端。
       这就是"数据生产"与"数据消费"的分离。

    用法:
        full = ""
        for delta in chat_stream(messages):
            print(delta, end="", flush=True)
            full += delta

    yield: str,每次一块文本增量(已过滤掉 content 为 None 的首/尾块)
    """
    resp = client.chat.completions.create(
        model=kwargs.pop("model", _MODEL_NAME),
        messages=messages,
        temperature=kwargs.pop("temperature", _TEMPERATURE),
        stream=True
    )

    for chunk in resp:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


if __name__ == "__main__":
    # 冒烟测试:确认 .env 配好了、模型能通
    msg = chat([{"role": "user", "content": "用一句话回答:你是谁?"}])
    print("模型响应:", msg.content)
