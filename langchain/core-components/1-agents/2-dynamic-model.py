"""动态模型示例：基于对话轮数在基础模型与高级模型之间切换。"""

import os

from langchain import agents
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from dotenv import load_dotenv
from openai import api_key
from pydantic import SecretStr


load_dotenv(dotenv_path=".env")

# 读取统一的网关与温度配置，两个模型共用。
api_key = os.getenv("SILICONFLOW_API_KEY")
base_url = os.getenv("SILICONFLOW_BASE_URL")
temperature = os.getenv("SILICONFLOW_TEMPERATURE", 0.2)

if not api_key or not base_url:
    raise ValueError(
        "SILICONFLOW_API_KEY and SILICONFLOW_BASE_URL must be set in the .env file"
    )

basic_model = ChatOpenAI(
    # 成本更低的默认模型。
    model="Qwen/Qwen2.5-72B-Instruct",
    api_key=SecretStr(api_key),
    base_url=base_url,
    temperature=float(temperature),
    timeout=60,
)

advanced_model = ChatOpenAI(
    # 质量更高的模型，复杂/长对话时使用。
    model="Pro/MiniMaxAI/MiniMax-M2.5",
    api_key=SecretStr(api_key),
    base_url=base_url,
    temperature=float(temperature),
    timeout=60,
)


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    # 通过 state 里的消息数量近似判断当前对话复杂度。
    message_count = len(request.state["messages"])

    if message_count > 10:
        model = advanced_model
    else:
        model = basic_model

    # 用 override 替换本次调用使用的模型，再交给后续处理链。
    return handler(request.override(model=model))


agent = create_agent(
    model=basic_model,
    # tools=tools
    middleware=[dynamic_model_selection],
)
