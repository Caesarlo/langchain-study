"""按会话状态过滤工具：基于鉴权状态与消息数控制可用工具。"""

from typing import Any
import os
from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable
from pydantic import SecretStr

load_dotenv()


def get_tool_name(t: BaseTool | dict[str, Any]) -> str:
    # request.tools 既可能是 BaseTool，也可能是工具描述字典。
    if isinstance(t, dict):
        return str(t.get("name", ""))
    return t.name


@wrap_model_call
def state_based_tools(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on conversation State."""
    # state 通常由运行时在每轮对话中维护。
    state = request.state
    is_authenticated = state.get("authenticated", False)
    message_count = len(state.get("messages"))

    if not is_authenticated:
        # 未登录用户仅开放 public_ 前缀工具。
        tools = [t for t in request.tools if get_tool_name(t).startswith("public_")]
        request = request.override(tools=tools)
    elif message_count < 5:
        # 已登录但对话较短时，先隐藏高级检索工具。
        tools = [t for t in request.tools if get_tool_name(t) != "advanced_search"]
        request = request.override(tools=tools)

    return handler(request)


api_key = os.getenv("SILICONFLOW_API_KEY")
base_url = os.getenv("SILICONFLOW_BASE_URL")
temperature = os.getenv("SILICONFLOW_TEMPERATURE", 0.2)

if not api_key or not base_url:
    raise ValueError(
        "SILICONFLOW_API_KEY and SILICONFLOW_BASE_URL must be set in the .env file"
    )

model = ChatOpenAI(
    model="Pro/MiniMaxAI/MiniMax-M2.5",
    api_key=SecretStr(api_key),
    base_url=base_url,
    temperature=float(temperature),
    timeout=60,
)

agent = create_agent(
    model=model,
    # tools=[public_search, advanced_search, get_weather],
    middleware=[state_based_tools],
)
