"""按 Store 配置过滤工具：根据用户特性开关动态控制可用工具。"""

from gc import enable
from typing import Any
import os
from attr import dataclass
from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable
from pydantic import SecretStr

load_dotenv()


def get_tool_name(t: BaseTool | dict[str, Any]) -> str:
    # 兼容 BaseTool 与 dict 两种工具描述结构。
    if isinstance(t, dict):
        return str(t.get("name", ""))
    return t.name


@dataclass
class Context:
    # 运行时上下文中注入的用户标识。
    user_id: str


@wrap_model_call
def store_based_tools(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on Store preferences."""
    # 从运行时上下文中取用户 ID，用于读取个性化开关。
    context = request.runtime.context
    user_id = context.user_id if context else None

    if not user_id:
        return handler(request)

    # store 的 key 为 ("features",)，命名空间下按 user_id 读取配置。
    store = request.runtime.store
    if store is None:
        return handler(request)

    feature_flags = store.get(("features",), user_id)

    if feature_flags:
        # 仅保留配置中显式启用的工具。
        feature_flags_value = feature_flags.value or {}
        enabled_features = feature_flags_value.get("enabled_tools", [])
        tools = [t for t in request.tools if get_tool_name(
            t) in enabled_features]
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
    middleware=[store_based_tools],
    context_schema=Context,
    store=InMemoryStore()
)
