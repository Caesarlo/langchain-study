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


@dataclass
class Context:
    user_role: str


@wrap_model_call
def context_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on Runtime Context permissions."""
    if request.runtime is None or request.runtime.context is None:
        user_role = "viewer"
    else:
        user_role = request.runtime.context.user_role

    if user_role == "admin":
        pass
    elif user_role == "editor":
        tools = [t for t in request.tools if t.name != "delete_data"]
        request = request.override(tools=tools)
    else:
        tools = [t for t in request.tools if t.name.startswith("read_")]
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
    middleware=[context_based_tools],
    context_schema=Context,
    store=InMemoryStore()
)
