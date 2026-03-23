"""静态工具示例：预先注册固定工具列表给 agent。"""

import os
from pyexpat import model
from langchain import agents
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from dotenv import load_dotenv

load_dotenv()


@tool
def search(query: str) -> str:
    # 这里是示例实现，真实场景可替换为检索 API。
    """Search for information."""
    return f"Results for: {query}"


@tool
def get_weather(location: str) -> str:
    # 这里是示例实现，真实场景可接入天气服务。
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"


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

agent = create_agent(model, tools=[search, get_weather])
# 工具集在创建时固定，不会在运行时增删。
