from urllib import response

from langchain.messages import HumanMessage, AIMessage, SystemMessage
import os
from langchain.agents import AgentState
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.messages import AIMessage, HumanMessage
from pydantic import SecretStr
from langchain.agents import create_agent
from dotenv import load_dotenv
from loguru import logger


@tool
def get_weather(location: str):
    """Get the weather at a location."""
    return f"It's sunny in {location}."


load_dotenv()

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
    timeout=180,
)


model_with_tools = model.bind_tools([get_weather])


response = model_with_tools.invoke("What's the weather like in Boston?")
for tool_call in response.tool_calls:
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
