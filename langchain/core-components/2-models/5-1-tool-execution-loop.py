from urllib import response

from arrow import get
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

messages = [{"role": "user", "content": "What's the weather in Boston?"}]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)


for tool_call in ai_msg.tool_calls:
    tool_result = get_weather.invoke(tool_call)
    messages.append(tool_result)

final_response = model_with_tools.invoke(messages)
print(final_response.text)
