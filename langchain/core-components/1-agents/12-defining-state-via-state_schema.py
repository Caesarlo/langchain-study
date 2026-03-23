import os
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from pydantic import SecretStr
from typing import Any
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class CustomState(AgentState):
    user_preferences: dict


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
    model,
    # tools=tools,
    state_schema=CustomState
)


result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})

logger.info(f"Agent response: {result}")
