from langchain.messages import HumanMessage, AIMessage, SystemMessage
import os
from langchain.agents import AgentState
from langchain_openai import ChatOpenAI
from langchain.messages import AIMessage, HumanMessage
from pydantic import SecretStr
from langchain.agents import create_agent
from dotenv import load_dotenv
from loguru import logger


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


responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
], config={
    'max_concurrency': 5
})
for response in responses:
    print(response)
