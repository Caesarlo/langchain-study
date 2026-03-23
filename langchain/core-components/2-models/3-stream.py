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
    timeout=60,
)


# for chunk in model.stream("Why do parrots have colorful feathers?"):
#     print(chunk.text, end="|", flush=True)

# for chunk in model.stream("What color is the sky?"):
#     for block in chunk.content_blocks:
#         if block["type"] == "reasoning" and (reasoning := block.get("reasoning")):
#             print(f"Reasoning: {reasoning}")
#         elif block["type"] == "tool_call_chunk":
#             print(f"Tool call chunk: {block}")
#         elif block["type"] == "text":
#             print(block["text"])
#         else:
#             ...


full = None  # None | AIMessageChunk
for chunk in model.stream("What color is the sky?"):
    full = chunk if full is None else full + chunk
    print(full.text)
