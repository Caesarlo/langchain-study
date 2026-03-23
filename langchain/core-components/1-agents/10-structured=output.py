
import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from pydantic import BaseModel, Field, SecretStr
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class ContactInfo(BaseModel):
    name: str = Field(description="The person's full name")
    email: str = Field(description="The person's email address")
    phone: str = Field(description="The person's phone number")


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

# agent = create_agent(
#     model,
#     # tools,
#     response_format=ToolStrategy(ContactInfo)
# )

agent = create_agent(
    model,
    # tools,
    response_format=ProviderStrategy(ContactInfo)
)


result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]
logger.info(f"Extracted Contact Info: {result['structured_response']}")
