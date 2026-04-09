# Schema for structured output
from pydantic import BaseModel, Field, SecretStr

import os
import getpass

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()


api_key = os.getenv("MODEL_API_KEY")
base_url = os.getenv("MODEL_BASE_URL")
temperature = os.getenv("MODEL_TEMPERATURE", 0.2)
model_name = os.getenv("MODEL_NAME", "gpt-5.4-mini")

if not api_key or not base_url:
    raise ValueError(
        "MODEL_API_KEY and MODEL_BASE_URL must be set in the .env file"
    )

llm = ChatOpenAI(
    model=model_name,
    api_key=SecretStr(api_key),
    base_url=base_url,
    temperature=float(temperature),
    timeout=120,
)


class SearchQuery(BaseModel):
    search_query: str = Field(
        None, description="Query that is optimized web search.")
    justification: str = Field(
        None, description="Why this query is relevant to the user's request."
    )


# Augment the LLM with schema for structured output
structured_llm = llm.with_structured_output(SearchQuery)

# Invoke the augmented LLM
output = structured_llm.invoke(
    "How does Calcium CT score relate to high cholesterol?")

print(output)
print()
print(output.search_query)
print()
print(output.justification)
print("=" * 50)
# Define a tool


def multiply(a: int, b: int) -> int:
    return a * b


# Augment the LLM with tools
llm_with_tools = llm.bind_tools([multiply])

# Invoke the LLM with input that triggers the tool call
msg = llm_with_tools.invoke("What is 2 times 3?")

# Get the tool call
msg.tool_calls

print(msg)
