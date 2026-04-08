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


import json


json_schema = {
    "title": "Movie",
    "description": "A movie with details",
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "The title of the movie"
        },
        "year": {
            "type": "integer",
            "description": "The year the movie was released"
        },
        "director": {
            "type": "string",
            "description": "The director of the movie"
        },
        "rating": {
            "type": "number",
            "description": "The movie's rating out of 10"
        }
    },
    "required": ["title", "year", "director", "rating"]
}


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


model_with_structure = model.with_structured_output(
    json_schema,
    method="json_schema",
)
response = model_with_structure.invoke(
    "Provide details about the movie Inception")
# Movie(title="Inception", year=2010, director="Christopher Nolan", rating=8.8)
print(response)
