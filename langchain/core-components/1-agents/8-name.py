from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest


agent = create_agent(
    # model,
    # tools,
    name="research_assistant"
)
