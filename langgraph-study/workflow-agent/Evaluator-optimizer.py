import os

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from typing import Annotated, List, TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr
from IPython.display import display, Image

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


class State(TypedDict):
    joke: str
    topic: str
    feedback: str
    funny_or_not: str


class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description="Decide if the joke is funny or not.",
    )
    feedback: str = Field(
        description="If the joke is not funny, provide feedback on how to improve it.",
    )


evaluator = llm.with_structured_output(Feedback)


def llm_call_generator(state: State):
    """LLM generates a joke"""

    if state.get("feedback"):
        msg = llm.invoke(
            f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}"
        )
    else:
        msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}


def llm_call_evaluator(state: State):
    """LLM evaluates the joke"""

    grade = evaluator.invoke(f"Grade the joke {state['joke']}")
    return {"funny_or_not": grade.grade, "feedback": grade.feedback}


def route_joke(state: State):
    """Route back to joke generator or end based upon feedback from the evaluator"""

    if state["funny_or_not"] == "funny":
        return "Accepted"
    elif state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"


optimizer_builder = StateGraph(State)

# Add the nodes
optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_joke,
    {  # Name returned by route_joke : Name of next node to visit
        "Accepted": END,
        "Rejected + Feedback": "llm_call_generator",
    },
)


optimizer_workflow = optimizer_builder.compile()


display(Image(optimizer_workflow.get_graph().draw_mermaid_png()))


png_data = optimizer_workflow.get_graph().draw_mermaid_png()

with open("./img/Evaluator-optimizer.png", "wb") as f:
    f.write(png_data)

print("graph saved to Evaluator-optimizer.png")


state = optimizer_workflow.invoke({"topic": "Cats"})
print(state["joke"])
