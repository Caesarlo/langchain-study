from ast import Pass
from itertools import chain
import os

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

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


class State(TypedDict):
    topic: str
    joke: str
    imporoved_joke: str
    final_joke: str


def generate_joke(state: State):
    """First LLM call to generate initial joke"""
    msg = llm.invoke(f"Write a short joke about {state['topic']}")

    return {"joke": msg.content}


def check_punchline(state: State):
    """Gate function to check if the joke has a punchline"""

    if "?" in state['joke'] or '!' in state['joke']:
        return "Pass"
    return "Fail"


def imporove_joke(state: State):
    """Second LLM call to improve the joke"""

    msg = llm.invoke(
        f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}


def polish_joke(state: State):
    """Third LLM call for final polish"""
    msg = llm.invoke(
        f"Add a surprising twist to this joke: {state['improved_joke']}")
    return {"final_joke": msg.content}


workflow = StateGraph(State)

workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", imporove_joke)
workflow.add_node("polish_joke", polish_joke)

workflow.add_edge(START, "generate_joke")
workflow.add_conditional_edges("generate_joke", check_punchline, {
                               "Fail": "improve_joke", "Pass": END})
workflow.add_edge("improve_joke", "polish_joke")
workflow.add_edge("polish_joke", END)


chain = workflow.compile()


display(Image(chain.get_graph().draw_mermaid_png()))


state = chain.invoke({"topic": "cats"})
print("Initial joke:")
print(state["joke"])
print("\n--- --- ---\n")
if "improved_joke" in state:
    print("Improved joke:")
    print(state["improved_joke"])
    print("\n--- --- ---\n")

    print("Final joke:")
    print(state["final_joke"])
else:
    print("Final joke:")
    print(state["joke"])
