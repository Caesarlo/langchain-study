from IPython.display import Markdown
import operator
import os

from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END

from typing import Annotated, List, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

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


class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )


class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report.",
    )


class State(TypedDict):
    topic: str
    sections: list[Section]
    completed_sections: Annotated[list, operator.add]
    final_report: str


class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]


planner = llm.with_structured_output(Sections)


def orchestrator(state: State):
    """Orchestrator that generates a plan for the report"""

    report_sections = planner.invoke(
        [
            SystemMessage(content="Generate a plan for the report."),
            HumanMessage(
                content=f"Here is the report topic: {state['topic']}"),
        ]
    )

    return {"sections": report_sections.sections}


def llm_call(state: WorkerState):
    """Worker writes a section of the report"""

    section = llm.invoke(
        [
            SystemMessage(
                content="Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."
            ),
            HumanMessage(
                content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
            ),
        ]
    )

    return {"completed_sections": [section.content]}


def synthesizer(state: State):
    """Synthesize full report from sections"""

    completed_sections = state['completed_sections']

    completed_report_sections = "\n\n---\n\n".join(completed_sections)

    return {"final_report": completed_report_sections}


def assign_workers(state: State):
    """Assign a worker to each section in the plan"""

    return [Send("llm_call", {"section": s}) for s in state["sections"]]


orchestrator_worker_builder = StateGraph(State)


orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges(
    "orchestrator",
    assign_workers,
    ['llm_call']
)
orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

orchestrator_worker = orchestrator_worker_builder.compile()

png_data = orchestrator_worker.get_graph().draw_mermaid_png()

with open("./img/Creating-workers-in-LangGraph.png", "wb") as f:
    f.write(png_data)

print("graph saved to Creating-workers-in-LangGraph.png")

state = orchestrator_worker.invoke(
    {"topic": "Create a report on LLM scaling laws"})

Markdown(state['final_report'])

print(state['final_report'])
