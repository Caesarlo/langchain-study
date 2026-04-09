import os

from typing import Literal, TypedDict
from dotenv import load_dotenv
from pydantic import SecretStr

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt, Command, RetryPolicy
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str


class EmailAgentState(TypedDict):
    email_content: str
    sender_email: str
    email_id: str

    classification: EmailClassification | None

    search_results: list[str] | None
    customer_history: dict | None

    draft_response: str | None
    messages: list[str] | None


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


def read_email(state: EmailAgentState) -> dict:
    """Extract and parse email content"""
    return {
        "messages": [HumanMessage(content=f"Processing email: {state['email_content']}")]
    }


def classify_intent(state: EmailAgentState) -> Command[Literal["search_documentation", "human_review", "draft_response", "bug_tracking"]]:
    """Use LLM to classify email intent and urgency, then route accordingly"""

    classification_prompt = f"""
Analyze this customer email and classify it.

Email: {state['email_content']}
From: {state['sender_email']}
"""

    structured_llm = llm.with_structured_output(EmailClassification)
    try:
        classification = structured_llm.invoke(classification_prompt)
    except Exception:
        classification = {
            "intent": "complex",
            "urgency": "medium",
            "topic": "unparsed",
            "summary": state["email_content"][:200],
        }

    if classification["intent"] == "billing" or classification["urgency"] == "critical":
        goto = "human_review"
    elif classification["intent"] in ["question", "feature"]:
        goto = "search_documentation"
    elif classification["intent"] == "bug":
        goto = "bug_tracking"
    else:
        goto = "draft_response"

    return Command(
        update={"classification": classification},
        goto=goto
    )


def search_documentation(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """Search knowledge base for relevant information"""

    classification = state.get("classification", {})

    query = f"{classification.get('intent', '')} {classification.get('topic', '')}"

    try:
        search_results = [
            "Reset password via Settings > Security > Change Password",
            "Password must be at least 12 characters",
            "Include uppercase, lowercase, numbers, and symbols"
        ]
    except Exception as e:
        search_results = [f"Search temporarily unavailable: {str(e)}"]

    return Command(
        update={"search_results": search_results},
        goto="draft_response"
    )


def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """Create or update bug tracking ticket"""

    ticket_id = "BUG-12345"

    return Command(
        update={"search_results": [f"Bug ticket {ticket_id} created"]},
        goto="draft_response"
    )


def draft_response(state: EmailAgentState) -> Command[Literal["human_review", "send_reply"]]:
    """Generate response using context and route based on quality"""

    classification = state.get("classification", {})

    context_sections = []

    if state.get('search_results'):
        formatted_docs = "\n".join(
            [f"- {doc}" for doc in state['search_results']])
        context_sections.append(f"Relevant documentation:\n{formatted_docs}")

    if state.get("customer_history"):
        context_sections.append(
            f"Customer tier: {state['customer_history'].get('tier', 'standard')}"
        )

    draft_prompt = f"""
    Draft a response to this customer email:
    {state['email_content']}

    Email intent: {classification.get('intent', 'unknown')}
    Urgency level: {classification.get('urgency', 'medium')}

    {chr(10).join(context_sections)}

    Guidelines:
    - Be professional and helpful
    - Address their specific concern
    - Use the provided documentation when relevant
    """

    response = llm.invoke(draft_prompt)

    needs_review = (
        classification.get('urgency') in [
            'high', 'critical'] or classification.get('intent') == 'complex'
    )

    goto = "human_review"if needs_review else "send_reply"

    return Command(
        # Store only the raw response
        update={"draft_response": response.content},
        goto=goto
    )


def human_review(state: EmailAgentState) -> Command[Literal["send_reply", END]]:
    """Pause for human review using interrupt and route based on decision"""

    classification = state.get('classification', {})

    human_decision: dict = interrupt({
        "email_id": state.get('email_id', ''),
        "original_email": state.get('email_content', ''),
        "draft_response": state.get('draft_response', ''),
        "urgency": classification.get('urgency'),
        "intent": classification.get('intent'),
        "action": "Please review and approve/edit this response"
    })

    if human_decision.get("approved"):
        return Command(
            update={"draft_response": human_decision.get(
                "edited_response", state.get('draft_response', ''))},
            goto="send_reply"
        )
    else:
        return Command(update={}, goto=END)


def send_reply(state: EmailAgentState) -> dict:
    """Send the email response"""
    # Integrate with email service
    print(f"Sending reply: {state['draft_response'][:100]}...")
    return {}


workflow = StateGraph(EmailAgentState)

workflow.add_node("read_email", read_email)
workflow.add_node("classify_intent", classify_intent)


workflow.add_node(
    "search_documentation",
    search_documentation,
    retry_policy=RetryPolicy(max_attempts=3)
)

workflow.add_node("bug_tracking", bug_tracking)
workflow.add_node("draft_response", draft_response)
workflow.add_node("human_review", human_review)
workflow.add_node("send_reply", send_reply)


workflow.add_edge(START, "read_email")
workflow.add_edge("read_email", "classify_intent")
workflow.add_edge("send_reply", END)


memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


initial_state = {
    "email_content": "I was charged twice for my subscription! This is urgent!",
    "sender_email": "customer@example.com",
    "email_id": "email_123",
    "messages": []
}


config = {"configurable": {"thread_id": "customer_123"}}
result = app.invoke(initial_state, config)
print(f"human review interrupt:{result['__interrupt__']}")

human_response = Command(
    resume={
        "approved": True,
        "edited_response": "We sincerely apologize for the double charge. I've initiated an immediate refund..."
    }
)


final_result = app.invoke(human_response, config)
print(f"Email sent successfully!")
