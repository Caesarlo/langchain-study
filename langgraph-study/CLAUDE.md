# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A learning/study collection of standalone LangGraph scripts. Each file is a self-contained demo of one agent or workflow pattern (mostly from Anthropic's "Building Effective Agents"). Files are meant to be **run individually**, not imported — there is no package, library, build step, or test suite here.

## Repository layout (important)

`langgraph-study/` (where this file lives) is a **subproject**. The real project root is the **parent directory** (`../`, the `langchain-study` repo), which owns all tooling and config:

- `../pyproject.toml`, `../uv.lock` — dependencies, managed by **uv** (Python `>=3.13`)
- `../.venv/` — the virtual environment
- `../.env` — model credentials (required by every script; see below)
- `../.git/`, `../img/` — git root and the output dir for generated graph PNGs
- `../langchain/core-components/` — a sibling collection of LangChain v1 agent/model examples (separate from these LangGraph scripts)

## Running scripts

Run from the **repo root** (`../`), not from `langgraph-study/`, because:
- scripts write graph diagrams to the relative path `./img/...` (which exists only at the repo root — `open("./img/...png")` fails if that dir isn't present)
- `load_dotenv()` resolves `../.env` either way, but running from root keeps everything consistent

```bash
# from the repo root (parent of this dir)
uv run python langgraph-study/workflow-agent/parallelization.py
uv run python langgraph-study/ThinkInLangGraph/main.py
```

There is no lint/test command configured; `ruff` is available (see `../.ruff_cache`) if you want to lint manually.

## Required environment (.env)

Every script reads these from `../.env` and raises `ValueError` if `MODEL_API_KEY` or `MODEL_BASE_URL` is missing:

- `MODEL_API_KEY` — API key
- `MODEL_BASE_URL` — **OpenAI-compatible** endpoint (these scripts target a custom `base_url`, not api.openai.com)
- `MODEL_NAME` — model id (default `gpt-5.4-mini`)
- `MODEL_TEMPERATURE` — default `0.2`

## Shared code shape

Every script repeats the same boilerplate (copy it when adding a new one): `load_dotenv()`, read the four env vars, guard for missing key/url, then construct a single `ChatOpenAI(model, api_key=SecretStr(...), base_url, temperature, timeout=120)`. There is no shared/common module — the init block is duplicated per file by design.

The graph pattern is consistent across files: a `TypedDict` state, node functions `def node(state) -> dict` returning **partial** state updates, then `StateGraph(State)` + `add_node` / `add_edge` / `add_conditional_edges` → `.compile()` → `.invoke(initial_state)`.

## Pattern catalogue (the architecture)

These files exist to demonstrate distinct LangGraph mechanisms — know which file shows which:

- `workflow-agent/prompt_chain.py` — sequential chain with a gating `add_conditional_edges` check
- `workflow-agent/routing.py` — LLM router via `with_structured_output(Route)` → conditional edge to one of N nodes
- `workflow-agent/parallelization.py` — fan-out from `START` to multiple nodes, joined by an `aggregator` node
- `workflow-agent/Orchestrator-worker.py` & `Creating-workers-in-LangGraph.py` — dynamic fan-out with the `Send()` API; workers write concurrently into `Annotated[list, operator.add]` (the reducer that makes parallel writes safe)
- `workflow-agent/Evaluator-optimizer.py` — generator/evaluator feedback loop (conditional edge routes back until "funny")
- `workflow-agent/llm_and_augmentation.py` — LLM augmentation basics: `with_structured_output` and `bind_tools`
- `workflow-agent/Agents.py` — `@tool` definitions + `bind_tools` (tool-calling building blocks)
- `ThinkInLangGraph/main.py` — the most complete example: an email-triage agent showing **routing via `Command(goto=...)` returned from nodes** (an alternative to conditional edges), **human-in-the-loop** with `interrupt()` + `Command(resume=...)`, a `MemorySaver` checkpointer, `thread_id` config, and per-node `RetryPolicy`
- `demo.py`, `hello_world.ipynb` — minimal "first graph" examples

### Two routing styles to keep straight

1. **Conditional edges** — a router function returns a label; `add_conditional_edges(node, fn, {label: target})` maps it (used in `routing.py`, `prompt_chain.py`, `Evaluator-optimizer.py`).
2. **`Command`** — a node returns `Command(update={...}, goto="next_node")`, combining state update and routing in one (used throughout `ThinkInLangGraph/main.py`).

### Graph visualization

Several scripts call `display(Image(graph.get_graph().draw_mermaid_png()))` and/or write PNGs to `./img/`. `display`/`Image` come from IPython and render cleanly in Jupyter; as a plain script the PNG file write still works (when run from the repo root) but `display` has no effect.
