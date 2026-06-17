# langchain-study

LangChain / LangGraph 学习仓库 —— 从手撕 agent 原语，到框架实践与核心组件。

> 技术栈：Python 3.13（`uv` 管理）· DeepSeek 官方接口（OpenAI 兼容）· LangChain 1.x / LangGraph 1.x

---

## 目录导览

| 目录 | 内容 |
|---|---|
| [`agent-from-scratch/`](agent-from-scratch/README.md) | **手撕版 agent 系统化课程**：编号化、自包含的学习路线，从零实现每个核心概念（ReAct / 工具 / 记忆 / RAG / 规划 / 多智能体…），再用主流框架对照。详见其 [`README.md`](agent-from-scratch/README.md) 与 [`LEARNING_PLAN.md`](agent-from-scratch/LEARNING_PLAN.md)。 |
| [`langgraph-study/`](langgraph-study/) | **LangGraph 实践**：workflow / agent 模式的可运行示例（编排-worker、评估-优化、并行、路由、prompt 链等），见 `workflow-agent/`。 |
| [`langchain/`](langchain/) | LangChain **核心组件**练习（`core-components/`）。 |
| `img/` | 各示例配套的流程图 / 截图。 |

## 环境准备

仓库根用 `uv` 统一管理依赖与虚拟环境：

```bash
uv sync                 # 安装依赖、创建 .venv
uv run python <脚本>     # 在项目环境中运行
```

需要的环境变量（写入根目录 `.env`）：

```
MODEL_API_KEY=...
MODEL_BASE_URL=...
MODEL_NAME=...
MODEL_TEMPERATURE=...
```

> `agent-from-scratch` 各模块复用 `agent-from-scratch/_shared/common.py`（最小 LLM client）。
> 环境与冒烟测试细节见 [`agent-from-scratch/SETUP.md`](agent-from-scratch/SETUP.md)。

## 快速上手

- 想系统学 agent 原理 → 进 [`agent-from-scratch/`](agent-from-scratch/README.md)，按 `00 → 14` 主线推进。
- 想看 LangGraph 怎么搭 workflow / 多智能体 → 进 [`langgraph-study/workflow-agent/`](langgraph-study/workflow-agent/)。
- 想练 LangChain 基础组件 → 进 [`langchain/core-components/`](langchain/core-components/)。
