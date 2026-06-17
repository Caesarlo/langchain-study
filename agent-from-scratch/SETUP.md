# 环境准备（开始 `00` 之前看这一篇）

> 把课程从"能读"变成"能跑"。`_shared/common.py` 是所有模块的地基，先让它通。

## 1. 前置

- **Python 3.13**（仓库已用 `uv` 管理：根目录有 `pyproject.toml` / `uv.lock` / `.python-version`）
- 一个 **OpenAI 兼容**的模型 API key（推荐 DeepSeek 官方）

## 2. 安装依赖

手撕阶段（`00`–`03`、`05`–`08`）只需两个包：**`openai` + `python-dotenv`**。

- 根 `pyproject.toml` 已含 `python-dotenv`。
- **`openai` 目前是经 `langchain-openai` 间接装入**（实测 `openai 2.30.0` 可用），
  但课程明确"只依赖官方 `openai` SDK"——**建议显式声明**，避免日后版本漂移：

```bash
uv add openai
```

后续模块的额外依赖见 [§5 分模块依赖与坑](#5-分模块依赖与坑用到再看)。

## 3. 配 `.env`

```bash
cp agent-from-scratch/.env.example .env      # .env 放仓库根目录
# 然后编辑 .env，填入 MODEL_API_KEY 等
```

`common.py` 的 `load_dotenv()` 会从运行目录向上查找 `.env`。课程约定**从仓库根目录运行**。

## 4. 冒烟测试

```bash
python agent-from-scratch/_shared/common.py
```

打印出模型回答即环境通；报 `请在 .env 中设置...` 说明 `.env` 没配好。

## 5. 分模块依赖与坑（用到再看）

| 模块 | 额外依赖 | 安装 | 关键坑 |
|---|---|---|---|
| `00`–`03` | 无（openai + dotenv） | — | 从仓库根目录运行，import 路径才对 |
| `04` rag | 向量库 + embedding 来源 | `uv add faiss-cpu`（或 `chromadb`）；本地嵌入 `uv add sentence-transformers` | ⚠️ **DeepSeek 官方不提供 embedding 接口**，必须换源：硅基流动/通义的 embedding API，或本地 bge |
| `09` frameworks | langchain / langgraph | 根 pyproject 已含 ✓ | 对照用，非手撕必需 |
| `10` mcp | mcp SDK | `uv add mcp` | stdio 与 SSE 两种传输；注意子进程生命周期与超时 |
| `11` advanced | 浏览器 / 沙箱 | `uv add playwright` + `playwright install`；沙箱可选 docker / e2b | ⚠️ **执行模型生成的代码前必须沙箱**（见 `11`/`19`） |
| `12` eval | 追踪（可选） | langsmith（根已含 ✓）或 `uv add langfuse` | LLM-as-judge 有位置/冗长/自我偏好，见模块 |
| `13` production | fastapi / uvicorn | 根 pyproject 已含 ✓ | — |

## 6. 切换模型供应商

`common.py` 是 OpenAI 兼容封装，**改 `.env` 三个值即可切换**，代码一行不用动：

| 供应商 | `MODEL_BASE_URL` | `MODEL_NAME`（示例） |
|---|---|---|
| DeepSeek 官方 | `https://api.deepseek.com` | `deepseek-chat` / `deepseek-reasoner` |
| 硅基流动 | `https://api.siliconflow.cn/v1` | `deepseek-ai/DeepSeek-V3` 等 |
| 通义/百炼 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` 等 |

## 7. reasoning 模型注意事项（`20-reasoning-agents`）

- 用 `deepseek-reasoner`（R1）：改 `.env` 的 `MODEL_NAME`，或在 `chat(..., model="deepseek-reasoner")` 临时传
- 响应里会多一个 **`reasoning_content`**（思考链）字段，和正文 `content` 分开
- reasoning 模型**忽略 `temperature` / `top_p`** 等采样参数——别指望靠它们调输出
- 思考链很长、又贵又慢：上下文里通常**丢弃历史 reasoning_content**（见 `16` 上下文治理）
