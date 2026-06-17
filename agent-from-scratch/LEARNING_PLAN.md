# Agent 技术完整学习方案（2026 校准版）

> 目标：不是“知道 agent 有哪些概念”，而是能从协议、循环、工具、状态、上下文、
> harness、安全、评估到部署，亲手解释和实现一个可靠 agent 系统。

本文件是 `README.md` 的补充总控图。原 `00`-`20` 模块已经覆盖主干，但要达到
“对 agent 细节技术足够把握”，需要把每节都落到可运行代码、可复现实验和可度量验收。

## 最新资料校准结论

截至 2026-06，主流路线已经比较清晰：

- **底层原语仍然是核心**：agent 本质仍是 `model call -> tool execution -> observation`
  的循环，直到模型决定结束。
- **LangChain 1.x 的定位变了**：`create_agent` 是一个可配置 harness，
  LangChain agents 建在 LangGraph 之上；不要再把 LangChain 只当 chain 框架。
- **LangGraph 是低层编排运行时**：重点是 durable execution、persistence、
  streaming、human-in-the-loop、memory、subgraph，而不是“更复杂的 chain”。
- **Deep Agents 值得单独对照**：它把 planning、subagents、虚拟文件系统、
  context compression 做成 batteries-included agent，是你手撕 `16/17/18` 后的最好对照。
- **OpenAI 平台路线分两层**：Responses API 适合托管工具/平台编排；Agents SDK
  适合服务端自己掌控 orchestration、tool execution、state、approval、MCP 和 tracing。
- **MCP 已成为工具/数据接入标准，但不是生产安全答案**：MCP 标准化发现和调用工具，
  但权限、身份传递、预算、错误语义、审计仍要靠 harness 和架构补上。
- **context engineering 是可靠性的主战场**：不只是 memory，而是每一步把什么
  instructions/messages/tools/response format/tool result 放进模型调用。
- **security 必须前置到架构**：prompt 注入、间接注入、工具越权和数据外泄不能只靠
  prompt 防御，要靠最小权限、出站白名单、HITL、隔离和审计。

## 现有方案还需要补强的点

1. **统一验收包**
   每个模块现在有完成标准，但还缺统一格式：代码产物、测试命令、评估样例、
   失败样例、复盘笔记。建议每节完成后都产出：
   - `module_notes.md`：概念、坑、框架对照
   - `*.py`：手撕实现
   - `tests/` 或 `eval_cases.jsonl`：最小验证集
   - `run_report.md`：一次真实运行轨迹、指标、失败和修正

2. **状态模型要更明确**
   `03/16/17/18` 应统一区分：
   - `messages`：短期对话状态
   - `runtime_context`：用户、权限、环境、配置
   - `store`：跨会话长期记忆
   - `scratchpad/filesystem`：长程任务外置状态
   - `trace/replay`：调试与评估状态

3. **工具不是函数列表，而是 ACI**
   `02/15/17` 要补一张工具设计 rubric：
   - 名称是否具体
   - 何时用/何时不用是否清楚
   - 参数是否最小且强类型
   - 错误信息是否可操作
   - 返回是否分页/摘要/引用化
   - 是否有权限、超时、幂等、审计

4. **评估要贯穿，不要等到 `12`**
   从 `01` 开始就建立小 eval 集。每次改 prompt、tool schema、memory、context
   都跑同一批任务，记录成功率、步数、token、延迟、失败类型。

5. **框架对照要补 2026 栈**
   `09` 不只对照 LangGraph/OpenAI Agents SDK/CrewAI/smolagents，还要补：
   - LangChain `create_agent` + middleware
   - LangGraph persistence / interrupts / stores / subgraphs
   - Deep Agents
   - OpenAI Responses API vs Agents SDK
   - MCP tool server 的权限边界

6. **生产化要有 readiness checklist**
   `13/17/19` 需要一份上线清单：超时、重试、预算、速率限制、权限、HITL、
   trace、replay、告警、回滚、secret 管理、PII 脱敏、出站控制。

7. **capstone 要提前定义**
   不要到 `18` 才想做什么。建议从第 1 周就锁定一个真实任务，例如：
   - 代码仓库问答 + 修复建议 agent
   - 文档研究 agent
   - 本地资料整理与问答 agent
   - MCP 工具路由与审计 agent

## 推荐学习顺序

### Phase 0：环境与基线（0.5 天）

对应：`SETUP.md`、`00-foundations`

必须完成：
- 跑通 `_shared/common.py`
- 写 10 行多轮 chatbot，手动维护 `messages`
- 建立 `eval_cases.jsonl`，先放 5 条最简单任务

验收：
- 能解释模型无状态、messages 协议、tool role 回填顺序
- 能说明 DeepSeek OpenAI-compatible Chat Completions 和 OpenAI Responses API 的差异

### Phase 1：Agent 最小内核（2-3 天）

对应：`01-react-agent`、`02-tool-use`

必须完成：
- `react_agent.py`：while loop、tool_calls、tool_call_id 回填、max_steps
- `tool_registry.py`：`@tool`、schema 生成、参数校验、异常回填
- 支持一轮多个 tool_calls
- 10 条 eval：单工具、多工具、多轮、错误参数、工具异常

验收：
- 能画出 agent loop 时序图
- 能解释“单轮 function call 不是 agent”
- 能复现并修复 `tool_call_id` 错配、工具异常、无限循环三类 bug

### Phase 2：状态、记忆与检索（3-5 天）

对应：`03-memory`、`04-rag`、穿插 `16-context-engineering`

必须完成：
- `memory.py`：buffer + summary + token budget
- `scratchpad.py`：结构化笔记（facts / decisions / todos / open_questions）
- `rag.py`：chunk、embed、retrieve、rerank 可选
- `context_builder.py`：按预算组装 system/messages/retrieved/scratchpad/tool results

验收：
- 长对话不会爆上下文
- 能对比“全量塞上下文”与“最小充分上下文”的 token、成功率、失败类型
- 能说明 working / episodic / semantic memory 分别放在哪里

### Phase 3：规划、反思、多 agent 与 workflow 判断力（4-6 天）

对应：`05-planning`、`06-reflection`、`07-multi-agent`、`08-workflows`、穿插 `15`

必须完成：
- `plan_and_execute.py`：planner / executor / replan
- `reflection_loop.py`：generate / critique / refine
- `multi_agent.py`：supervisor + workers，至少一个 handoff
- `workflows/`：Anthropic 5 种 workflow 纯函数版
- `decision_matrix.md`：workflow vs agent、单 agent vs 多 agent、普通模型 vs reasoning 模型

验收：
- 能给一个新任务选型，并明确为什么不用更复杂结构
- 能用 eval 数据证明反思/规划是否真的带来收益
- 能指出多 agent 的上下文隔离收益和协作成本

### Phase 4：框架与协议对照（3-5 天）

对应：`09-frameworks`、`10-mcp`

必须完成：
- 用 LangChain `create_agent` 重写 `01`
- 用 LangGraph `StateGraph` 重写 `05` 或 `08` 中一个 workflow
- 用 Deep Agents 跑一个带文件系统/子 agent/context compression 的任务
- 用 OpenAI Agents SDK 或官方示例理解 handoffs / guardrails / sessions / tracing
- 写一个 MCP server + client，暴露 `02` 的工具

验收：
- 能解释 LangChain agent、LangGraph、Deep Agents、OpenAI Agents SDK 各自边界
- 能说明 MCP 解决的是工具接入标准，不自动解决权限与生产可靠性
- 能把手撕实现和框架抽象逐项对应起来

### Phase 5：Harness、生产化与安全（5-7 天）

对应：`11-advanced-agents`、`12-eval-observability`、`13-production`、
穿插 `17-harness-engineering`、`19-security`

必须完成：
- `harness.py`：预算、超时、防循环、工具运行时、危险动作审批
- `tracer.py`：记录 LLM/tool IO、token、耗时、错误
- `replay.py`：从 trace 复现一次失败
- `service/`：FastAPI + streaming + session
- `security_lab/`：间接注入、越权工具、出站外泄三类攻防实验

验收：
- agent 失控时能被预算/超时/防循环拦住
- 能从 trace 找到一次失败根因
- 能演示 lethal trifecta，并用架构隔离打断攻击链
- 高风险工具必须经过授权或 HITL

### Phase 6：Reasoning 模型与前沿（2-4 天）

对应：`20-reasoning-agents`、`14-frontier`

必须完成：
- 普通模型 + CoT vs reasoning 模型对照实验
- 难度路由：简单任务普通模型，复杂推理任务 reasoning 模型
- thinking/reasoning 内容上下文清理策略
- 选读 1 个 benchmark 或论文方向，写一页笔记

验收：
- 能解释 reasoning 模型没有取代 planning/context/harness，只是改变边界
- 有质量、成本、延迟数据支持“什么时候用 reasoning 模型”

### Phase 7：Capstone（5-10 天）

对应：`18-architecture-design`

必须完成：
- `AGENT_DESIGN_DOC.md`
- 端到端 agent 系统
- eval report
- security review
- 一轮基于数据的架构迭代

验收：
- 能用失败模式推导架构，而不是先选框架
- 能端到端跑通真实任务
- 有数字证明迭代变好
- 能讲清每个核心决策：模型、工具、状态、上下文、拓扑、安全、成本

## 贯穿全课的掌握清单

学完后你应该能不看框架源码，独立回答：

- **协议层**：messages、roles、tool_calls、tool_call_id、structured output、streaming 怎么工作？
- **循环层**：agent loop 的退出条件、预算、防循环、错误恢复怎么设计？
- **工具层**：schema 怎么生成？工具怎么命名、授权、分页、报错、审计？
- **状态层**：messages、checkpoint、store、scratchpad、trace 分别存什么？
- **上下文层**：每一步放什么？怎么 compress/write/select/isolate？
- **规划层**：ReAct、Plan-and-Execute、ReWOO、ToT、workflow 各适合什么？
- **多 agent 层**：何时拆 agent？谁有上下文？谁有权限？如何 handoff？
- **评估层**：如何评最终答案、轨迹、工具调用、成本、延迟、安全？
- **安全层**：如何防直接/间接注入、越权、外泄、恶意 MCP server？
- **生产层**：如何做 streaming、HITL、persistence、replay、监控、限流、降级？
- **框架层**：手撕实现分别对应 LangChain/LangGraph/Deep Agents/OpenAI Agents SDK 的哪些抽象？

## 必读资料（按优先级）

1. Anthropic, *Building Effective Agents*：
   `https://www.anthropic.com/engineering/building-effective-agents`
2. LangChain docs, *LangChain overview* / *Context engineering in agents*：
   `https://docs.langchain.com/oss/python/langchain/overview`
   `https://docs.langchain.com/oss/python/langchain/context-engineering`
3. LangGraph docs, *LangGraph overview*：
   `https://docs.langchain.com/oss/python/langgraph/overview`
4. Anthropic, *Effective context engineering for AI agents*：
   `https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents`
5. Anthropic, *Writing effective tools for AI agents*：
   `https://www.anthropic.com/engineering/writing-tools-for-agents`
6. OpenAI Agents SDK docs：
   `https://openai.github.io/openai-agents-python/`
7. OpenAI platform, *Agents* guide：
   `https://platform.openai.com/docs/guides/agents`
8. Model Context Protocol docs：
   `https://modelcontextprotocol.io/docs/getting-started/intro`
9. OWASP Top 10 for LLM Applications：
   `https://owasp.org/www-project-top-10-for-large-language-model-applications/`
10. HumanLayer, *12-Factor Agents*：
    `https://github.com/humanlayer/12-factor-agents`

## 建议的目录补充

后续每个模块可以逐步补成如下结构：

```text
NN-module/
  README.md
  src/
    *.py
  tests/
    test_*.py
  eval_cases.jsonl
  run_report.md
  module_notes.md
```

不要一开始就补全所有文件。每学到一节，先写最小可跑代码，再补测试和报告。
