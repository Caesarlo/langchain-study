# 02 · Tool Use（工具使用）

> `01` 里你手写了工具的 JSON schema —— 很啰嗦。本模块把它**自动化**，并深入
> function calling 的工程细节：并行调用、错误处理、schema 设计。

## 学习目标

- 理解 function calling 的完整协议（schema → tool_calls → 回填）
- 从 Python 函数签名**自动生成** JSON schema
- 做一个 `@tool` 装饰器 + 工具注册表（registry）
- 处理并行工具调用（一轮里多个 tool_calls）
- 工具错误处理、超时、重试
- 了解 Toolformer 的"模型自学用工具"思想

## 核心概念

### 1. JSON Schema 设计
- `name` / `description` / `parameters`（含类型、必填、枚举）
- **描述质量 = 调用准确率**：description 写不好，模型就乱调
- 类型提示（type hints）+ docstring → schema 的映射

### 2. `@tool` 装饰器
- 用 `inspect` 读函数签名、参数注解、默认值
- 用 docstring 当 description
- 自动产出 OpenAI tools 格式的 dict
- 注册表：`name -> callable` 的查找表，agent 执行时按名分发

### 3. 并行工具调用
- 一轮 response 可能含**多个** tool_calls
- 顺序执行 vs 并发执行（`asyncio` / 线程池）
- 全部回填后再进入下一轮

### 4. 健壮性
- 参数校验（模型可能传错类型）
- 异常 → 友好 observation（而不是崩溃）
- 超时与重试

### 5. 工具设计 rubric（ACI 自检表）⭐
工具不是"函数列表"，是 **Agent–Computer Interface**——像给人设计 UI 一样为模型设计。
设计 / review 每个工具，逐条自检（呼应 `17` ACI、`15` prompt、`16` 上下文）：

- [ ] **命名具体**：`search_orders` 而非 `query`；模型靠名字猜用途
- [ ] **何时用 / 何时不用**写进 description：减少误调
- [ ] **参数最小且强类型**：必填明确、用 enum 限定取值、少留自由文本兜底
- [ ] **错误信息可操作**：返回"哪错了、可以怎么改"，而非裸 stacktrace
- [ ] **返回分页 / 摘要 / 引用化**：大结果别原样回填（呼应 `16`/`17`）
- [ ] **权限 · 超时 · 幂等 · 审计**：危险动作要授权、副作用工具标幂等并留日志

## 代表性参考

- OpenAI _Function calling_ 官方文档（协议权威来源）
- **论文**：_Toolformer: Language Models Can Teach Themselves to Use Tools_
  （Schick et al., 2023）`arXiv:2302.04761`
- LangChain `@tool` 装饰器、Pydantic schema 生成（看它怎么做，你来手撕简化版）
- `smolagents` 的 `Tool` 抽象

## 手撕任务

> 骨架 `tool_decorator.py`，核心留 TODO：

1. [ ] 写 `@tool` 装饰器：`inspect` 签名 → 自动生成 schema
2. [ ] 写 `ToolRegistry`：注册、按名查找、统一 `execute(name, args)`
3. [ ] 改造 `01` 的 agent，用注册表替代手写 schema 和 if-else 分发
4. [ ] 支持一轮多个 tool_calls（先顺序，再选做并发）
5. [ ] 加参数校验 + 异常回填

## 框架对照

对比 LangChain 的 `@tool` 和 Pydantic `args_schema` —— 你会发现它做的就是你
手撕的这套，只是用 Pydantic 做了更强的校验。

## 完成标准

- [ ] `@tool` 能自动生成正确 schema，无需手写
- [ ] agent 通过注册表分发，新增工具只需加一个被装饰的函数
- [ ] 能处理一轮多个工具调用
- [ ] 工具报错时 agent 不崩、能拿到错误信息继续

## 下一步
→ `03-memory`：agent 现在每次都从零开始，给它装上记忆。
