# 13 · Production（生产化）

> 把玩具变产品。streaming、human-in-the-loop、guardrails、成本/延迟优化、
> 持久化、并发、部署。本模块把一个 agent 做成 production-ready 的 API 服务。

## 学习目标

- streaming：把中间过程实时推给前端
- human-in-the-loop（HITL）：关键步骤人工审批/介入
- guardrails：输入输出护栏、越权/有害拦截
- 成本与延迟优化：prompt caching、模型分级、并发、超时、限流
- 持久化与恢复：会话状态存储、断点续跑
- 部署：FastAPI 封装、错误处理、监控

## 核心概念

### 1. Streaming
- 流式输出 token 和工具调用进度，改善 UX
- SSE / WebSocket，与 agent loop 的整合

### 2. Human-in-the-Loop
- 高风险动作（付款、删数据、发消息）前**暂停等审批**
- 复用 `12` 的状态持久化：暂停 → 人审 → 恢复
- LangGraph 的 interrupt 机制是工业参考

### 3. Guardrails（护栏）
- 输入：注入检测、越权请求过滤
- 输出：有害内容、敏感信息、格式校验
- 工具层：危险操作白名单/审批（呼应 `11` 沙箱）

### 4. 成本 / 延迟优化
- **prompt caching**：缓存稳定前缀，省钱省时（Claude/DeepSeek 都支持）
- 模型分级：简单步用小模型，难步用大模型
- 并发、超时、重试、限流、降级

### 5. 持久化与会话
- 会话状态入库（Redis/DB），支持多轮、多用户、断点续跑

### 6. 部署
- FastAPI 暴露接口（你 pyproject 里已有 fastapi/uvicorn）
- 健康检查、日志、监控、限流

## 上线 readiness checklist（部署前逐条过）

> 把"能跑"和"敢上线"分开——下面每条没答案就别上生产（呼应 `17`/`19`）。

- [ ] **超时 / 重试**：每次 LLM、每个工具都有超时；可重试的标幂等
- [ ] **预算上限**：单请求 token / 时间 / $ 上限 + 防循环（呼应 `17`）
- [ ] **速率限制 / 降级**：限流，过载有降级路径
- [ ] **最小权限**：工具按调用者权限校验，别信模型自觉（呼应 `19`）
- [ ] **HITL**：高风险动作（付款 / 删除 / 外发）前人工审批
- [ ] **出站控制**：外发域名 / 动作白名单，防数据外泄（呼应 `19` 致命三要素）
- [ ] **trace / replay**：每步可追踪、可离线复现（呼应 `12`/`17`）
- [ ] **告警 / 监控**：错误率、延迟、成本、失败类型有面板与告警
- [ ] **回滚**：prompt / 模型 / 工具版本可回滚
- [ ] **secret 管理**：key 不进代码 / 日志 / 上下文
- [ ] **PII 脱敏**：敏感信息进出有脱敏
- [ ] **guardrails**：输入输出护栏各至少一条，且有测试

## 代表性参考

- Anthropic _Building Effective Agents_ 的工程建议 + prompt caching 文档
- LangGraph：persistence / HITL / streaming 官方文档
- OpenAI Agents SDK 的 guardrails / sessions
- FastAPI 官方文档（你已在用）
- NeMo Guardrails（护栏框架，参考思路）

## 手撕任务

> 骨架 `service/`（FastAPI 应用），核心留 TODO：

1. [ ] 把 `01`/`05` 的 agent 包成 FastAPI 接口
2. [ ] 加 streaming 响应（SSE）
3. [ ] 加一个 HITL：某高风险工具调用前暂停、等确认再继续
4. [ ] 加输入/输出 guardrails（至少各一条规则）
5. [ ] 接 prompt caching / 超时 / 重试，记录成本与延迟
6. [ ] 会话持久化（先内存/文件，再可选 Redis）
7. [ ]（选做）压测并发，观察延迟与失败率

## 完成标准

- [ ] agent 以 API 形式可调、支持流式
- [ ] HITL 能在关键步暂停并恢复
- [ ] guardrails 能拦住至少一类不良输入/输出
- [ ] 能展示 prompt caching 带来的成本/延迟改善
- [ ] 服务有基本的错误处理、超时、日志

## 下一步
→ `14-frontier`：了解前沿方向，决定深入哪个。
