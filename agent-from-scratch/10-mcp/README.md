# 10 · MCP（Model Context Protocol）

> 工具/数据接入的**标准协议**。以前每个框架自定义工具格式，MCP 统一了它——
> 写一次 server，任何支持 MCP 的客户端（Claude Code、IDE 等）都能用。本模块
> 手撕一个 MCP server + client。

## 学习目标

- 理解 MCP 解决什么问题：M×N 集成 → M+N
- 协议三大原语：tools / resources / prompts
- server 与 client 的职责与握手
- 传输方式：stdio / SSE / streamable HTTP
- 把 `02` 的工具用 MCP 暴露出去，被标准客户端调用

## 核心概念

### 1. 为什么需要 MCP
- 没有标准时：N 个工具 × M 个 app = N×M 套适配
- MCP：工具方实现一次 server，所有兼容 client 即插即用
- 类比"USB-C 接口"

### 2. 三大原语
- **Tools**：可被模型调用的函数（同 `02` 的工具，但走标准协议）
- **Resources**：可读取的数据/文件（只读上下文）
- **Prompts**：可复用的提示模板

### 3. 架构
- **Host**（如 Claude Code）→ 内置 **Client** ←→ **Server**（你写的）
- client/server 一对一，host 可连多个 server

### 4. 传输层
- **stdio**：本地子进程，最常用
- **SSE / streamable HTTP**：远程

### 5. 与前面的联系
- MCP server 里的 tool 本质就是 `02` 的工具，只是包了标准协议外壳

## 代表性参考

- **官方文档**：`https://modelcontextprotocol.io/`
- **官方 Python SDK**：`https://github.com/modelcontextprotocol/python-sdk`
- Anthropic MCP 介绍博客（搜 "Introducing the Model Context Protocol"）
- 官方 servers 示例库：`https://github.com/modelcontextprotocol/servers`

## 手撕任务

> 用官方 `mcp` SDK（这层不重复造轮子，重点是理解协议）：

1. [ ] 写一个 MCP **server**：暴露 2-3 个 tools（复用 `02` 的工具逻辑）
2. [ ] 加一个 **resource**（如读取本地某文件）
3. [ ] 写一个 MCP **client**，连上 server、列出并调用工具
4. [ ]（选做）把这个 server 配进 Claude Code，实际用起来
5. [ ] 对比：MCP 工具 vs `02` 的本地工具，多了哪些标准化收益

## 完成标准

- [ ] server 能被 client 成功发现并调用工具
- [ ] 能讲清 tools/resources/prompts 三原语区别
- [ ] 理解 stdio vs HTTP 传输的适用场景
- [ ] 能说出 MCP 相比框架自定义工具的价值（M+N）

## 下一步
→ `11-advanced-agents`：code agent、computer/browser use 等更强的 agent 形态。
