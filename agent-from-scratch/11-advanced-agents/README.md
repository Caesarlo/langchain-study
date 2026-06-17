# 11 · Advanced Agents（高级 agent 形态）

> code agent（写代码并执行）、computer/browser use（操作 GUI/浏览器）。这些是
> 当前最强也最危险的 agent 形态，**沙箱安全是重中之重**。

## 学习目标

- Code Agent：让 LLM 写代码并执行（比 JSON 工具调用更强的表达力）
- 代码执行沙箱：隔离、超时、资源限制、危险操作拦截
- Computer Use：截图 + 坐标点击，操作真实 GUI
- Browser Agent：用 Playwright 驱动浏览器完成网页任务
- 这些形态的安全边界与失控风险

## 核心概念

### 1. Code Agent（为什么"写代码"更强）
- JSON 工具调用：每个能力要预先定义成工具
- code agent：模型直接写 Python，组合循环/条件/库 → 表达力强得多
- 代表：smolagents 的 `CodeAgent`
- 代价：必须沙箱，否则等于给模型你机器的 shell

### 2. 沙箱（本模块安全核心）
- 隔离手段：子进程 + 受限环境 / Docker / 远程沙箱（E2B 等）
- 超时、内存/CPU 限制、禁网、文件系统隔离
- 危险调用拦截（`os.system`、文件删除等）
- **永远假设模型会写出危险代码**

### 3. Computer Use
- 循环：截图 → 模型给出动作（点哪、打什么字）→ 执行 → 再截图
- Anthropic computer use、OpenAI computer-using-agent
- 慢、易错、高风险，适合有人盯着的场景

### 4. Browser Agent
- 用 Playwright/Puppeteer 操作 DOM 或视觉
- 任务：填表、抓数据、自动化流程
- 反爬、动态页面、登录态等现实难题

## 代表性参考

- smolagents `CodeAgent`：`https://github.com/huggingface/smolagents`
- Anthropic _Computer Use_ 文档与参考实现
- OpenAI _Computer-Using Agent_ / Operator
- **Browser-use** 项目：`https://github.com/browser-use/browser-use`
- E2B 沙箱：`https://e2b.dev/`（安全代码执行）

## 手撕任务

> 骨架 `code_agent.py`，安全第一：

1. [ ] 写一个 code agent：模型输出代码 → 解析 → 在**沙箱**执行 → 回填结果
2. [ ] 实现最小沙箱：子进程 + 超时 + 受限 builtins（先本地后可上 Docker）
3. [ ] 加危险操作检测/拦截，写测试验证拦得住
4. [ ]（选做）用 Playwright 写一个浏览器 agent，完成一个简单网页任务
5. [ ]（选做）对比 code agent vs `02` 的 JSON 工具 agent 在某任务上的表现

## ⚠️ 安全须知

- 沙箱没做好之前，**不要**让 agent 执行模型生成的代码
- 测试时禁网、用临时目录、限制权限
- computer/browser use 务必在可控环境、有人监督下跑

## 完成标准

- [ ] code agent 能写代码并安全执行、拿回结果
- [ ] 沙箱能拦住危险操作（有测试证明）
- [ ] 能讲清 code agent 为何比 JSON 工具表达力强、代价是什么
- [ ] 理解 computer/browser use 的工作循环与风险

## 下一步
→ `12-eval-observability`：agent 强了也要能评估和观测，否则无法迭代。
