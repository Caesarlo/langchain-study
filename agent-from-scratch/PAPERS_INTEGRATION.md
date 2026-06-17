# 前沿论文 → 模块整合方案（Frontier Labs）

> 把 2026-06 顶会/arXiv 最新 agent 论文里的**核心机制**，拆成可在本课程技术栈
> （DeepSeek + 纯 `openai` SDK，不依赖框架）手撕复现的「前沿实践包」。
>
> 这份文件是 `LEARNING_PLAN.md` 的延伸：总控图负责"学什么原语"，本文件负责
> "学完原语后，用最新论文把它推到前沿"。

---

## 整合原则（重要）

1. **只抽机制，不抄全文**：每篇论文提炼成一句话机制 → 能用 <150 行 DeepSeek 代码跑通的最小版本。
2. **必须能在本栈复现**：凡是需要 RL / 权重更新的（如 Meta *Early Experience*），
   只取其**数据流思想**做 prompt 层模拟，不做真训练。
3. **eval 优先**：论文里大量是 benchmark，正好喂给课程最该提前建的 `eval_cases.jsonl`。
4. **嵌入而非新增模块**：每个相关模块下加 `frontier_lab/` 子目录，不打乱 `00–20` 主干。

每个 `frontier_lab/` 的标准结构：

```text
NN-module/frontier_lab/
  paper_notes.md      # 1 页：机制 + 原文链接 + 我们取哪部分
  <mechanism>.py      # 最小复现（核心逻辑留 TODO 给你手撕）
  eval_cases.jsonl    # 该机制的验证集
  run_report.md       # 跑通后的指标对比（有/无该机制）
```

---

## 论文 → 模块映射表

| 论文（机制） | 落到模块 | 手撕产物 | 价值 | 可行性 |
|---|---|---|---|---|
| **SkillWeaver** — 分解→检索→组合技能 | `02` / `05-planning` | `skill_router.py`：工具调用拆成可复用 skill，检索后组装 | 高 | ✅ 纯 prompt+检索 |
| **SING / GIST-CMTF** — 意图图、暴露工具前先验证意图 | `02` / `15` | `intent_gate.py`：大工具集下先推断目标→只暴露相关工具 | 高 | ✅ |
| **TokenPilot** — 双粒度上下文 + 保 prompt cache | `16-context-engineering` ⭐ | `cache_context_builder.py`：cache 友好的上下文组装 | **极高** | ✅ 与现有产物直接重叠 |
| **EvoArena / MemTrace** — 记忆演化 + 知识点粒度评测 | `03-memory` | `memory.py` + `mem_eval.jsonl`：测"证据是否被用上" | 高 | ✅ |
| **Closing the Feedback Loop** — 经验提取→insight 治理（verbal RL） | `06-reflection` / `14` | `reflection_loop.py` 升级：反思沉淀成可复用 insight 库 | 高 | ✅ |
| **Cordon** — 语义事务，不可逆动作提交前暂存+验证 | `19-security` / `17-harness` | `tool_transaction.py`：危险工具 staging→confirm→commit | **极高** | ✅ 与 HITL/审批重叠 |
| **ProvenanceGuard** — MCP agent 来源感知事实核查 | `10-mcp` / `19` | `provenance.py`：工具结果带 source，检测跨源混淆 | 高 | ✅ |
| **AgentFairBench** — agent **行为**偏见评测 | `12-eval` / `19` | `fairness_eval.jsonl`：同任务换人口学属性看决策差异 | 中 | ✅ |
| **A Framework for Evaluating Agentic Skills** | `12-eval` | eval harness 的方法论模板 | 高 | ✅ |
| **CoffeeBench / CEO-Bench** — 长程/多角色经济模拟 | `07-multi-agent` / `14` | capstone 备选任务 | 中 | ⚠️ 较重 |
| **SWE-Explore** — coding agent 如何探索仓库 | `11-advanced-agents` | `repo_explore.py`：仓库导航策略 | 中 | ✅ |
| **Agent Learning via Early Experience**（Meta） | `14-frontier`（选读） | 只做笔记 + prompt 层经验回放模拟 | 概念 | ⚠️ 真训练超栈 |

---

## 最高 ROI 三件套（已开工）

这三个与课程**现有手撕产物直接重叠**，投入产出比最高：

1. **`16` + TokenPilot** → `cache_context_builder.py`：cache 友好 + 双粒度组装，**量化 token 节省**。✅ 骨架已建
2. **`19/17` + Cordon** → `tool_transaction.py`：把"危险动作 HITL"做成正经的事务语义。🔲 骨架已建
3. **`12` + 各 benchmark** → 按论文方法论提前建 `eval_cases.jsonl`，补齐"评估要贯穿"的短板。🔲 骨架已建

---

## 前沿实践包追踪

| 模块 | 机制 | 状态 |
|---|---|---|
| 16 · context-engineering | TokenPilot（cache 友好上下文） | 🔄 骨架就绪，待手撕 |
| 19 · security | Cordon（工具语义事务） | 🔲 骨架就绪，待手撕 |
| 12 · eval-observability | Agentic-Skills / Fairness 评测 | 🔲 骨架就绪，待手撕 |
| 02 · tool-use | SkillWeaver / SING | ⬜ 计划中 |
| 03 · memory | EvoArena / MemTrace | ⬜ 计划中 |
| 06 · reflection | Closing the Feedback Loop | ⬜ 计划中 |
| 10 · mcp | ProvenanceGuard | ⬜ 计划中 |

图例：✅ 完成 · 🔄 进行中 · 🔲 骨架就绪 · ⬜ 计划中

---

## 论文来源（2026-06 抓取）

- arXiv recent "LLM agent"：`http://export.arxiv.org/api/query?search_query=abs:%22LLM+agent%22&sortBy=submittedDate`
- Hugging Face Papers：`https://huggingface.co/papers?q=agent`

> 注：本批论文为 2026-06 最新预印本/热榜，多数在投/已录顶会（NeurIPS/ICLR/ACL/EMNLP）。
> 正式录用与获奖信息以 OpenReview / ACL Anthology / 各会官网为准。
