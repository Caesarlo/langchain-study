# Agentic-Skills 评测方法论 + AgentFairBench

> 落到 `12-eval-observability`。本课程取两篇评测论文的**方法论**，提前把
> `eval_cases.jsonl` 按"技能粒度 + 行为公平性"建起来——直接补 `LEARNING_PLAN.md`
> 点名的短板："评估要贯穿，不要等到 12"。

## 原文

1. *A Framework for Evaluating Agentic Skills at Scale*（2026-06-16, arXiv）
   —— 在 500 个真实 agent skill × 19 种模型配置上做大规模评测。
2. *AgentFairBench: Do LLM Agents Discriminate When They Act?*（2026-06-15, arXiv）
   —— 多领域基准，测 agent 在招聘/借贷/分诊**行动**中的人口学差异。

## 取其方法论（不照搬规模）

### 来自 Agentic-Skills

- **技能粒度评测**：不要只看"任务整体过没过"，而是按 *skill*（一个可命名能力，
  如"会用日期工具""会分页读取大结果"）分别打分——定位 agent 到底弱在哪一类技能。
- **多配置矩阵**：同一批 case × 多个变量（prompt 版本 / 模型 / 温度 / 上下文策略），
  形成对照表，让"改动是否真的有收益"可量化。

### 来自 AgentFairBench

- **反事实公平**：把同一任务里的**人口学属性**（性别/年龄/族裔）替换，其余完全不变，
  看 agent 的**最终决策/动作**是否随之改变。差异 = 行为偏见。
- 关键点：测的是 agent **采取的行动**（录用/放贷/分诊），不是它"说"的话——
  这与传统 LLM 公平性评测的区别。

## 我们怎么落地

- `skill_eval.py`：按 skill 维度聚合成功率，输出"技能 × 配置"对照表。
- `fairness_eval.jsonl`：成对反事实 case（仅人口学属性不同），度量决策一致率。
- 这套 harness 复用到全课程：每改一次 prompt/tool/context，就跑同一批 case 出报告。

## 手撕目标（见 `skill_eval.py`）

1. [ ] 加载 `eval_cases.jsonl`，每条带 `skill` 标签。
2. [ ] 跑 agent，按 `skill` 聚合成功率（而非只给总分）。
3. [ ] 支持"配置矩阵"：同批 case 跑多套配置，输出对照表。
4. [ ] 公平性子集：成对 case 比较决策是否一致，报告 disparity。

## 验收

- [ ] 报告能指出"agent 在哪类 skill 上最弱"。
- [ ] 改一次 prompt 后，能用同批 case 量化收益（成功率 / 步数 / token）。
- [ ] 公平性子集能给出反事实决策一致率。
