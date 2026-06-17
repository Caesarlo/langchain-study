"""
Agentic-Skills 评测方法论手撕 —— 按技能粒度 + 配置矩阵聚合，含公平性子集。

落到 12-eval-observability。配套笔记见 paper_notes.md。

惯例同其他模块:核心逻辑留 TODO 给你手撕。这套 harness 设计成可复用到全课程——
每改一次 prompt/tool/context 策略,就跑同一批 case 出对照报告。
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent


@dataclass
class EvalCase:
    id: str
    task: str
    skill: str                       # 技能标签:按它聚合,而非只给总分
    expect_contains: str | None = None
    pair_id: str | None = None       # 公平性:成对反事实 case 共享同一 pair_id
    meta: dict = field(default_factory=dict)


def load_cases(path: Path) -> list[EvalCase]:
    """读取 jsonl -> EvalCase 列表。

    TODO（你手撕）:
    - 逐行 json.loads,映射到 EvalCase（缺失字段给默认值）。
    """
    raise NotImplementedError("TODO: 加载 eval_cases.jsonl")


def run_case(case: EvalCase, agent_fn: Callable[[str], str]) -> bool:
    """
    跑单条 case,判定是否通过。

    agent_fn: 接收 task 文本,返回 agent 的最终输出（你接自己的 agent loop）。

    TODO（你手撕）:
    - 调 agent_fn(case.task) 拿输出。
    - 若 case.expect_contains 非空,判断是否包含即可（基线判定）；
      进阶可换 LLM-as-judge。
    """
    raise NotImplementedError("TODO: 实现单条判定")


def eval_by_skill(cases: list[EvalCase], agent_fn: Callable[[str], str]) -> dict[str, float]:
    """
    按 skill 聚合成功率 —— 论文核心:定位 agent 到底弱在哪类技能。

    TODO（你手撕）:
    - 对每条 case 跑 run_case,按 case.skill 累加 通过/总数。
    - 返回 {skill: success_rate}。
    """
    raise NotImplementedError("TODO: 按 skill 聚合成功率")


def eval_matrix(cases: list[EvalCase],
                configs: dict[str, Callable[[str], str]]) -> dict[str, dict[str, float]]:
    """
    配置矩阵:同一批 case 跑多套配置（不同 prompt/模型/上下文策略）。

    configs: {配置名: 对应的 agent_fn}
    返回: {配置名: {skill: success_rate}}，用于直接对照"哪套配置更好"。

    TODO（你手撕）: 对每个 config 调 eval_by_skill。
    """
    raise NotImplementedError("TODO: 实现配置矩阵对照")


def eval_fairness(cases: list[EvalCase], agent_fn: Callable[[str], str]) -> float:
    """
    公平性子集:成对反事实 case（仅人口学属性不同）比较决策是否一致。

    TODO（你手撕）:
    - 按 pair_id 把 case 配对（每对 2 条,其余条件相同）。
    - 跑两条,比较 agent 的决策是否相同。
    - 返回"决策一致率"（1.0 = 完全无差异 = 无行为偏见）。
    """
    raise NotImplementedError("TODO: 实现反事实公平一致率")


if __name__ == "__main__":
    cases = "（实现 load_cases 后）"
    print("期望用法:")
    print("  cases = load_cases(HERE / 'eval_cases.jsonl')")
    print("  print(eval_by_skill(cases, my_agent))           # 各技能成功率")
    print("  print(eval_matrix(cases, {'v1': a1, 'v2': a2})) # 配置对照")
    print("  print(eval_fairness(cases, my_agent))           # 反事实一致率")
