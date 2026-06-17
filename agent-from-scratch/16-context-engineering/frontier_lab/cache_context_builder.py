"""
TokenPilot 机制手撕 —— cache 友好的双粒度上下文组装器。

落到 16-context-engineering。配套笔记见 paper_notes.md。

设计沿用课程惯例:
- 只依赖 _shared/common.py 的 chat（DeepSeek / openai SDK），不引入框架。
- 核心逻辑留 TODO 给你手撕；骨架、数据结构、可跑的 __main__ 已备好。

两个要复现的机制:
1. 双粒度（dual-granularity）: 每条历史保留 full + digest，近期用 full、旧的用 digest。
2. cache 友好顺序: [稳定前缀] -> [digest 旧] -> [full 近期] -> [本步]，最大化 prefix cache 命中。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

# 复用课程共享 client（与其他模块一致的 import 方式）
sys.path.append(str(Path(__file__).resolve().parents[2] / "_shared"))
from common import chat  # noqa: E402


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class Turn:
    """一条历史消息，同时持有两种粒度。"""
    id: int
    role: str
    full: str                 # 细粒度:原始完整内容
    digest: str | None = None  # 粗粒度:压缩摘要，None 表示尚未生成


@dataclass
class ContextLog:
    """整段历史 + 稳定前缀（system / 工具定义等不变内容）。"""
    stable_prefix: list[dict] = field(default_factory=list)  # [{role, content}]，逐字节稳定
    turns: list[Turn] = field(default_factory=list)
    _next_id: int = 0

    def add(self, role: str, full: str) -> Turn:
        t = Turn(id=self._next_id, role=role, full=full)
        self._next_id += 1
        self.turns.append(t)
        return t


# ---------------------------------------------------------------------------
# 机制 1: 双粒度 —— 生成 digest
# ---------------------------------------------------------------------------
def digest(turn: Turn, max_chars: int = 200, use_llm: bool = False) -> str:
    """
    把一条 full 消息压成粗粒度 digest。

    TODO（你手撕）:
    - 基线版: 规则截断到 max_chars，并在末尾标注可回溯的引用，如
        f"[turn#{turn.id} digest] {turn.full[:max_chars]}…(full 可按 id 取回)"
    - 进阶版（use_llm=True）: 调 chat() 让模型抽取"关键事实/决定/数字"，
      要求输出 <= max_chars。注意 digest 也要带上 turn#id 以便回溯。

    返回: digest 字符串（同时建议写回 turn.digest 缓存，避免重复压缩）
    """
    raise NotImplementedError("TODO: 实现 digest（基线截断 + 选做 LLM 摘要）")


def get_full_by_id(log: ContextLog, turn_id: int) -> str | None:
    """按 id 取回某条历史的 full 内容（双粒度的"可回溯"保证）。"""
    for t in log.turns:
        if t.id == turn_id:
            return t.full
    return None


# ---------------------------------------------------------------------------
# 机制 2: cache 友好的双粒度组装
# ---------------------------------------------------------------------------
def build_context(log: ContextLog, full_window: int = 4) -> list[dict]:
    """
    组装成可直接喂给 chat() 的 messages，顺序为:
        [稳定前缀(原样)] + [旧历史 digest] + [近 full_window 条 full] + (本步在调用处追加)

    关键约束（cache 友好）:
    - stable_prefix 必须逐字节不变且永远放最前。
    - digest 段也应尽量稳定（同一条旧消息每次产生相同 digest）。
    - 每步真正变化的内容放在最后追加，使变化点尽量靠后 -> prefix cache 命中最大化。

    TODO（你手撕）:
    1. 先放 log.stable_prefix（原样 dict）。
    2. 把 turns[:-full_window] 逐条转成 {role, content=digest(t)}。
    3. 把 turns[-full_window:] 逐条转成 {role, content=t.full}。
    4. 返回拼好的 list[dict]。
    """
    raise NotImplementedError("TODO: 实现 cache 友好的双粒度组装")


# ---------------------------------------------------------------------------
# cache 命中近似指标
# ---------------------------------------------------------------------------
def serialize(messages: list[dict]) -> str:
    """把 messages 拍平成一个字符串，用于估算 prefix 重合。"""
    return "\n".join(f"{m.get('role')}:{m.get('content')}" for m in messages)


def cache_prefix_len(prev: list[dict], curr: list[dict]) -> int:
    """
    估算两次组装结果共享的"逐字符前缀"长度 —— prefix cache 命中的近似指标。
    返回相同前缀的字符数（越大越好）。

    TODO（你手撕）:
    - serialize 两次结果，逐字符比较，返回第一个不同处的下标。
    - 这是最朴素的近似（真实命中按 token 边界），用于直观对比"顺序好不好"。
    """
    raise NotImplementedError("TODO: 实现共享前缀长度估算")


# ---------------------------------------------------------------------------
# 反例: 不稳定顺序（每步都让 cache 失效），用于 eval 对照
# ---------------------------------------------------------------------------
def build_context_naive(log: ContextLog, volatile_header: str) -> list[dict]:
    """
    朴素/错误顺序: 把每步都变的内容（volatile_header）放在最前面，
    导致后续所有 token 的 cache 全部 miss。仅作对照基线，不要在生产里这么写。
    """
    msgs: list[dict] = [{"role": "system", "content": volatile_header}]
    msgs += log.stable_prefix
    for t in log.turns:
        msgs.append({"role": t.role, "content": t.full})
    return msgs


# ---------------------------------------------------------------------------
# 冒烟入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log = ContextLog(stable_prefix=[
        {"role": "system", "content": "你是一个严谨的研究助手。"},
    ])
    for i in range(8):
        log.add("user", f"第 {i} 轮问题：请分析数据点 {i}，这里有很长的原始观测……" * 3)
        log.add("assistant", f"第 {i} 轮结论：数据点 {i} 的关键指标为 {i * 7}。")

    print("turns:", len(log.turns))
    print("提示：实现上面的 TODO 后，下面两行才能跑通：")
    print("  ctx = build_context(log, full_window=4)")
    print("  print(cache_prefix_len(build_context(log), build_context(log)))  # 应等于全长")
