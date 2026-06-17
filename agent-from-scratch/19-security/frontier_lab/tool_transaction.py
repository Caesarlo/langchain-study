"""
Cordon 机制手撕 —— 工具语义事务（staging -> validate -> commit/rollback）。

落到 19-security（与 17-harness 协同）。配套笔记见 paper_notes.md。

惯例同其他模块:只依赖标准库 + _shared/common.py；核心逻辑留 TODO 给你手撕。

目标:让不可逆的危险工具（删文件、发邮件、转账…）在真正提交前
先被暂存和验证,验证不过则 rollback,对外从未发生。
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class TxState(str, Enum):
    STAGED = "staged"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


@dataclass
class StagedEffect:
    """一个危险动作的"声明":它将要做什么,但还没做。"""
    tool_name: str
    args: dict
    description: str                 # 人类可读:这个动作将造成什么副作用
    state: TxState = TxState.STAGED
    audit: list[str] = field(default_factory=list)  # 事务生命周期的审计轨迹


# 危险工具注册表:name -> 真正执行的函数（只有 commit 时才会被调用）
_EXECUTORS: dict[str, Callable[..., Any]] = {}


def transactional(description: str):
    """
    装饰器:把一个危险工具变成"返回 StagedEffect 而非直接执行"。

    用法:
        @transactional("将删除指定文件,不可逆")
        def delete_file(path: str): ...

        eff = delete_file(path="/tmp/x")   # 此时并未真正删除,只拿到 StagedEffect
        commit(eff) / rollback(eff)        # 由闸口决定

    TODO（你手撕）:
    - 把被装饰函数登记到 _EXECUTORS[func.__name__]。
    - wrapper 不执行原函数,而是构造并返回 StagedEffect(tool_name, args, description)。
    - 记得把调用参数收集进 args（注意 kwargs）。
    """
    def decorator(func: Callable[..., Any]):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            raise NotImplementedError("TODO: 返回 StagedEffect 而非直接执行")
        return wrapper
    return decorator


def validate(eff: StagedEffect, *, allowlist: set[str] | None = None,
             budget_ok: bool = True, require_hitl: bool = False) -> bool:
    """
    闸口:对 staged 效果做校验。任一不过即拒绝。

    TODO（你手撕，组合下列任意闸口）:
    - 白名单: eff.tool_name 必须在 allowlist 内（None 表示不限）。
    - 预算: budget_ok 为 False 直接拒。
    - HITL: require_hitl 时,打印 eff.description 并请求人工确认
            （课程里可用 input()；生产里走审批队列）。
    - 把每步判定 append 到 eff.audit,便于复盘。
    返回: True=放行, False=拒绝
    """
    raise NotImplementedError("TODO: 实现白名单/预算/HITL 闸口")


def commit(eff: StagedEffect) -> Any:
    """
    原子提交:验证通过后才真正执行底层动作。

    TODO（你手撕）:
    - 若 eff.state != STAGED 抛错（不能重复提交）。
    - 从 _EXECUTORS 取出真正的执行函数,用 eff.args 调用。
    - 成功后置 eff.state = COMMITTED,写审计,返回执行结果。
    """
    raise NotImplementedError("TODO: 实现 commit")


def rollback(eff: StagedEffect) -> None:
    """
    回滚:丢弃 staged 效果,底层动作从未发生。

    TODO（你手撕）:
    - 置 eff.state = ROLLED_BACK,写审计。
    - 因为 staged 阶段从未真正执行,所以无需"撤销"——这正是事务化的价值。
    """
    raise NotImplementedError("TODO: 实现 rollback")


# ---------------------------------------------------------------------------
# 示例危险工具 + 冒烟入口
# ---------------------------------------------------------------------------
@transactional("将删除指定文件,不可逆")
def delete_file(path: str) -> str:
    # 真正的删除逻辑（仅 commit 时执行）。课程里先模拟,别真删。
    return f"[已删除] {path}"


@transactional("将向外部地址发送邮件,属对外通信")
def send_email(to: str, body: str) -> str:
    return f"[已发送] to={to} bytes={len(body)}"


if __name__ == "__main__":
    print("实现 TODO 后,期望流程:")
    print("  eff = delete_file(path='/tmp/important')   # -> StagedEffect, 未真正删除")
    print("  if validate(eff, allowlist={'send_email'}): commit(eff)")
    print("  else: rollback(eff)                         # 删文件不在白名单 -> 被拦下")
