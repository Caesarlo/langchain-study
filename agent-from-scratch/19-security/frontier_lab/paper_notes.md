# Cordon — Semantic Transactions for Tool-Using LLM Agents

> 落到 `19-security`（与 `17-harness` 协同）。本课程取其**事务化危险动作**机制：
> 不可逆的工具效果在真正提交前，先 staging、再验证、最后 commit。

## 原文

- 标题：*Cordon: Semantic Transactions for Tool-Using LLM Agents*（2026-06-16, arXiv）
- 作者：Zheng Chen, Hanqing Liu, Duling Xu, … Jidong Zhai 等
- 一句话：一个事务化运行时，**把不可逆的 agent 效果在提交前暂存并验证**。

## 为什么这是安全/harness 的天然升级

课程 Phase 5 已要求"高风险工具必须经过授权或 HITL"。Cordon 把这件事从
"弹个框让人点同意"升级成**数据库式事务语义**：

```
begin → stage(声明将产生的副作用) → validate(规则/HITL/第二模型) → commit | rollback
```

好处：危险动作（删文件、发邮件、转账、写数据库）在 **commit 之前对外不可见**，
验证不过就 rollback，整个动作从未真正发生。这正面打击 lethal trifecta 里的
"对外通信"那一环——把外泄/越权动作卡在 commit 闸口。

## 核心机制（我们取这些）

1. **副作用声明**：危险工具不直接执行，而是返回一个 `StagedEffect`（描述它**将要**做什么）。
2. **闸口验证**：对 staged 效果跑校验——白名单、预算、HITL 审批、或第二模型审。
3. **原子提交/回滚**：验证通过才真正执行（commit）；否则丢弃（rollback），agent 看到的是"被拒绝"。
4. **审计**：每个事务（staged 内容、验证结果、commit/rollback）落 trace。

## 我们不取的部分

- 论文里跨多工具的分布式事务一致性（2PC 之类）较重，先做**单工具事务**；
  多工具原子性留作进阶 TODO。

## 手撕目标（见 `tool_transaction.py`）

1. [ ] `@transactional` 装饰器：把危险工具包成"返回 StagedEffect 而非直接执行"。
2. [ ] `validate()`：白名单 + 预算 + HITL 三选一/组合的闸口。
3. [ ] `commit()` / `rollback()`：原子提交或丢弃。
4. [ ] 审计：每个事务写入 trace（呼应 `12` 的 tracer）。
5. [ ] 攻防 eval：间接注入诱导"删全部文件"/"外发邮件"，验证被闸口拦下。

## 验收

- [ ] 危险动作在 commit 前对外不可见，rollback 后无副作用。
- [ ] 能演示一次被注入诱导的越权动作被事务闸口拦截。
- [ ] 每个事务在 trace 中可复盘。
