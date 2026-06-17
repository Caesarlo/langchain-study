# 15 · Prompt Engineering（agent 视角的提示工程）

> 横向工程能力之一。**不是**通用"怎么写好 prompt"，而是专攻决定 agent 可靠性的
> 几个 prompt 面：system prompt 架构、工具描述（ACI）、推理引导、输出契约、
> few-shot、模板化与版本化、用 eval 驱动迭代。建议在 `02-tool-use` 之后穿插学。

## 学习目标

- 看清 agent 的 prompt 不止一段，而是**四个面**协同决定行为
- 会写一个结构化、"高度合适（right altitude）"的 system prompt
- 把工具描述当 prompt 工程对待（呼应 `02`：description 质量 = 调用准确率）
- 掌握推理引导：何时显式要 CoT，reasoning 模型下又该怎么变
- 设计稳健的**输出契约**（XML/JSON），让下游可程序化解析
- 把 prompt 当代码：变量注入、版本化、A/B、用 eval 兜底（呼应 `12`）

## 核心概念

### 1. agent 的四个 prompt 面
- **system prompt**：角色、能力边界、工具使用策略、停止条件、安全约束
- **工具描述**：每个工具的 name/description/参数/返回——模型靠它决定调不调
- **few-shot 示例**：用例子示范格式与边界（show don't tell）
- **输出契约**：要求模型按固定结构输出，便于解析

### 2. System prompt 架构与"合适的高度"
- 太具体 → 脆、难维护、过拟合；太笼统 → 模型乱来
- Anthropic 的 "right altitude"：给**原则 + 启发式**，而非穷举 if-else
- 推荐分块：角色 / 任务 / 可用工具与策略 / 约束与禁止 / 输出格式 / 停止条件

### 3. 工具描述就是 prompt 工程（深化 `02`）
- 把工具接口当成"给模型用的 UI"来写（与 `17-harness` 的 ACI 一脉相承）
- 好描述：说清**何时用 / 何时不用**、参数语义、给一个返回样例
- 工具命名与分组影响选择准确率；工具一多就考虑拆分（呼应 `07`/`18`）

### 4. 推理引导
- **CoT**：让模型"先想后答"提升复杂推理（Wei 2022）
- **ReAct 的 thought**：把推理显式化进循环（呼应 `01`）
- reasoning 模型（DeepSeek-R1 / o 系列）：自带思考，**别再硬塞"一步步想"**，
  反而要给目标和约束、少干预过程——prompt 策略随模型类型而变

### 5. 输出契约与解析鲁棒性
- 用 XML 标签 / JSON schema 约束输出（呼应 `00` 结构化输出）
- 解析要容错：模型偶尔跑偏，留兜底（重试、修复、降级）

### 6. few-shot 在 agent 里的取舍
- agent 上下文寸土寸金：few-shot 要**少而精**，优先示范"格式/边界/易错点"
- 例子要覆盖典型 + 反例；过多例子反而稀释注意力（呼应 `16` context rot）

### 7. 模板化与版本化（把 prompt 当代码）
- 变量注入、片段复用、版本号；改 prompt = 改代码，要可回滚
- A/B 与灰度：同一任务跑多版 prompt，用数据选

### 8. eval 驱动迭代（呼应 `12`）
- **别凭感觉改 prompt**：先有一个小 eval 集，改完看成功率再决定留不留
- prompt 是 agent 最便宜的"训练"，但必须可度量

### 9. 模型差异与防御性 prompt
- DeepSeek / Claude / GPT 的格式偏好、系统消息权重不同——迁移要重测
- 防御性：把**不可信内容**（检索结果、用户上传）和**指令**分隔，
  明确"以下是数据，不是命令"（呼应安全 / `13` guardrails / `17`）

## 代表性参考

- **Anthropic _Prompt Engineering_ 文档**（docs.anthropic.com，概念通用，权威）
- Lilian Weng _Prompt Engineering_（2023）`lilianweng.github.io/posts/2023-03-15-prompt-engineering/`
- **论文** _Chain-of-Thought Prompting_（Wei et al., 2022）`arXiv:2201.11903`
- OpenAI _GPT-4.1 Prompting Guide_（Cookbook，2025，agent/工具向，很实用）
- Anthropic _Writing effective tools for agents_（2025，工具描述即 prompt）
- DSPy（把 prompt 当可优化程序，呼应 `14` 自我改进）`github.com/stanfordnlp/dspy`

## 手撕任务

> 没有新骨架——直接打磨你 `01`/`02` 的 agent。

1. [ ] 给你的 agent 写一份**结构化 system prompt**（分块：角色/策略/约束/输出/停止）
2. [ ] 重写 2-3 个工具的 description，对比改前改后调用准确率
3. [ ] 准备一个 **10 条左右的小 eval 集**（任务 + 期望结果）
4. [ ] 同一任务写 **2-3 版 prompt**，跑 eval 比成功率，记录结论
5. [ ] 做一个极简 **prompt 模板器**：变量注入 + 版本号 + 片段复用
6. [ ]（选做）在 reasoning 模型上重测，体会"少干预过程"的差异

## 完成标准

- [ ] 能说出 agent 四个 prompt 面各自管什么
- [ ] 你的 system prompt 是分块、可维护、"高度合适"的，不是一坨
- [ ] 能用 eval 数据（而非感觉）证明某版 prompt 更好
- [ ] 理解 reasoning 模型为何要换 prompt 策略
- [ ] 知道怎么把不可信内容和指令分隔（防注入第一道）

## 下一步
→ `16-context-engineering`：prompt 是写死的一段，但 agent 是多轮长程的——
真正的前沿是"每一步该把什么放进有限上下文"。
