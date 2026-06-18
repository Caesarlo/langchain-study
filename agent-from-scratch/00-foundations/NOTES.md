# 00-foundations 手撕复盘笔记

> 这份笔记记录我（学习者）手撕 `00-foundations` 时**真实遇到的问题、当时的错法、最终的解法、背后的原理**。
> 按时间顺序 + 知识点组织，方便日后回看。代码全部我自己敲，老师只讲解 + 挑刺。

---

## 0. 这一课的核心认知（一句话地基）

> **模型是一个纯函数**：`输出 = f(这次发给它的全部 messages)`。
> 没有隐藏状态、没有 session、没有数据库。
> "它记得我" 是个**幻觉**——这个幻觉是我（代码）用"每轮把全部历史重新发过去"制造出来的。

直接影响：多轮对话循环里必须有一个 `messages` 列表，**只增不减**，每轮 append 新内容再整包发出去。

---

## 1. 多轮对话循环（run.py）

目标：命令行连续问答，手动维护 `messages` 历史。

### 我踩的坑（逐个）

| # | 我当时的错法 | 问题 | 正确解法 | 原理 |
|---|---|---|---|---|
| 1 | `user_input = [{"role":"user","content":"写死的问题"}]` | 输入写死 + 类型是 list，循环永远问同一句、退不出 | 用 `input()` 读**字符串** | `input()` 阻塞等键盘输入，回车后返回 str |
| 2 | `if user_input == "exit"` 但 `user_input` 是 list | list 永远 != str，退出失效 | 拿**字符串** `input_text` 去比 `"exit"` | "原始输入" 和 "消息字典" 是两个概念，要分开 |
| 3 | `messages.append(user_input)`（user_input 是 list） | 列表里套列表，发给 API 报错 | `messages.append({"role":"user","content":input_text})` | messages 每个元素必须是 dict |
| 4 | `messages.append(reply)`（reply 是对象） | append 的是 `ChatCompletionMessage` 对象/裸字符串，不规范、跨 provider 易碎 | `messages.append({"role":"assistant","content":reply.content})` | 存回历史要重新组装成标准 dict |
| 5 | `{"role":"assitant",...}` 拼写错 | role 少个 `s`，**第二轮**才报 `Invalid value for 'role'` | 改成 `assistant` | role 是 API 区分"谁在说话"的关键字段，必须精确匹配 |

### 概念纠错：role 不能混

一开始我以为 "content 里放用户提示词 + 一些系统提示词"。**错。**
system / user / assistant 是**三条独立的消息**，各有各的 role，不能揉进一个 content：

```python
{"role": "system",    "content": "你是一个..."}   # 行为设定，单独一条，放最前
{"role": "user",      "content": "我叫小明"}        # 用户说的
{"role": "assistant", "content": "你好小明"}        # 模型回的
```

为什么不能混：模型靠 role 区分"谁在说话"，system 指令的信任/遵守权重最高。把系统指令
伪装成 user 消息，模型对待方式不同，而且是 `19-security` 里 prompt injection 的温床。

### 最终骨架（记死它，后面所有 agent 都是它的变体）

```python
messages = [system]
while True:
    messages.append(user)        # 1. 拿到新输入
    reply = chat(messages)       # 2. 整包历史 → 模型
    messages.append(assistant)   # 3. 把模型输出存回历史
```

### 验证记忆生效的方法

第二句故意用**指代词**："铁电材料是什么" → "那它和压电材料的区别？"。
实测模型主动说"结合你刚才了解的铁电材料"——证明"重发历史"真的把上下文续上了。
（记忆不是模型的，是我代码 append 出来的。）

---

## 2. 采样参数（01-sampling.py）

目标：理解 temperature / max_tokens / top_p / stop 对 agent 的影响。

### 怎么传参

`common.py` 的 `chat(messages, tools=None, **kwargs)` 内部有 `kwargs.pop("temperature", _TEMPERATURE)`，
所以直接覆盖即可：

```python
chat(messages=messages, temperature=1.3, max_tokens=200)
```

### 我做的对照实验（output/01.md）

- `temperature=0` 跑两遍同一开放题 → 开头措辞高度趋同 → **确定、可复现**（agent 该用）
- `temperature=1.3` → 风格/结构明显发散 → **创意场景用**

### 严谨性要点（老师提醒）

要严格对比"同一问题两次"，得**每次重启脚本、只问一句**，否则多轮历史会污染对照。
我的做法（每次重跑脚本）恰好避开了这个坑。

### 参数速查

| 参数 | 作用 | agent 怎么设 |
|---|---|---|
| temperature | 分布陡峭/平坦，控随机性 | 0~0.3 求稳 |
| top_p | 核采样，只在累积概率前 p 的词里挑 | 和 temperature 二选一调 |
| max_tokens | 回复最多生成多少 token | 防失控/控成本 |
| stop | 遇到指定字符串立刻停 | 切断模型自问自答 |

---

## 3. 结构化输出 / JSON mode（02-structured.py）

目标：让模型吐**可程序化解析**的 JSON，而不是自然语言。

### 关键链路

```python
reply = chat(messages, temperature=0, response_format={"type": "json_object"})
data = json.loads(reply.content)   # reply.content 是【字符串】，必须 loads 成 dict
print(data["city"])                # 这才证明它能当字典索引
```

### 我亲手踩的坑（output/02.md）

故意先不在 prompt 里写 "json"，直接报错：

```
openai.BadRequestError: 400 -
"Prompt must contain the word 'json' in some form to use 'response_format' of type 'json_object'."
```

解法：system 里加"输出结果为 json"。
原理：开 JSON mode 时 prompt 必须含 "json" 字样（provider 的硬约束），否则 400。

**为什么 agent 需要它**：把"自然语言"变成"可解析字段"，是 agent 把模型输出当数据用的基础。

---

## 4. 流式 / streaming（03-streaming.py + common.py::chat_stream）

目标：边生成边显示（打字机效果），而不是等全部生成完一次性返回。

### 我踩的坑（演进过程）

**第一版（错）**：
```python
from _shared.common import chat_stream   # 当时这函数根本不存在 → ImportError
reply = chat_stream(messages, stream=True)
messages.append({"content": reply})       # reply 是迭代器，不是字符串
```
三个错：① 函数不存在；② 没写"逐块取 delta 并打印"的循环；③ append 了迭代器对象。

**第二版**：在 `common.py` 里实现了 `chat_stream`，但：
- 🐞 **致命**：顶部混进一行 `from turtle import mode`（IDE 误补全！turtle 是画图教学库，
  跟项目无关，会拖垮**所有**import 它的课程）→ 看到就删。
- ⚠️ 把 `print` 焊死在函数里 → 公共地基不该假设调用方一定想 print。

**最终版（generator，推荐）**：
```python
# common.py —— 只负责"生产"，yield 增量，不 print
def chat_stream(messages, **kwargs):
    resp = client.chat.completions.create(..., stream=True)
    for chunk in resp:
        delta = chunk.choices[0].delta.content
        if delta:            # 首/尾块 content 为 None，判空跳过
            yield delta

# 03-streaming.py —— 调用方决定怎么"消费"
full = ""
for delta in chat_stream(messages, temperature=0):
    print(delta, end="", flush=True)   # 边收边打，不换行，立即刷新
    full += delta
print()
messages.append({"role": "assistant", "content": full})   # 存完整字符串
```

### 关键原理

- 流式取的是 `chunk.choices[0].delta.content`，**不是** `.message.content`。
  `delta` = **增量**：每块是"新增的一小段文字"，整段回答 = 所有 delta 拼接。
- 非流式 `chat()` 写死了 `return resp.choices[0].message`（等全部完成），所以**流式不能用 chat()**。
- **生产/消费分离**：`chat_stream` 只 `yield`，谁打印/写文件/转发前端由调用方决定。
  这是 agent 工程的重要分界（`13-production` 流式 UX 会用到）。
- 最终文本和非流式**一模一样**，区别只在**过程**（用户体验）。

---

## 5. 通用经验（这一课学到的"元技能"）

1. **延迟一轮才炸的 bug 最折磨人**：`assitant` 拼写错、role 错，第一轮看着好好的，第二轮才报错。
2. **读报错是手撕的一部分**：400 报错信息直接告诉我"prompt 必须含 json"，比猜快得多。
3. **IDE 误补全的 import**（`from turtle import ...`）混进公共文件危害极大，看到就删。
4. **公共地基函数要"薄"**：只做数据生产，把消费/打印的决策权交给调用方。
5. **概念要分清**：原始输入(str) vs 消息字典(dict)、完整 message vs 增量 delta、
   字符串 vs 解析后的 dict——很多 bug 本质是把两个概念搅在一起。

---

## 下一步

→ `01-react-agent`：在三步循环的第 2、3 步之间塞一个"模型自己决定要不要调工具"的判断，
并引入第四种 role：`tool`。ReAct agent ≠ 新东西，就是「本课循环 + 自主决策 + tool role」。
