# 01-react-agent 手撕复盘笔记

> 记录我手撕 ReAct agent 循环时**真实遇到的问题、当时的错法、最终解法、背后的原理**。
> 这是整套课程最重要的一节：chatbot → agent 的惊险一跃。代码全部我自己敲，老师只讲解 + 挑刺。

---

## 0. 一句话核心认知

> **Agent = LLM + 循环 + 工具 + 模型自主停止**
>
> 把 `00` 那个"人每轮 input 驱动"的循环，换成"**模型自己决定下一步、自己决定何时停**"。
> 人不再是每轮的发动机，模型才是。

退出信号：**`reply.tool_calls` 为空** = 模型认为不需要再调工具了，`content` 即最终答案。

---

## 1. 工具层（schema + 函数 + 映射）

agent 调工具前，要先把"工具说明书"用固定格式告诉模型。三个部分缺一不可：

```python
# ① 真正的 Python 函数（工具结果最终要塞进 content，必须是【字符串】）
def get_weather(city: str):
    return f"The weather in {city} is 36℃"

# ② schema（OpenAI 兼容格式）—— 告诉模型有什么工具、怎么调
get_weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",                 # 模型用它"点名"，必须和 Python 函数对得上
        "description": "查询指定城市的当前天气",  # 模型靠它判断【何时该用】——这是 prompt 的一部分，不是注释
        "parameters": {                         # JSON Schema 描述参数
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名"}},
            "required": ["city"],
        },
    },
}

# ③ 名字 → 函数 的映射（纽带！模型返回字符串名字，靠它找到真正的函数）
TOOLS_MAP = {"get_weather": get_weather, "calculator": calculator}
TOOLS = [get_weather_tool, calculator_tool]
```

### 关键理解

- **`name` 是唯一纽带**：schema 的 name → 模型返回 `tool_call.function.name` → 我用它查 `TOOLS_MAP` 找函数。
- **`TOOLS_MAP` 存的是函数引用 `get_weather`，不是调用 `get_weather()`**。
  引用 = 把函数本身当值传递，以后再调；加括号 = 立刻执行，存进去的会变成返回值。
- **`description` 是给模型读的 prompt**，写得含糊模型就会该调不调、不该调乱调（`02` 深入）。
- **工具返回值必须是字符串**：`calculator` 用 `ast` 自己写了 `safe_eval`（白名单放行常量/四则/一元运算，
  拒绝其它），比 `eval()`（能跑任意代码）和 `sympify`（解析面太宽）都安全；最后 `str(result)` 转字符串。

---

## 2. agent 循环（run_agent）—— 全课心脏

```python
import json

def run_agent(user_input: str, max_iterations: int = 5):
    messages = [
        {"role": "system", "content": "你是一个助手，可以使用工具。"},
        {"role": "user", "content": user_input},
    ]
    for i in range(max_iterations):                  # ① for+上限，防死循环
        reply = chat(messages, tools=TOOLS)
        if not reply.tool_calls:                     # ② 模型不要工具了 = 任务完成
            return reply.content
        messages.append(reply)                       # ③ 头号坑！整条 assistant 消息原样留下
        for tool_call in reply.tool_calls:           # ④ 一轮可能多个工具（并行）
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)  # ⑤ arguments 是字符串，要 loads
            func = TOOLS_MAP[name]
            try:                                     # ⑥ 工具异常捕获，错误当 observation 回填
                result = func(**args)
            except Exception as e:
                result = f"工具执行出错: {e}"
            messages.append({                        # ⑦ role=tool 回填，tool_call_id 原样带回
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })
    return "达到最大迭代次数，任务未完成"             # ⑧ 兜底
```

---

## 3. 我踩的坑（按犯错次数排序，血泪）

### 🥇 头号坑：第 ③ 步 `messages.append(reply)` —— 我错了三次

| 版本 | 我写的 | 为什么错 |
|---|---|---|
| 错法1 | `append({"role":"assistant","content":None,"tool_calls":[{"id": reply.tool_calls.id}]})` | 手动重建 dict，只塞了 id，丢了 name/arguments；且 `reply.tool_calls` 是列表没有 `.id` |
| 错法2 | `append({"role":"assistant","content":None,"tool_calls": reply})` | `tool_calls` 字段塞了整个 message 对象，结构套娃 |
| ✅正解 | `messages.append(reply)` | `reply` 本身就是合法的完整 assistant 消息，原样留下最稳，别手动重建 |

**为什么必须留 assistant 的 tool_calls 请求**（反证法）：
若只 append tool 结果、不留请求，第二轮 messages 里会出现一条 `role=tool, tool_call_id=call_abc`
的消息，但上文没有任何 assistant 发起过 `call_abc` → API 直接报错
`messages with role 'tool' must be a response to a preceding message with 'tool_calls'`。
**请求(assistant.tool_calls) 和 结果(tool 消息) 是一对必须同时存在的因果。**

> 对比 `00` 的教训：`00` 里我说"append reply 对象不规范、要拆 dict"——那是**纯文本**场景。
> 这一课**有 tool_calls 时反过来**：整条 append 才稳，手动拆最容易漏 tool_calls。**场景不同，做法不同。**

### 🥈 `reply.tool_calls` 是【列表】不是单个对象
写了 `reply.tool_calls.id` / `.function.name`（错，列表没这些属性，AttributeError）。
正解：`for tool_call in reply.tool_calls:` 然后用 **`tool_call.xxx`**（循环变量，取当前这一个）。

### 🥉 `json.load` vs `json.loads`（又是少个字母，同 `00` 的 assitant）
- `json.load(f)` 从**文件对象**读；`json.loads(s)` 从**字符串**读（s = string）。
- `arguments` 是字符串，必须用 `json.loads`。

### 其它
- `arguments` 是 JSON **字符串**，不 `json.loads` 直接 `func(**args)` 会炸（`**` 只能展开 dict）。
- `-> Number` 类型注解没 import 也没定义 → `NameError`（要么 `from numbers import Number`，要么删注解）。
- 工具执行没 `try/except`：`safe_eval` 会主动 raise，不捕获整个 agent 崩；捕获后把错误文字回填，模型能自我纠正。
- `__main__` 里 `run_agent(...)` 没 `print`，跑完屏幕只有分隔线，看不到答案。

---

## 4. tool_call_id 的真正作用：配对/对账，不是"持久化"

一轮**并行**调了 2 个工具时，messages 里有 2 条 tool 结果。模型靠 `tool_call_id`
把"每个结果"和"它对应的那次请求"一一对上号——像外卖取餐号，3 杯奶茶靠号码区分。
它是**关联标识符**，不是存数据库的持久化。

---

## 5. 眼见为实：循环真的转了多轮

加 log 后实测：

```
# 单步："北京天气怎么样"
--- 第 1 轮 --- [模型要调的工具] ['get_weather']
--- 第 2 轮 --- [最终回答] 无工具调用，退出

# 多步："北京和上海的温度差多少"
--- 第 1 轮 --- [模型要调的工具] ['get_weather', 'get_weather']   ← 一轮内【并行】调两个！
--- 第 2 轮 --- [最终回答] 无工具调用，退出
```

看到两种形态：**并行**（一轮多个 tool_calls，互不依赖时）+ **多轮**（串行，下一步依赖上一步结果时）。
没有一行代码写"先查北京再查上海再相减"——是模型自己规划的。这就是 agent 的灵魂。

> 已知小局限：`get_weather` 写死所有城市都 36℃，所以温差恒为 0，多步验证说服力打折。
> 要更有说服力可让不同城市返回不同温度（小字典或 hash 城市名）。核心目标（循环跑通）已达成。

---

## 6. 对照旧代码：为什么它【不是】agent

旧代码 `langchain/core-components/2-models/5-1-tool-execution-loop.py`：

```python
ai_msg = model_with_tools.invoke(messages)        # 第1次问
for tool_call in ai_msg.tool_calls: ...           # 执行这一轮工具
final_response = model_with_tools.invoke(messages) # 第2次问，直接当最终答案 print
```

| | 旧代码 5-1 | 我的 react_agent.py |
|---|---|---|
| 结构 | 固定两步直线 `invoke→tool→invoke` | `while` 循环，轮数不定 |
| 何时停 | **程序员写死**（必停在第二次） | **模型决定**（tool_calls 为空） |
| 多步任务 | ❌ 超过一轮工具就完不成 | ✅ 转 N 轮直到完成 |
| 防失控 | 不涉及 | ✅ max_iterations |

**结论**：旧代码是"带工具的一次性问答"，我的才是"会自己转圈直到任务完成的 agent"。
**循环 + 模型自主停止**，就是这一跃的全部秘密。

---

## 下一步

→ `02-tool-use`：把"手动写 schema"这个体力活，用一个 `@tool` 装饰器自动化
（从函数签名 + docstring 自动生成 schema）。
