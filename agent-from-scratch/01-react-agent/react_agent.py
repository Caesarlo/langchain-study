import operator
import ast
import json

from _shared.common import chat


def get_weather(city: str):
    return f"The weather in {city} is 36℃"


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval(expr: str):
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise ValueError("表达式语法错误")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return node.value
            raise ValueError("只支持数字常量")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in OPS:
                raise ValueError("只支持加减乘除运算")

            left = visit(node.left)
            right = visit(node.right)

            if op_type is ast.Div and right == 0:
                raise ValueError("除数不能为 0")

            return OPS[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in UNARY_OPS:
                raise ValueError("不支持该一元运算")

            return UNARY_OPS[op_type](visit(node.operand))

        raise ValueError("不支持的表达式")

    return visit(node)


def calculator(expression: str):
    result = safe_eval(expression)
    return str(result)


get_weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "模拟查询指定城市的天气，用于工具调用测试",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名,如 北京、上海"
                }
            },
            "required": ["city"]
        }
    }
}

calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "计算加减乘除表达式",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "加减乘除表达式"
                }
            },
            "required": ["expression"],
            "additionalProperties": False
        }
    }
}


TOOLS_MAP = {
    "get_weather": get_weather,
    "calculator": calculator
}

TOOLS = [get_weather_tool, calculator_tool]


def run_agent(user_input: str, max_iterations: int = 5):
    messages = [
        {"role": "system", "content": "你是一个助手，可以使用工具。"},
        {"role": "user", "content": user_input}
    ]

    for i in range(max_iterations):
        reply = chat(messages, tools=TOOLS)

        print(f"--- 第 {i+1} 轮 ---")

        if not reply.tool_calls:
            print(f"[最终回答] 无工具调用,退出")
            return reply.content

        print(f"[模型要调的工具] {[tc.function.name for tc in reply.tool_calls]}")

        # messages.append(
        #     {"role": "assistant", "content": None, "tool_calls": reply.tool_calls})
        messages.append(reply)

        for tool_call in reply.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            func = TOOLS_MAP[name]

            try:
                result = func(**args)
            except Exception as e:
                result = f"工具执行出错: {e}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    return "达到最大迭代次数,任务未完成"


if __name__ == "__main__":
    print(run_agent("北京天气怎么样", max_iterations=5))
    print("="*100)
    print(run_agent(user_input="北京和上海的温度差多少", max_iterations=5))
