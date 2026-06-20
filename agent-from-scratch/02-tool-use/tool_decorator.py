import inspect

TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean"}


def tool(func):
    name = func.__name__
    description = inspect.getdoc(func)
    if description is None:
        raise ValueError("doc 不能为空")

    sig = inspect.signature(func)

    properties = {}
    required = []

    for pname, param in sig.parameters.items():
        annotation = param.annotation

        if annotation is inspect.Parameter.empty:
            raise TypeError(f"{name} 函数参数 {pname} 必须有类型注解")
        else:
            if annotation not in TYPE_MAP:
                raise ValueError(f"{name} 的参数 {pname} 类型 {annotation} 不支持")
            p_type = TYPE_MAP[annotation]

        properties[pname] = {
            "type": p_type,
            # "description": pname
        }

        if param.default is inspect.Parameter.empty:
            required.append(pname)

    func.schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }

    return func


@tool
def get_weather(city: str):
    """根据城市名，查询天气"""
    return f"The weather in {city} is 36℃"


print(get_weather.schema)
print(get_weather("北京"))
