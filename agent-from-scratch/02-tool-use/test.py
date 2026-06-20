import inspect


def get_weather(city: str):
    """查询指定城市的当前天气"""
    return f"..."


print(get_weather.__name__)
print(get_weather.__doc__)
sig = inspect.signature(get_weather)
print(sig.parameters)
for pname, param in sig.parameters.items():
    print(pname, param.annotation, param.default)
