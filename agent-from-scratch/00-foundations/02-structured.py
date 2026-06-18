import json

from _shared.common import chat

messages = [{"role": "system", "content": "你是一个科研助手。输出结果为json。"}]


while True:
    user = input()
    if user == "exit":
        break

    messages.append({"role": "user", "content": user})
    reply = chat(messages=messages, temperature=0,
                 response_format={"type": "json_object"})
    reply_json = json.loads(reply.content)

    print(reply_json)

    messages.append({"role": "assistant", "content": reply.content})
