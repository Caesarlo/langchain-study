from _shared.common import chat

messages = [{"role": "system", "content": "你是一个科研助手"}]


while True:
    user = input()
    if user == "exit":
        break

    messages.append({"role": "user", "content": user})
    reply = chat(messages=messages, temperature=0, max_tokens=50)
    print(reply.content)

    messages.append({"role": "assistant", "content": reply.content})
