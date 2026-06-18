from _shared.common import chat_stream

messages = [{"role": "system", "content": "你是一个科研助手。"}]


while True:
    user = input()
    if user == "exit":
        break

    messages.append({"role": "user", "content": user})

    full = ""
    for delta in chat_stream(messages=messages, temperature=0):
        print(delta, end="", flush=True)
        full += delta
    print()

    messages.append({"role": "assistant", "content": full})
