from ollama import chat
import tiktoken

model='gemma3:1b'
enc = tiktoken.get_encoding("cl100k_base")
instruction = "you are a friendly ai assistant"
# token counter
def count_tokens(messages):
    total = 0
    for m in messages:
        total += len(enc.encode(m["content"]))
    return total

# store conversation history
messages = [
    {"role": "system", "content": instruction}
]
while True:
    user_input = input("you: ")
    if user_input.lower() in ['exit', 'end', 'bye']:
        break
    # add user message
    messages.append({"role": "user", "content": user_input})
    stream = chat(
        model=model,
        messages=messages,
        stream=True,
    )
    full_reply = ""

    for chunk in stream:
        content = chunk['message']['content']
        full_reply += content
        print(content, end='', flush=True)

    print()

    # add assistant reply to memory
    messages.append({"role": "assistant", "content": full_reply})

    # print token usage
    print("Tokens:", count_tokens(messages))