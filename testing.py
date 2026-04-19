from ollama import chat

instruction=input("enter the system instruction:... ")
while True:
    user_input = input("you: ")
    if user_input.lower() in ['exit', 'end', 'bye']:
        break

    stream = chat(
        model='gemma3:1b',
        messages=[
        {'role': 'system', 'content': instruction},
        {'role': 'user', 'content': user_input}
        ],
        stream=True,
    )
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

    print()  