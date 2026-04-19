import ollama
chat_history=[]

while True:
    user_input=input("you: ")
    if user_input.lower in ['exit','end','bye']:
        break
    else:
        chat_history.append({'role':'user','content':user_input})
        response=ollama.chat(
        model='gemma3:1b',messages=chat_history)

        bot_replay=response['message']['content']
        print("bot:",bot_replay)

        chat_history.append({'role':'assistant','content':bot_replay})


