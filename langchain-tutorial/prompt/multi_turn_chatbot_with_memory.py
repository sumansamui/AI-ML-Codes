
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = ChatOllama(
    model="llama3.2:3b"
)


chat_history = [
    SystemMessage(content="You are a helpful AI assistant. You will always answer in a concise and clear manner in short paragraph within 25 words. If you don't know the answer, say 'I don't know'.")
]


while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
         print("AI: Good bye")
         break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)

print(chat_history)