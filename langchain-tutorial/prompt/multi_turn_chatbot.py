# from langchain_core.prompts import ChatPromptTemplate

# chat_template = ChatPromptTemplate([
#     ('system', 'You are a helpful {domain} expert'),
#     ('human', 'Explain in simple terms, what is {topic}')
# ])

# prompt = chat_template.invoke({'domain':'cricket','topic':'Dusra'})

# print(prompt)

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

model = ChatOllama(
    model="llama3.2:3b"
)

system_message = SystemMessage(
    content="You are a helpful AI assistant. You will always answer in a concise and clear manner in short paragraph within 25 words. If you don't know the answer, say 'I don't know'."
)

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("AI: Good bye")
        break

    messages = [
        system_message,
        HumanMessage(content=user_input)
    ]

    result = model.invoke(messages)

    print("AI:", result.content)