from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

model = ChatOllama(model="llama3.2:3b")

# Static chat prompt: no placeholders and no user input
chat_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an experienced research mentor."
    ),
    (
        "human",
        """
Explain the research topic titled:
"Foundation Models in AI Engineering"

Use a beginner-friendly explanation style.
Keep the explanation short.

Provide a single paragraph including:
- the main idea
- important applications
"""
    )
])

prompt = chat_template.invoke({})

result = model.invoke(prompt)

print(result.content)

print(prompt)