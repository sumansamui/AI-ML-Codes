
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

model = ChatOllama(model="llama3.2:3b")


# Static prompt: no placeholders and no user input
prompt = """
You are an experienced research mentor.

Explain the research topic titled:
"Foundation models in AI Engineering"

Use a beginner-friendly explanation style.
Keep the explanation short in length.

Provide a single paragraph including the main idea, and applications.
"""

result = model.invoke(prompt)

print(result.content)