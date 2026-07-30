

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

model = ChatOllama(model="llama3.2:3b")



template = PromptTemplate.from_template(
    """
You are an experienced research mentor
Explain the research topic: {topic_input}

Explanation style: {style_input}
Explanation length: {length_input}

Provide a single paragraph including the main idea, and applications.
"""
)

topic_input = input("Enter research topic: ")
style_input = input("Enter explanation style: ")
length_input = input("Enter explanation length: ")

prompt = template.invoke({
    "topic_input": topic_input,
    "style_input": style_input,
    "length_input": length_input
})

result = model.invoke(prompt)

print("\nSummary:\n")
print(result.content)