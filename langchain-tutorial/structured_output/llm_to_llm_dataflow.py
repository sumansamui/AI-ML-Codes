from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate


# Load the local Ollama model
model = ChatOllama(
    model="llama3.2:3b",
    temperature=0.2
)


# First prompt: generate a detailed report
template1 = PromptTemplate(
    template="Write a 10-line report on {topic}.",
    input_variables=["topic"]
)


# Second prompt: summarize the generated report
template2 = PromptTemplate(
    template="""
Write a 2-line summary of the following text:

{text}
""",
    input_variables=["text"]
)


# Generate the first prompt
prompt1 = template1.invoke(
    {
        "topic": "black hole"
    }
)


# Generate the detailed report
result = model.invoke(prompt1)

print(result)

# Insert the report into the second prompt
prompt2 = template2.invoke(
    {
        "text": result.content
    }
)


# Generate the summary
result2 = model.invoke(prompt2)


print(result2.content)

with open("output.txt", "w", encoding="utf-8") as file:
    file.write("DETAILED REPORT\n")
    file.write("=" * 50 + "\n")
    file.write(result.content)

    file.write("\n\nSUMMARY\n")
    file.write("=" * 50 + "\n")
    file.write(result2.content)