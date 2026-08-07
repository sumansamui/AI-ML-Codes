from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


prompt = ChatPromptTemplate.from_template(
    "Explain {topic} using 5 sentences in simple language."
)

model = ChatOllama(
    model="llama3.2:3b",
    temperature=0.2
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke(
    {
        "topic": "machine learning"
    }
)

print("Final output from the chain:")
print(result)

# Step 1: Format the prompt manually
formatted_prompt = prompt.invoke(
    {
        "topic": "machine learning"
    }
)

print("Formatted prompt:")
print(formatted_prompt)


# Step 2: Pass the formatted prompt to the LLM manually
model_output = model.invoke(formatted_prompt)

print("\nRaw model output:")
print(model_output)


# Step 3: Pass the LLM output to the parser manually
final_output = parser.invoke(model_output)

print("\nFinal parsed output:")
print(final_output)