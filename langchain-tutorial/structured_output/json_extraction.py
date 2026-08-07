from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# Define the local Ollama model
model = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# Define the JSON parser
parser = JsonOutputParser()

# Create the prompt template
template = PromptTemplate(
    template=(
        "Make a one-day itinerary for {city}.\n {format_instruction}"
    ),
    input_variables=["city"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

# Create the formatted prompt
prompt = template.invoke({
    "city": "Kolkata"
})

# Send the prompt to Ollama
response = model.invoke(prompt)

# Parse the model output into Python data
result = parser.invoke(response)

print(result)