from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# First prompt: generate a detailed report
prompt1 = PromptTemplate.from_template(
    "Generate a detailed report on {topic}"
)

# Second prompt: summarize the generated report
prompt2 = PromptTemplate.from_template(
    """
    Generate a five-point summary from the following text:

    {text}
    """
)

# Local Ollama model
model = ChatOllama(
    model="llama3.2:3b",
    temperature=0.3
)


parser = StrOutputParser()

# Sequential chain
chain = (prompt1| model| parser| prompt2| model| parser
)

result = chain.invoke(
    {
        "topic": "MOSFET Amplifier"
    }
)

print(result)


# Display the chain structure
#chain.get_graph().print_ascii()
