import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


# ---------------------------------------------------------
# Step 1: Load variables from the .env file
# ---------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------
# Step 2: Initialize the Groq-hosted LLM
# ---------------------------------------------------------
#Llama 3.1 8B Instant
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
)


result = llm.invoke("Who won FiFA World Cup in 2026?")

print(result.content)
