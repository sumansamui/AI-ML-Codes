import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


# ---------------------------------------------------------
# Step 1: Load environment variables from the .env file
# ---------------------------------------------------------
load_dotenv()


# # ---------------------------------------------------------
# # Step 2: Check whether the Groq API key is available
# # ---------------------------------------------------------
# if not os.getenv("GROQ_API_KEY"):
#     raise ValueError(
#         "GROQ_API_KEY was not found. "
#         "Please add it to the .env file."
#     )


# ---------------------------------------------------------
# Step 3: Initialize the Groq-hosted LLM
# ---------------------------------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
)


# ---------------------------------------------------------
# Step 4: Define a function for generating the reply
# ---------------------------------------------------------
def chatbot_reply(message: str) -> str:
    """
    Send one user message to the LLM and return its response.

    No previous conversation is sent to the model.
    Therefore, this chatbot has no memory.
    """

    result = llm.invoke(message)

    return result.content


# ---------------------------------------------------------
# Step 5: Start the chatbot
# ---------------------------------------------------------
print("=" * 50)
print("Simple Groq Chatbot")
print("Type 'exit' to stop.")
print("=" * 50)


while True:
    # Take input from the user
    user_message = input("\nYou: ").strip()

    # Ignore empty messages
    if not user_message:
        continue

    # Stop the chatbot
    if user_message.lower() == "exit":
        print("AI: Goodbye!")
        break

    try:
        # Send only the current message to the LLM
        response = chatbot_reply(user_message)

        # Display the LLM response
        print("\nBot:", response)

    except Exception as error:
        print("\nError:", error)