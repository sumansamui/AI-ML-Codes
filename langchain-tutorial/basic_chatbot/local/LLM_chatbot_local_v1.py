from langchain_ollama import ChatOllama


# ---------------------------------------------------------
# Step 1: Initialize the locally running Llama model
# ---------------------------------------------------------
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.3,
    # Explicitly connect to the local Ollama server
    #http://localhost:11434
     base_url="http://127.0.0.1:11434",

    # Maximum number of output tokens
    num_predict=200,
)


# ---------------------------------------------------------
# Step 2: Define a function to generate a response
# ---------------------------------------------------------
def chatbot_reply(message: str) -> str:
    """
    Send only the current user message to the model.

    Because previous messages are not supplied,
    this chatbot has no memory or chat history.
    """

    
    # Send the messages to the local Ollama model
    result = llm.invoke(message)

    # Extract and return the generated text
    return result.content


# ---------------------------------------------------------
# Step 3: Start the chatbot
# ---------------------------------------------------------
print("=" * 50)
print("Ollama + LangChain Chatbot")
print("Model: Llama 3.2")
print("Type 'exit' to stop.")
print("=" * 50)


while True:
    # Receive input and remove extra surrounding spaces
    user_message = input("\nYou: ").strip()

    # Ignore an empty input
    if not user_message:
        continue

    # Exit the chatbot
    if user_message.lower() == "exit":
        print("AI: Goodbye!")
        break

    try:
        # Generate a response to the current message
        response = chatbot_reply(user_message)

        # Display the response
        print("\nAI:", response)

    except Exception as error:
        print("\nError:", error)
        print("Make sure Ollama is installed and running.")