from langchain_ollama import OllamaEmbeddings


# Initialize a sentence/document embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434",
)


# Convert text into a numerical vector
vector = embeddings.embed_query(
    "Artificial intelligence is transforming education."
)

print(type(vector))
print("Vector dimension:", len(vector))
print("First five values:", vector[:5])