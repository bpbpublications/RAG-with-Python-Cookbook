# This code generates embeddings for a list of text inputs using the HuggingFaceEmbeddings model from LangChain.
from langchain_huggingface import HuggingFaceEmbeddings

# 1️. Load the embedding model (any from sentence-transformers)
# You can choose a different model if needed, but this one is efficient for many tasks.
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2️. Your text input
# Replace these with your own texts or data if needed.
texts = [
    "Retrieval Augmented Generation (RAG) is an architecture that combines the ability of large language models (LLMs) with a retrieval system.",
    "Traditional generative models rely solely on internal parameters for producing responses.",
    "RAG mitigates this by augmenting the generation process with real-time retrieval from external knowledge sources."
]

# 3️. Generate embeddings
# This will convert the list of texts into their corresponding vector embeddings.
embeddings = embedding_model.embed_documents(texts)

# 4️. Output the embeddings
# Print the number of embeddings and their dimensions for verification.
print(f"Generated {len(embeddings)} embeddings.")
print(f"Each vector has {len(embeddings[0])} dimensions.\n")

# Display the first 5 dimensions of each embedding
for i, vector in enumerate(embeddings):
    print(f"--- Embedding {i+1} (first 5 dims) ---")
    print(vector[:5])  # Show only first 5 dimensions for brevity
    print()
