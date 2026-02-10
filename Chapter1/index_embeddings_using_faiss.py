# This code creates a FAISS index from text embeddings using LangChain's HuggingFaceEmbeddings.
# It allows for efficient similarity search on the indexed texts.
# Ensure you have the required libraries installed:
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Sample texts to embed
texts = [
    "Retrieval Augmented Generation (RAG) is an architecture that combines the ability of large language models (LLMs) with a retrieval system to enhance the factual accuracy.",
    "Traditional generative models rely solely on internal parameters for producing responses, which limits their ability to provide up-to-date or domain-specific knowledge.",
    "RAG mitigates this by augmenting the generation process with real-time retrieval from external knowledge sources.",
    "Traditional generative models are now mostly replaced or augmented by deep learning-based transformer models, which offer greater accuracy, coherence, and scalability."   
]

# 2. Load the embedding model
# You can choose a different model if needed, but this one is efficient for many tasks.
# This model is from the sentence-transformers library, which is commonly used for generating text embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Create a FAISS index from the embeddings
# This will convert the list of texts into their corresponding vector embeddings and create an index.
vector_store = FAISS.from_texts(texts, embedding=embedding_model)


# 4. Perform a similarity search
query = "benefit of using RAG?"
results = vector_store.similarity_search(query, k=3)  # top 3 matches

# 5. Output the results
# Print the query and the top results from the FAISS index.
print(f"\n Query: {query}\n")
for i, doc in enumerate(results):
    print(f"Result {i+1}: {doc.page_content}")
