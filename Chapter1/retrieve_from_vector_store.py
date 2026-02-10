# Retrieve from Chroma Vector Store
# This code retrieves text chunks from a Chroma vector store using LangChain's HuggingFace
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load the embedding model
# You can choose a different model if needed, but this one is efficient for many tasks.
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Initialize the Chroma vector store
# Ensure the persist_directory matches where you saved your vector store.
vector_store = Chroma(
    persist_directory="chroma_vector_store",
    embedding_function=embedding_model
)

# 3. Define your query
# This is the question you want to answer using the retrieved context.
query = "What is RAG?"

# 4. Perform a similarity search
# This will find the most relevant text chunks for the given query.
results = vector_store.similarity_search(query, k=1)

# 5. Output the results
# This will print the content of the retrieved documents.
print(f"\nQuery: {query}\n")
for i, doc in enumerate(results):
    print(f"Result {i+1}: {doc.page_content}\n")
