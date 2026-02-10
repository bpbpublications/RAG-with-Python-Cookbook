# This code creates a Chroma vector store from text embeddings using LangChain's HuggingFaceEmbeddings.
# It allows for efficient similarity search on the indexed texts.
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Sample texts to embed
# Replace these with your own texts if needed.
texts = [
    "Chroma is a popular vector database used to store embeddings (vectors).",
    "We are using sentence transformers for generating embeddings.",
    "RAG is a polpular framework to make Agentic AI applications.",
    "LangChain is a framework for builing applications using LLM.",
]

# 2. Load the embedding model
# You can choose a different model if needed, but this one is efficient for many tasks.
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Create a Chroma vector store from the embeddings
# This will convert the list of texts into their corresponding vector embeddings and create an index.
vector_store = Chroma.from_texts(
    texts,
    embedding=embedding_model,
    persist_directory="chroma_vector_store"
)    

# 4. Persist the vector store
vector_store.add_texts(texts)

# 5. Perform a similarity search
# This will find the top k most similar texts to the query based on their embeddings.
query = "Which database is used to store embeddings?"
results = vector_store.similarity_search(query, k=1)

# 6. Output the results
# Print the query and the top results from the Chroma vector store.
print(f"\n Query: {query}\n")
for i, doc in enumerate(results):
    print(f"Result {i+1}: {doc.page_content}")
