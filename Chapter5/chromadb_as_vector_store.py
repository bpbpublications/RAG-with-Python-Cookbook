# chromadb_as_vector_store.py
# This script demonstrates how to use ChromaDB as a vector store for text embeddings.
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Load a pre-trained SentenceTransformer model
# This model is used to generate embeddings for the text documents.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Initialize ChromaDB as a vector store
# This will create a local ChromaDB instance to store and retrieve embeddings.
persist_directory = "./chroma_db_offline"
vectorstore = Chroma(
    collection_name="offline_collection",
    embedding_function=embeddings,
    persist_directory=persist_directory
)

# 3. Create some example documents
# These documents will be added to the vector store.
docs = [
    Document(page_content="ChromaDB is an open-source vector database for AI applications.", metadata={"source": "wiki"}),
    Document(page_content="LangChain helps developers build context-aware applications with LLMs.", metadata={"source": "wiki"}),
    Document(page_content="Python is a popular programming language for AI and data science.", metadata={"source": "wiki"})
]

# 4. Add documents to the vector store
# This will generate embeddings for the documents and store them in ChromaDB.
vectorstore.add_documents(docs)

print("Documents added and saved (offline).")

# 5. Perform a similarity search
# This will search for documents similar to the query based on their embeddings.
query = "Tell me about ChromaDB"
results = vectorstore.similarity_search(query, k=2)

# 6. Print the search results
# Displaying the results of the similarity search.
print("\nSearch Results:")
for i, doc in enumerate(results, start=1):
    print(f"{i}. {doc.page_content} (source: {doc.metadata['source']})")
