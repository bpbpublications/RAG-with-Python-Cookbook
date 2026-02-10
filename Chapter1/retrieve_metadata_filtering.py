# This code demonstrates how to perform a similarity search with metadata filtering using LangChain and FAISS.
# It includes steps to load documents, split them while preserving metadata, create embeddings, and perform a filtered similarity search.
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 1. Raw data for processing of the program
# This is a list of dictionaries simulating the content to be processed.    
# Each dictionary contains a 'text' key for the content and a 'metadata' key for additional information.
# This simulates a scenario where you have multiple pieces of content with associated metadata.        
raw_docs = [
    {
        "text": "In RAG LangChain supports various vector stores including FAISS and Chroma.",
        "metadata": {"source": "content1", "category": "LangChain"}
    },
    {
        "text": "In RAG LangChain is used to build RAG applications.",
        "metadata": {"source": "content2", "category": "LangChain"}
    },
    {
        "text": "In RAG FAISS is a library for efficient similarity search.",
        "metadata": {"source": "content3", "category": "FAISS"}
    },
]

documents = [Document(page_content=doc["text"], metadata=doc["metadata"]) for doc in raw_docs]

# 2. Split with metadata preserved
# This step splits the documents into smaller chunks while preserving their metadata.
# This is important for efficient processing and retrieval.
splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
split_docs = splitter.split_documents(documents)

# 3. Create embeddings using Hugging Face
# This step converts the text chunks into numerical vector representations using a pre-trained model.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Store in FAISS with metadata
# This step creates a FAISS vector store from the document chunks and their embeddings.
faiss_index = FAISS.from_documents(split_docs, embeddings)

# 5. Filtered similarity search (filter by metadata)
# This step performs a similarity search on the FAISS index, filtering results based on metadata.
query = "What is RAG?"
results = faiss_index.similarity_search(
    query=query,
    k=5,
    filter={"category": "LangChain"}  # Filtering condition
)

# 6. Print the results
# This will print the content of the retrieved documents along with their metadata.
# This is useful for understanding which documents were retrieved and their associated metadata.
print("\nFiltered Similarity Search Results:")
for i, doc in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"Source: {doc.metadata.get('source')}")
    print(f"Category: {doc.metadata.get('category')}")
    print(f"Content: {doc.page_content}")
