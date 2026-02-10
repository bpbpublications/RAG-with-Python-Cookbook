# document_split_vector_store.py
# This script demonstrates how to split a document into chunks and store them in ChromaDB as a vector store.
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Choose your file (PDF or TXT)
file_path = "RAG.pdf"  # change to your file

# 2. Load the document
# Depending on the file type, we use different loaders.
if file_path.lower().endswith(".pdf"):
    loader = PyPDFLoader(file_path)
elif file_path.lower().endswith(".txt"):
    loader = TextLoader(file_path)
else:
    raise ValueError("Unsupported file type. Use PDF or TXT.")

documents = loader.load()
print(f"Loaded {len(documents)} pages from {file_path}")

# 3. Split the document into chunks
# This will create smaller chunks of text for better processing and storage.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # characters per chunk
    chunk_overlap=50 # overlap between chunks
)
docs = text_splitter.split_documents(documents)
print(f"Split into {len(docs)} chunks.")

# 4. Load SentenceTransformer embeddings
# This model is used to generate embeddings for the text chunks.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 5. Initialize ChromaDB as a vector store
# This will create a local ChromaDB instance to store and retrieve embeddings.
persist_directory = "./chapter5_chroma_db_docs"
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory
)

# 6. Add documents to the vector store
# This will store the embeddings in ChromaDB.
print("Documents stored in ChromaDB.")

# 7. Perform a similarity search
# This will search for documents similar to the query based on their embeddings.
query = "Explain RAG"
results = vectorstore.similarity_search(query, k=1)

# 8. Display the search results
print("\nSearch Results:")
for i, doc in enumerate(results, start=1):
    print(f"{i}. {doc.page_content.strip()[:200]}...")  # Print first 200 characters of each result")
