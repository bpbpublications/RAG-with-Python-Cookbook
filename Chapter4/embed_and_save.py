# embed_and_save.py
# This script demonstrates how to create embeddings for a large document,
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Create a large document and split it into chunks
# This simulates a large document by repeating a phrase multiple times.
large_text = "Generative models laid the foundation for today’s LLMs.\n" * 20

# 2. Split the large text into smaller chunks for embedding
# This is necessary because large documents can be too big for embedding models.
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_text(large_text)
documents = [Document(page_content=chunk) for chunk in chunks]

# 3. Initialize the embedding model
# This uses the HuggingFaceEmbeddings class to create embeddings for the text chunks.
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Create a FAISS vector store from the documents
# The FAISS vector store will allow for efficient similarity search on the embedded documents.
# Save the FAISS index to disk
# so it can be reused later without needing to recompute the embeddings.

# 4. Create a FAISS vector store from the embedded documents
# The FAISS vector store enables fast similarity search across document embeddings.
vectorstore = FAISS.from_documents(documents, embedding=embedding_model)

# 5. Save the FAISS index to disk
# This allows reusing the stored embeddings later without recomputation.
faiss_index_path = "embed_faiss_index"
vectorstore.save_local(faiss_index_path)

print("Embeddings created and saved to disk.")
