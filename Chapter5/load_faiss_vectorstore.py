# load_faiss_vectorstore.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Load the embedding model
# You can choose a different model if needed, but this one is efficient for many tasks.
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Prepare your text documents
# These documents will be indexed in the FAISS vector store.
documents = [
    Document(page_content="LangChain is a framework for building LLM-powered apps."),
    Document(page_content="FAISS is a vector database for fast similarity search."),
    Document(page_content="Transformers from Hugging Face are widely used in NLP."),
]

# 3. Create the FAISS vector store
# This will index the documents using the specified embedding model.
vectorstore = FAISS.from_documents(documents, embedding=embedding_model)

# 4. Save the FAISS index to disk
# This allows you to persist the index and load it later without re-indexing.
save_dir = "chapter5_faiss_store"
vectorstore.save_local(folder_path=save_dir)

print("Vector Store saved to:", save_dir)
