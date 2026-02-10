# faiss_vector_store_from_text.py

# Step 1: import necessary libraries
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Step 2: Prepare your text documents
# These documents will be indexed in the FAISS vector store.
texts = [
    "Retrieval Augmented Generation enhances LLMs output by injecting external knowledge.",
    "LangChain supports multiple vector stores including FAISS and Chroma.",
    "FAISS is a library for efficient similarity search.",
    "Embeddings transform text into numerical vector representations."
]

# Step 3: Convert texts to Document objects
# This is necessary for LangChain to handle the text properly.
documents = [Document(page_content=text) for text in texts]

# Step 4: Load the embedding model
# You can choose a different model if needed, but this one is efficient for many tasks.
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 5: Create the FAISS vector store
# This will index the documents using the specified embedding model.
vectorstore = FAISS.from_documents(documents, embedding_model)

# Step 6: Save the FAISS index to disk
# This allows you to persist the index and load it later without re-indexing.
vectorstore.save_local("chapter5_faiss_index")

# Step 7: Perform a similarity search
# This will find the most relevant documents for a given query.
query = "What is LangChain?"
results = vectorstore.similarity_search(query, k=2)

# Step 8: Output the results
# This will print the content of the retrieved documents.
for i, res in enumerate(results):
    print(f"Result {i+1}: {res.page_content}")
