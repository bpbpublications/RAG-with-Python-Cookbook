# embeddings_with_metadata_filtering.py
# This script demonstrates how to create embeddings with metadata filtering using FAISS.
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Create documents with metadata
# These documents will be used to demonstrate metadata filtering in the FAISS vector store.
docs = [
    Document(page_content="LangChain enables LLM applications.", metadata={"source": "docs/langchain.md", "topic": "LangChain"}),
    Document(page_content="Hugging Face provides transformer models.", metadata={"source": "docs/huggingface.md", "topic": "Transformers"}),
    Document(page_content="OpenAI offers GPT models for developers.", metadata={"source": "docs/openai.md", "topic": "OpenAI"}),
]

# 2. Initialize the embedding model
# This uses the HuggingFaceEmbeddings class to create embeddings for the text chunks.
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Create a FAISS vector store from the documents
# The FAISS vector store will allow for efficient similarity search on the embedded documents.
vectorstore = FAISS.from_documents(docs, embedding=embedding_model)

# 4. Run a similarity search to retrieve the top-k most similar chunks for the query.  
query = "How to use transformer models?"
results = vectorstore.similarity_search(query, k=2)

# 5. Print the results with metadata
# This will display the content of the documents along with their metadata.
print("\n=== Semantic Search with Metadata ===")
for res in results:
    print(f"- Content: {res.page_content}")
    print(f"  Metadata: {res.metadata}")
