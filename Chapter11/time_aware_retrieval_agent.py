# time_aware_retrieval_agent.py
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Setup local embeddings using HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Example documents with timestamps
docs = [
    {"page_content": "RAG improves document-grounded responses.", "metadata": {"date": "2024-09-01"}},
    {"page_content": "FAISS is a library for efficient similarity search.", "metadata": {"date": "2025-01-10"}},
]

# 3. Build FAISS vector store
vectorstore = FAISS.from_texts(
    texts=[doc["page_content"] for doc in docs],
    embedding=embeddings,
    metadatas=[doc["metadata"] for doc in docs]
)

# Time-aware retriever (prioritize recent docs)
def time_aware_retrieve(query, top_k=1):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    results = retriever.invoke(query)

    # sort by date (most recent first)
    results.sort(key=lambda x: x.metadata.get("date", "1900-01-01"), reverse=True)
    return results

# 4. Simulated queries example
queries = [
    "How does RAG work?",
    "Tell me about FAISS."
]

# 5. For the example queries, retrieve based on time interval and print the responses
for q in queries:
    results = time_aware_retrieve(q, top_k=1)
    print("\n--- User ---")
    print(q)
    print("\n--- Assistant ---")
    if results:
        for r in results:
            print(f"{r.page_content} (Date: {r.metadata['date']})")
    else:
        print("No relevant information found.")
