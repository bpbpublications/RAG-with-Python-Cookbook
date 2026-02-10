# cold_hot_storage_separation.py
# This script demonstrates how to separate hot and cold storage for documents
import os
import json
import numpy as np
from datetime import datetime
import dateutil.parser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Set up embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Sample documents
# These documents should have a 'last_accessed' field indicating when they were last used
documents = [
    {"content": "Artificial Intelligence is transforming healthcare.", "last_accessed": "2025-08-01"},
    {"content": "Machine Learning models need lots of labeled data.", "last_accessed": "2024-03-15"},
    {"content": "FAISS enables fast similarity search on large datasets.", "last_accessed": "2025-08-10"},
    {"content": "Old research papers are often useful for background reading.", "last_accessed": "2022-11-20"},
]

# 3. Separate hot and cold documents
# Hot documents are those accessed within the last 6 months
def is_hot(doc):
    """Hot if accessed within last 6 months"""
    now = datetime.now()
    last_accessed = dateutil.parser.parse(doc["last_accessed"])
    delta = (now - last_accessed).days
    return delta <= 180   # within 6 months

hot_docs = [doc for doc in documents if is_hot(doc)]
cold_docs = [doc for doc in documents if not is_hot(doc)]

# 4. Store HOT docs in FAISS
# Convert hot documents to text format for FAISS
hot_texts = [d["content"] for d in hot_docs]

if hot_texts:
    hot_store = FAISS.from_texts(hot_texts, embedding_model)
else:
    hot_store = None

# 5. Store COLD docs in JSON
# Convert cold documents to a simple JSON format
with open("cold_storage.json", "w", encoding="utf-8") as f:
    json.dump(cold_docs, f, indent=2)

print("Hot storage (FAISS):", hot_texts)
print("Cold storage (JSON):", [d["content"] for d in cold_docs])

# 6. Retrieval function
# This function retrieves documents from both hot and cold storage based on a query
def retrieve(query, top_k=2):
    results = {"hot": [], "cold": []}

    # Search in HOT store
    if hot_store:
        hot_results = hot_store.similarity_search(query, k=top_k)
        results["hot"] = [r.page_content for r in hot_results]

    # Load cold store
    with open("cold_storage.json", "r", encoding="utf-8") as f:
        cold_data = json.load(f)

    if cold_data:
        cold_texts = [d["content"] for d in cold_data]
        cold_embeds = embedding_model.embed_documents(cold_texts)
        query_vec = embedding_model.embed_query(query)

        similarities = np.dot(cold_embeds, query_vec) / (
            np.linalg.norm(cold_embeds, axis=1) * np.linalg.norm(query_vec)
        )
        top_cold_idx = np.argsort(similarities)[::-1][:top_k]
        results["cold"] = [cold_texts[i] for i in top_cold_idx]

    return results


# 7. Example query
# This will retrieve relevant documents from both hot and cold storage
results = retrieve("AI in healthcare")

# 8. Display results
print("\nQuery Results:")
print("HOT store:", results["hot"])
print("COLD store:", results["cold"])
