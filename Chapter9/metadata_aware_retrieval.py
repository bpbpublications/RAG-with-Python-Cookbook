# metadata_aware_retrieval.py
# Example of metadata-aware dense retrieval using Sentence Transformers
from sentence_transformers import SentenceTransformer, util
import torch

# 1. Sample documents with metadata (tags) for demonstration purposes 
DOCUMENTS = [
    {"text": "RAG reduces hallucinations by grounding answers in retrieved evidence.",
     "category": "AI", "author": "Alice", "year": 2023},
    {"text": "Dense retrieval uses vector embeddings instead of keyword matching.",
     "category": "AI", "author": "Bob", "year": 2022},
    {"text": "Cooking recipes require precise measurements and timing.",
     "category": "Cooking", "author": "Carol", "year": 2021},
    {"text": "Vector databases store embeddings to enable semantic search.",
     "category": "AI", "author": "Alice", "year": 2022},
    {"text": "Traveling to new countries helps you learn about culture and history.",
     "category": "Travel", "author": "Dave", "year": 2023}
]

# 2. Load pre-trained Sentence Transformer model
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

# 3. Pre-compute document embeddings and store with metadata for retrieval 
doc_texts = [doc["text"] for doc in DOCUMENTS]
doc_embeddings = model.encode(doc_texts, convert_to_tensor=True, normalize_embeddings=True)

# 4. Metadata-aware retrieval to filter by metadata and retrieve top-k relevant documents
# Example metadata filters used: {"category": "AI", "author": "Alice"}
# It returns documents matching metadata and most similar to query based on cosine similarity 
# If no documents match metadata, returns empty list
# If metadata_filters is None, retrieves from all documents
def metadata_aware_retrieve(query, metadata_filters=None, top_k=3):
    """
    Retrieve top-k documents relevant to query and metadata filters.
    metadata_filters: dict, e.g., {"category": "AI", "author": "Alice"}
    """
    # Filter documents by metadata
    if metadata_filters:
        filtered_docs = []
        filtered_embeddings = []
        for doc, emb in zip(DOCUMENTS, doc_embeddings):
            match = all(doc.get(key) == value for key, value in metadata_filters.items())
            if match:
                filtered_docs.append(doc["text"])
                filtered_embeddings.append(emb)
        if not filtered_docs:
            return []  # No document matches metadata
        embeddings_tensor = torch.stack(filtered_embeddings)
    else:
        filtered_docs = doc_texts
        embeddings_tensor = doc_embeddings

    # Encode query
    query_emb = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)

    # Compute cosine similarity
    cos_scores = util.cos_sim(query_emb, embeddings_tensor)[0]

    # Get top-k results
    top_results = torch.topk(cos_scores, k=min(top_k, len(filtered_docs)))
    return [(filtered_docs[idx], float(score)) for score, idx in zip(top_results.values, top_results.indices)]

if __name__ == "__main__":
    query = "How does RAG reduce hallucinations?"
    filters = {"category": "AI", "author": "Alice"}

    results = metadata_aware_retrieve(query, metadata_filters=filters, top_k=3)

# 5. Print results with similarity scores formatted to 4 decimal places
    print(f"Query: {query}")
    print(f"Metadata Filters: {filters}")
    print("Top Documents:")
    for doc, score in results:
        print(f"  similarity={score:.4f} | {doc}")
