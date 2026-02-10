# time_aware_retrieval.py
# Example of time-aware document retrieval using embeddings and timestamps
from sentence_transformers import SentenceTransformer, util
import torch
from datetime import datetime

# 1. Sample documents with timestamps (YYYY-MM-DD) for demonstration purposes 
DOCUMENTS = [
    {"text": "RAG reduces hallucinations by grounding answers in retrieved evidence.",
     "timestamp": "2023-05-10"},
    {"text": "Dense retrieval uses vector embeddings instead of keyword matching.",
     "timestamp": "2022-08-15"},
    {"text": "Cooking recipes require precise measurements and timing.",
     "timestamp": "2021-12-01"},
    {"text": "Vector databases store embeddings to enable semantic search.",
     "timestamp": "2022-11-20"},
    {"text": "Traveling to new countries helps you learn about culture and history.",
     "timestamp": "2023-01-05"}
]

# 2. Load pre-trained sentence transformer model
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

# 3. Pre-compute embeddings for documents 
doc_texts = [doc["text"] for doc in DOCUMENTS]
doc_embeddings = model.encode(doc_texts, convert_to_tensor=True, normalize_embeddings=True)

# 4. Time-aware retrieval function 
# Query with optional time range filtering with start_date and end_date,
# query is the input query string and top_k which specifies how many top documents to return    
# Returns list of (document_text, similarity_score) tuples  
# If no documents match the time filter, returns an empty list
def time_aware_retrieve(query, start_date=None, end_date=None, top_k=3):
    """
    Retrieve top-k documents relevant to query and within the time range.
    start_date, end_date: string in 'YYYY-MM-DD' format
    """
    # Filter documents by timestamp
    filtered_docs = []
    filtered_embeddings = []
    for doc, emb in zip(DOCUMENTS, doc_embeddings):
        doc_date = datetime.strptime(doc["timestamp"], "%Y-%m-%d")
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            if doc_date < start_dt:
                continue
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            if doc_date > end_dt:
                continue
        filtered_docs.append(doc["text"])
        filtered_embeddings.append(emb)
    
    if not filtered_docs:
        return []  # No document matches the time filter
    
    embeddings_tensor = torch.stack(filtered_embeddings)

    # Encode query
    query_emb = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)

    # Compute cosine similarity
    cos_scores = util.cos_sim(query_emb, embeddings_tensor)[0]

    # Get top-k results
    top_results = torch.topk(cos_scores, k=min(top_k, len(filtered_docs)))
    return [(filtered_docs[idx], float(score)) for score, idx in zip(top_results.values, top_results.indices)]

if __name__ == "__main__":
    # Example Query
    query = "How does RAG reduce hallucinations?"
    results = time_aware_retrieve(query, start_date="2023-01-01", end_date="2023-12-31", top_k=3)

# 5. Print query results, time range, and top documents with similarity scores
    print(f"Query: {query}")
    print("Time Range: 2023-01-01 to 2023-12-31")
    print("Top Documents:")
    for doc, score in results:
        print(f"  similarity={score:.4f} | {doc}")
