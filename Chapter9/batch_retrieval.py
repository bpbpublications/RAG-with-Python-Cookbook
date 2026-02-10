# batch_retrieval.py
# Example of batch retrieval using SentenceTransformer embeddings and cosine similarity
from sentence_transformers import SentenceTransformer, util
import torch

# 1. Sample documents for demonstration purposes
# These can be longer documents that will be chunked into smaller pieces
DOCUMENTS = [
    "RAG reduces hallucinations by grounding answers in retrieved evidence.",
    "Dense retrieval uses vector embeddings instead of keyword matching.",
    "BM25 is a sparse retrieval method based on term frequency statistics.",
    "FAISS is a fast vector similarity library from Meta.",
    "Vector databases store embeddings to enable semantic search.",
    "Cooking recipes require precise measurements and timing.",
    "Traveling to new countries helps you learn about culture and history."
]

# 2. Sample batch of queries
# These can be multiple queries to retrieve documents at once
QUERIES = [
    "How does RAG reduce hallucinations?",
    "Best methods for dense retrieval?",
    "How to cook precise recipes?"
]

# 3. Initialize SentenceTransformer model for embeddings
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

# 4. Precompute document embeddings for efficiency 
doc_embeddings = model.encode(DOCUMENTS, convert_to_tensor=True, normalize_embeddings=True)

# 5. Retrieve top-k relevant documents for each query in batch 
def batch_retrieve(queries, top_k=3):
    """
    Retrieve top_k relevant documents for each query in batch.
    """
    query_embeddings = model.encode(queries, convert_to_tensor=True, normalize_embeddings=True)
    
    # Compute cosine similarity in batch
    cos_scores = util.cos_sim(query_embeddings, doc_embeddings)  # shape: [num_queries, num_docs]
    
    results = []
    for i, query in enumerate(queries):
        top_results = torch.topk(cos_scores[i], k=min(top_k, len(DOCUMENTS)))
        retrieved = [(DOCUMENTS[idx], float(score)) for score, idx in zip(top_results.values, top_results.indices)]
        results.append((query, retrieved))
    return results

if __name__ == "__main__":
    batch_results = batch_retrieve(QUERIES, top_k=3)

    # 6. Display results for each query with scores 
    for query, retrieved_docs in batch_results:
        print(f"\nQuery: {query}")
        print("Top Documents:")
        for doc, score in retrieved_docs:
            print(f"  similarity={score:.4f} | {doc}")
