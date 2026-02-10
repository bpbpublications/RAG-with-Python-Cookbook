# semantic_filtering.py
from sentence_transformers import SentenceTransformer, util
import torch
from torch.nn.functional import normalize

# 1. Create sample documents to filter based on semantic similarity
DOCS = [
    "RAG reduces hallucinations by grounding answers in retrieved evidence.",
    "Dense retrieval uses vector embeddings instead of keyword matching.",
    "FAISS is a fast vector similarity library from Meta.",
    "BM25 is a sparse retrieval method based on term frequency statistics.",
    "Sentence-Transformers provides easy embedding models for sentences.",
    "Vector databases store embeddings to enable semantic search.",
    "Cooking recipes often require precise measurements and timing.",
    "Traveling to new countries helps you learn about culture and history."
]

# 2. Load pre-trained SentenceTransformer model
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

# 3. Compute using cosine similarity and filter documents
# Keep only those above a certain threshold 
# (e.g., 0.5 for moderate similarity)
def semantic_filter(query: str, docs, threshold: float = 0.5):
    """
    Compute cosine similarity between query and docs,
    normalize embeddings to avoid negative/weird values,
    and filter by threshold.
    """
    # Encode and normalize
    query_emb = model.encode(query, convert_to_tensor=True)
    query_emb = normalize(query_emb, p=2, dim=0)

    doc_embs = model.encode(docs, convert_to_tensor=True)
    doc_embs = normalize(doc_embs, p=2, dim=1)

    # Cosine similarity
    similarities = util.cos_sim(query_emb, doc_embs)[0]

    print("\n--- All Docs with Scores ---")
    for i, score in enumerate(similarities):
        print(f"{score:.4f} | {docs[i]}")

    # Keep only docs above threshold
    filtered = [(docs[i], float(similarities[i])) for i in range(len(docs)) if similarities[i] >= threshold]
    filtered.sort(key=lambda x: x[1], reverse=True)

    return filtered

if __name__ == "__main__":
    query = "How does RAG reduce hallucinations?"
    results = semantic_filter(query, DOCS, threshold=0.5)

# 4. Display filtered results only above threshold 0.5 
    print(f"\nQuery: {query}\n")
    print("Filtered Results (similarity > 0.5):")
    for text, score in results:
        print(f"  similarity={score:.4f} | {text}")
