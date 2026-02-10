# hybrid_search.py
# Hybrid search combining BM25 and Dense Retrieval using FAISS and Sentence-Transformers
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Create a small document corpus
# In practice, use a larger document or a vector database
# For demonstration, we use a small set of example documents
DOCS = [
    "RAG combines retrieval with generation to ground answers in evidence.",
    "Dense retrieval uses vector embeddings instead of keyword matching.",
    "FAISS is a fast vector similarity library from Facebook/Meta.",
    "BM25 is a sparse retrieval method based on term frequency statistics.",
    "Sentence-Transformers provides easy embedding models for sentences.",
    "Vector databases store embeddings to enable semantic search.",
]

# 2. Tokenize documents for BM25 index using simple tokenization and lowercasing
tokenized_docs = [doc.lower().split() for doc in DOCS]
bm25 = BM25Okapi(tokenized_docs)

# 3. Create FAISS index for dense retrieval using Sentence-Transformers embeddings 
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(DOCS, convert_to_numpy=True, normalize_embeddings=True)
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings) # type: ignore

def bm25_search(query: str, k: int = 3):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
    return [(float(score), DOCS[idx]) for idx, score in ranked[:k]]

def dense_search(query: str, k: int = 3):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(q_emb, k) # type: ignore
    return [(float(scores[0][i]), DOCS[int(idxs[0][i])]) for i in range(k)]

def hybrid_search(query: str, k: int = 3, alpha: float = 0.5):
    """alpha=0 → BM25 only, alpha=1 → Dense only"""
    tokens = query.lower().split()
    bm25_scores = np.array(bm25.get_scores(tokens))

    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    dense_scores, _ = index.search(q_emb, len(DOCS)) # type: ignore
    dense_scores = dense_scores[0]

    # Normalize
    def normalize(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)
    bm25_norm = normalize(bm25_scores)
    dense_norm = normalize(dense_scores)

    # Hybrid scoring
    hybrid_scores = alpha * dense_norm + (1 - alpha) * bm25_norm
    ranked = np.argsort(-hybrid_scores)
    return [(float(hybrid_scores[i]), DOCS[i]) for i in ranked[:k]]

# Run Demo
if __name__ == "__main__":
    # 4. Example query to test the search functions using BM25, Dense, and Hybrid methods
    query = "How does RAG reduce hallucinations?"

    print(f"\nQuery: {query}\n")

    print(" BM25 Results:")
    # 5. Perform BM25 search and Print results with scores
    for score, text in bm25_search(query):
        print(f"  score={score:.4f} | {text}")

    print("\n Dense Results:")
    # 6. Perform Dense search and Print results with scores
    for score, text in dense_search(query):
        print(f"  score={score:.4f} | {text}")

    print("\n Hybrid Results (alpha=0.5):")
    # 7. Perform Hybrid search and Print results with scores
    for score, text in hybrid_search(query, alpha=0.5):
        print(f"  score={score:.4f} | {text}")
