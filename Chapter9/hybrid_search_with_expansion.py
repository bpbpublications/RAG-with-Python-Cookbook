# hybrid_search_with_expansion.py
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Sample documents, in practice, use a larger corpus or a vector DB
DOCS = [
    "RAG combines retrieval with generation to ground answers in evidence.",
    "Dense retrieval uses vector embeddings instead of keyword matching.",
    "FAISS is a fast vector similarity library from Facebook/Meta.",
    "BM25 is a sparse retrieval method based on term frequency statistics.",
    "Sentence-Transformers provides easy embedding models for sentences.",
    "Vector databases store embeddings to enable semantic search.",
]

# 2. Candidate expansion terms for query expansion (determined beforehand)
EXPANSION_CANDIDATES = [
    "retrieval",
    "search",
    "semantic search",
    "dense retrieval",
    "vector database",
    "BM25",
    "information retrieval",
    "keyword search",
    "ranking",
    "query understanding",
    "knowledge base",
    "embeddings",
    "document matching",
    "relevance",
    "natural language processing",
]

# 3. Load Sentence Transformer model for embeddings and similarity search 
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# 4. Build BM25 index for sparse retrieval
tokenized_docs = [doc.lower().split() for doc in DOCS]
bm25 = BM25Okapi(tokenized_docs)

# 5. Build FAISS index for dense retrieval   
embeddings = model.encode(DOCS, convert_to_numpy=True, normalize_embeddings=True)
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings) # type: ignore

# --- Query Expansion ---
def expand_query(query: str, top_k: int = 3):
    query_emb = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    cand_embs = model.encode(EXPANSION_CANDIDATES, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(query_emb, cand_embs)[0]
    top_indices = scores.argsort(descending=True)[:top_k]
    return [EXPANSION_CANDIDATES[i] for i in top_indices]

def build_expanded_query(query: str, expansions):
    return query + " " + " ".join(expansions)

# 8. bm25 search function using the expanded query terms 
def bm25_search(query: str, k: int = 3):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
    return [(float(score), DOCS[idx]) for idx, score in ranked[:k]]

# 9. dense search function using the expanded query terms
def dense_search(query: str, k: int = 3):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(q_emb, k) # type: ignore
    return [(float(scores[0][i]), DOCS[int(idxs[0][i])]) for i in range(k)]

# 10. hybrid search function combining BM25 and dense scores using the expanded query terms
def hybrid_search(query: str, k: int = 3, alpha: float = 0.5):
    tokens = query.lower().split()
    bm25_scores = np.array(bm25.get_scores(tokens))

    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    dense_scores, _ = index.search(q_emb, len(DOCS)) # type: ignore
    dense_scores = dense_scores[0]

    # Normalize
    def normalize(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)
    bm25_norm = normalize(bm25_scores)
    dense_norm = normalize(dense_scores)

    # Weighted sum
    hybrid_scores = alpha * dense_norm + (1 - alpha) * bm25_norm
    ranked = np.argsort(-hybrid_scores)
    return [(float(hybrid_scores[i]), DOCS[i]) for i in ranked[:k]]

# --- Run Demo ---
if __name__ == "__main__":
    # 6. Example query to test the system with expansion terms added in the query string for better results 
    query = "How does RAG reduce hallucinations?"

    # 7. Expand query with top 3 expansion terms  
    expansions = expand_query(query, top_k=3)
    expanded_query = build_expanded_query(query, expansions)

    # Display results  
    print(f"\nOriginal Query: {query}")
    print(f"Expanded Terms: {', '.join(expansions)}")
    print(f"Expanded Query String: {expanded_query}\n")

    print(" BM25 Results:")
    for score, text in bm25_search(expanded_query):
        print(f"  score={score:.4f} | {text}")

    print("\n Dense Results:")
    for score, text in dense_search(expanded_query):
        print(f"  score={score:.4f} | {text}")

    print("\n Hybrid Results (alpha=0.5):")
    for score, text in hybrid_search(expanded_query, alpha=0.5):
        print(f"  score={score:.4f} | {text}")
