# hierarchical_retrieval.py
# Example of hierarchical retrieval using BM25 and Cross-Encoder reranking
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from rank_bm25 import BM25Okapi
import torch

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Sample corpus for demonstration purposes 
CORPUS = [
    "RAG reduces hallucinations by grounding answers in retrieved evidence.",
    "Dense retrieval uses vector embeddings instead of keyword matching.",
    "BM25 is a sparse retrieval method based on term frequency statistics.",
    "FAISS is a fast vector similarity library from Meta.",
    "Vector databases store embeddings to enable semantic search.",
    "Cooking recipes often require precise measurements and timing.",
    "Traveling to new countries helps you learn about culture and history."
]

# 2. Initialize BM25 with the corpus documents tokenized into words  
tokenized_corpus = [doc.lower().split() for doc in CORPUS]
bm25 = BM25Okapi(tokenized_corpus)

# 4. Coarse retrieval function to get top-k documents using bm25 
def coarse_retrieve(query, top_k=5):
    tokenized_query = query.lower().split()
    doc_scores = bm25.get_scores(tokenized_query)
    top_indices = doc_scores.argsort()[-top_k:][::-1]
    return [CORPUS[i] for i in top_indices]

# 3. Initialize Cross-Encoder for fine reranking  
cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
cross_encoder = CrossEncoder(cross_encoder_model)

# 5. Fine reranking to refine the coarse results using Cross-Encoder 
def fine_rerank(query, candidates):
    pairs = [(query, doc) for doc in candidates]
    scores = cross_encoder.predict(pairs)
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return reranked

if __name__ == "__main__":
    # Example query
    query = "How does RAG reduce hallucinations?"
    
    coarse_results = coarse_retrieve(query, top_k=5)
    # 6. Print coarse retrieval results.
    print(f"Coarse Retrieval ({len(coarse_results)} results):")
    for doc in coarse_results:
        print(f"  {doc}")
    
    fine_results = fine_rerank(query, coarse_results)
    # 7. Print fine reranked results.
    print(f"\nFine Reranked Results:")
    for doc, score in fine_results:
        print(f"  score={score:.4f} | {doc}")
