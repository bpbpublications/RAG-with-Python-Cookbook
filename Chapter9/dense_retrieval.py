# dense_retrieval.py
# Example of dense retrieval using FAISS and Sentence-Transformers
from typing import List, Tuple
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. These are short example documents.
# You can replace this with your own documents.
DOCS = [
    "RAG combines retrieval with generation to ground answers in evidence.",
    "Dense retrieval uses vector embeddings instead of keyword matching.",
    "FAISS is a fast vector similarity library from Facebook/Meta.",
    "BM25 is a sparse retrieval method based on term frequency statistics.",
    "Sentence-Transformers provides easy embedding models for sentences.",
    "Vector databases store embeddings to enable semantic search.",
]

# 2. Use a pre-trained Sentence-Transformer model to get dense embeddings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# 3. Normalize embeddings to use cosine similarity
embeddings = model.encode(DOCS, convert_to_numpy=True, normalize_embeddings=True)
dim = embeddings.shape[1]

# 4. Build a FAISS index for fast similarity search
# Using IndexFlatIP for cosine similarity (with normalized vectors)
index = faiss.IndexFlatIP(dim)
index.add(embeddings) # type: ignore

def search(query: str, k: int = 3) -> List[Tuple[float, str]]:
    # 6. Search for the top-k most similar documents to the query string
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    # 7. Perform the search in the FAISS index 
    scores, idxs = index.search(q_emb, k)  # type: ignore # scores are cosine similarities
    # 8. Return the top-k results with their scores and texts
    return [(float(scores[0][i]), DOCS[int(idxs[0][i])]) for i in range(k)]

if __name__ == "__main__":
    # 5.Example query to test the search function 
    query = "How does RAG reduce hallucinations?"
    results = search(query, k=3)

# 9. Print the search results 
    print(f"Query: {query}\n")
    for rank, (score, text) in enumerate(results, start=1):
        print(f"{rank}. score={score:.4f}  |  {text}")
