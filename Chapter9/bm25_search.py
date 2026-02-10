# bm25_search.py
from rank_bm25 import BM25Okapi

# 1. For demonstration, we use a small set of example documents
DOCS = [
    "RAG combines retrieval with generation to ground answers in evidence.",
    "Dense retrieval uses vector embeddings instead of keyword matching.",
    "FAISS is a fast vector similarity library from Facebook/Meta.",
    "BM25 is a sparse retrieval method based on term frequency statistics.",
    "Sentence-Transformers provides easy embedding models for sentences.",
    "Vector databases store embeddings to enable semantic search.",
]

# 2. Preprocess documents using simple tokenization and lowercasing
tokenized_docs = [doc.lower().split() for doc in DOCS]

# 3. Initialize BM25 with the tokenized documents  
bm25 = BM25Okapi(tokenized_docs)

def search(query: str, k: int = 3):
    # 5. Preprocess the query using the same tokenization method as documents 
    tokenized_query = query.lower().split()
    # 6. Get BM25 scores and rank documents based on these scores 
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
    # 7. Return the top-k results with their scores and original text 
    return [(score, DOCS[idx]) for idx, score in ranked[:k]]

if __name__ == "__main__":
    # 4. Sample query to test the search function 
    query = "How does RAG reduce hallucinations?"
    results = search(query, k=3)

    print(f"Query: {query}\n")

    # 8. Display results with rank, score, and text  
    for rank, (score, text) in enumerate(results, start=1):
        print(f"{rank}. score={score:.4f}  |  {text}")
