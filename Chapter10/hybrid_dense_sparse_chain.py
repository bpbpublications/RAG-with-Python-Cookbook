# hybrid_dense_sparse_chain.py
from typing import List

# 1. Example documents with topics for dense and sparse retrieval
documents = [
    {"id": "doc1", "topic": "health", "text": "Alice wrote about intermittent fasting and health."},
    {"id": "doc2", "topic": "finance", "text": "The 2008 financial crisis impacted global markets."},
    {"id": "doc3", "topic": "history", "text": "The French Revolution began in 1789."},
]

# 2. Keywords for topics in dense retrieval simulation
topic_keywords = {
    "health": ["health", "medical", "wellness", "intermittent fasting"],
    "finance": ["finance", "financial", "market", "crisis"],
    "history": ["history", "revolution", "before 1900"]
}

# Dense retriever simulates semantic search using topic keywords
class DenseRetriever:
    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query: str, top_k: int = 5):
        query_lower = query.lower()
        relevant_docs = []
        for doc in self.docs:
            keywords = topic_keywords.get(doc["topic"], [])
            if any(k.lower() in query_lower for k in keywords):
                relevant_docs.append({"id": doc["id"], "text": doc["text"], "score": 1.0})
        return relevant_docs[:top_k]

# Sparse retriever simulates keyword matching
class SparseRetriever:
    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query: str, top_k: int = 5):
        # simple word matching
        query_words = set(query.lower().split())
        relevant_docs = []
        for doc in self.docs:
            doc_words = set(doc["text"].lower().split())
            if query_words & doc_words:  # intersection
                relevant_docs.append({"id": doc["id"], "text": doc["text"], "score": 1.0})
        return relevant_docs[:top_k]

# Hybrid retriever combining dense and sparse results
class HybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, top_k=2):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.top_k = top_k

    def retrieve(self, query: str):
        dense_results = self.dense.retrieve(query, top_k=self.top_k)
        sparse_results = self.sparse.retrieve(query, top_k=self.top_k)
        # Merge by unique doc id
        seen = set()
        merged = []
        for r in dense_results + sparse_results:
            if r["id"] not in seen:
                merged.append(r)
                seen.add(r["id"])
        return merged[:self.top_k]

# 5. Simple QA chain that formats answers with sources from retriever results 
def qa_chain(query: str, retriever: HybridRetriever):
    results = retriever.retrieve(query)
    if not results:
        return "No relevant documents found."
    output = ""
    for r in results:
        output += f"Answer: {r['text']} (Source: {r['id']})\n"
    return output.strip()

# 3. Setup dense, sparse and hybrid retriever  
dense_retriever = DenseRetriever(documents)
sparse_retriever = SparseRetriever(documents)
hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever, top_k=2)

# 4. Prepare example queries
queries = [
    "Tell me about health",
    "Financial events after 2005",
    "History events before 1900"
]

# 6. Execute Queries and Print Results with Sources 
for q in queries:
    print(f"Query: {q}")
    answer = qa_chain(q, hybrid_retriever)
    print(answer)
    print()
