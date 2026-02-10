# query_expansion_agent.py
import nltk
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---------------- Setup ----------------
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 2   # results per expanded query

# 1. Create a simple dataset with id for demonstration purposes
DOCS = [
    {"id": "doc1", "text": "Alice wrote about intermittent fasting and health."},
    {"id": "doc2", "text": "The 2008 financial crisis impacted global markets."},
    {"id": "doc3", "text": "The French Revolution began in 1789."},
    {"id": "doc4", "text": "Research shows fasting improves insulin sensitivity."},
    {"id": "doc5", "text": "Global financial markets collapsed in 2008."},
]

# ---------------- Query Expansion ----------------
def expand_query(query: str, max_synonyms: int = 2):
    tokens = query.lower().split()
    expansions = set([query])  # include original

    for word in tokens:
        for syn in wn.synsets(word):
            for lemma in syn.lemmas()[:max_synonyms]: # type: ignore
                new_q = query.replace(word, lemma.name().replace("_", " "))
                expansions.add(new_q)
    return list(expansions)


# ---------------- Dense Index ----------------
class DenseRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.model = SentenceTransformer(EMBED_MODEL)
        self.embeddings = np.array(self.model.encode([d["text"] for d in docs], convert_to_numpy=True))
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings) # type: ignore

    def search(self, query: str, k: int = TOP_K):
        q_vec = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_vec, k) # type: ignore
        results = []
        for idx, dist in zip(I[0], D[0]):
            results.append({"id": self.docs[idx]["id"], "text": self.docs[idx]["text"], "score": float(dist)})
        return results


# ---------------- Query Expansion Agent ----------------
class QueryExpansionAgent:
    def __init__(self, docs):
        self.retriever = DenseRetriever(docs)

    def run(self, query: str):
        expanded = expand_query(query)
        print(f"Original Query: {query}")
        print(f"Expanded Queries: {expanded}\n")

        seen = {}
        for q in expanded:
            hits = self.retriever.search(q, k=TOP_K)
            for h in hits:
                if h["id"] not in seen or h["score"] < seen[h["id"]]["score"]:
                    seen[h["id"]] = h
        # sort by similarity score
        results = sorted(seen.values(), key=lambda x: x["score"])
        return results


# ---------------- Example ----------------
if __name__ == "__main__":
    agent = QueryExpansionAgent(DOCS)

# 2. Create queries to demonstrate query expansion 
    queries = [
        "Tell me about health",
        "Financial events after 2005",
        "History before 1900"
    ]

# 3. For each query, perform query expansion and use dense retriever to get the response.
# 4. Print the query and received responses
    for q in queries:
        results = agent.run(q)
        print(f"Results for '{q}':")
        for r in results[:3]:
            print(f"{r['text']} (id={r['id']}, score={r['score']:.4f})")
        print("-" * 60)
