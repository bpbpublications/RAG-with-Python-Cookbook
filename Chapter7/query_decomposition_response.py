# query_decomposition_response.py
from collections import defaultdict
import re
import textwrap

# 1. Simple in-memory knowledge base of documents
# In a real scenario, this would be replaced with a proper document store
DOCS = {
    "doc1": """
    Retrieval-Augmented Generation (RAG) combines a retriever with a generator.
    The retriever pulls relevant passages from a knowledge base.
    The generator uses those passages to produce grounded answers and reduce hallucinations.
    """,
    "doc2": """
    Query decomposition breaks a complex question into simpler sub-questions.
    It improves coverage, lets you retrieve per sub-question, and merge answers.
    It is useful when a question asks for multiple facts or steps.
    """,
    "doc3": """
    A simple retriever can score passages by token overlap (e.g., Jaccard similarity).
    More advanced retrievers use TF-IDF, BM25, or dense embeddings.
    """,
    "doc4": """
    To merge answers, order sub-answers logically and keep citations if available.
    When unsure, say so and surface the evidence used to answer.
    """,
}

# Splitting a sentences
def split_sentences(text: str):
    # Simple sentence splitter; good enough for the demo
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if s.strip()]

KB = []
for doc_id, raw in DOCS.items():
    for sent in split_sentences(raw):
        KB.append({"doc": doc_id, "text": sent})

# Normalize text for token overlap
def normalize(text: str):
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()

# Simple Jaccard similarity for token overlap
def jaccard(a_tokens, b_tokens):
    a, b = set(a_tokens), set(b_tokens)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)

# Decompose query into sub-queries
def decompose_query(query: str):
    """
    Super-simple heuristic decomposition:
    - Split on connectors (and/also/then/;/.), question marks
    - Keep non-empty, trimmed parts
    """
    # Preserve question words by splitting on connectors but not removing words
    chunks = re.split(r'\b(?:and then|and|also|then)\b|[?;]|(?<=\.)', query, flags=re.I)
    subs = [c.strip(" .") for c in chunks if c and c.strip(" .")]
    # If nothing sensible found, fall back to the whole query
    return subs or [query.strip()]

# Retrieval method based on token overlap
def retrieve_snippets(subquery: str, k: int = 3):
    q_tokens = normalize(subquery)
    scored = []
    for item in KB:
        s_tokens = normalize(item["text"])
        score = jaccard(q_tokens, s_tokens)
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored[:k]]

# Answer a subquery using retrieved snippets
def answer_subquery(subquery: str):
    hits = retrieve_snippets(subquery, k=3)
    if not hits:
        return {
            "subquery": subquery,
            "answer": "No direct match found in the knowledge base.",
            "evidence": [],
        }
    # Stitch the top snippets as a concise answer
    unique = []
    seen = set()
    for h in hits:
        if h["text"] not in seen:
            unique.append(h)
            seen.add(h["text"])
    evidence = [f'{h["text"]} (from {h["doc"]})' for h in unique]
    answer_text = " ".join(h["text"] for h in unique)
    return {"subquery": subquery, "answer": answer_text, "evidence": evidence}

# Compose final structured response
def compose_response(query: str, subanswers):
    lines = [f"Original query: {query}", "", "Subquery:"]
    for i, a in enumerate(subanswers, 1):
        lines.append(f"{i}. {a['subquery']}")
    lines.append("")
    lines.append("Subquery Answers:")
    for i, a in enumerate(subanswers, 1):
        wrapped = textwrap.fill(a["answer"], width=88)
        lines.append(f"{i}) {wrapped}")
    lines.append("")
    lines.append("Evidence used:")
    for i, a in enumerate(subanswers, 1):
        for ev in a["evidence"]:
            lines.append(f"- [{i}] {ev}")
    return "\n".join(lines)

# Demo run
if __name__ == "__main__":
# 1. Create a complex query that needs decomposition
    complex_query = (
        "What is RAG and why is query decomposition helpful, "
        "also mention a simple retrieval method and how to merge the answers?"
    )
# 2. Decompose the query into sub-queries
    subqs = decompose_query(complex_query)
# 3. Answer each subquery
    subanswers = [answer_subquery(sq) for sq in subqs]
# 4. Compose and print the final structured response
    print(compose_response(complex_query, subanswers))
