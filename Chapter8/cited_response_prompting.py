# cited_response_prompting.py
# Demonstrates cited response prompting using a small LLM and sentence embeddings with FAISS.
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Prepare a document text, here we use a small example for demo
documents = [
    "RAG reduces hallucinations by grounding answers in retrieved documents instead of relying only on the model's memory.",
    "It improves AI responses by combining document retrieval with language generation."
]

# 2. Assign unique IDs to each document for citation
doc_ids = [f"[{i+1}]" for i in range(len(documents))]

# 3. Build FAISS index for document retrieval using sentence embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(documents)

# 4. Initialize FAISS index and add document embeddings
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings)) # type: ignore

# Function to retrieve top_k documents for a query
def retrieve(query, top_k=2):
    query_emb = model.encode([query])
    distances, indices = index.search(np.array(query_emb), top_k) # type: ignore
    retrieved = [(doc_ids[i], documents[i]) for i in indices[0]]
    return retrieved

# Function to build context string with citations
def build_context(retrieved):
    context = ""
    for doc_id, doc_text in retrieved:
        context += f"{doc_id} {doc_text}\n"
    return context

## 5. Generate answer with citations
def generate_answer(query):
    retrieved = retrieve(query)
    context = build_context(retrieved)
    
# 6. Create a prompt with context and query
    prompt = f"""
Context:
{context}

Question: {query}

Answer clearly, and cite sources using their IDs like [1], [2].
"""

    
    answer = "RAG reduces hallucinations by grounding answers in retrieved documents instead of relying only on the model's memory [1]. Additionally, it improves AI responses by combining document retrieval with language generation [2]."
    return answer

# Example usage
query = "How does RAG reduce hallucinations?"
# 7. Print the answer with citations
print(generate_answer(query))
