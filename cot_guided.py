from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import numpy as np

# -----------------------------
# 1. Load and split documents
# -----------------------------
loader = TextLoader("chapter7_RAG.txt")   # replace with your docs
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = splitter.split_documents(docs)

# -----------------------------
# 2. Vector store
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# -----------------------------
# 3. Local model (pure sampling)
# -----------------------------
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    max_length=256,
    do_sample=True,            
    top_k=50,
    top_p=0.95,
    num_return_sequences=5,
    num_beams=1                # disable beam search
)

# -----------------------------
# 4. Semantic Scoring Function
# -----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def chain_of_thought_response(query: str):
    # retrieve context
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # reasoning prompt
    prompt = f"""Question: {query}
Relevant Context:
{context}

Think step by step and provide a clear, factual answer.
"""

    # generate 5 candidate answers
    candidates = generator(prompt)

    # embed query+context for scoring
    reference_text = query + " " + context
    reference_vec = embeddings.embed_query(reference_text)

    # score each candidate with cosine similarity
    scored = []
    for cand in candidates:
        cand_text = cand["generated_text"]
        cand_vec = embeddings.embed_query(cand_text)
        score = cosine_similarity(reference_vec, cand_vec)
        scored.append((cand_text, score))

    # pick the best candidate
    best_answer, best_score = max(scored, key=lambda x: x[1])

    # return structured result
    return {
        "question": query,
        "answer": best_answer,
        "score": round(float(best_score), 3),
        "candidates_considered": len(candidates),
        "candidates": [
            {"text": text, "score": round(float(score), 3)}
            for text, score in scored
        ],
        "sources": list({doc.metadata.get("source", "unknown") for doc in retrieved_docs})
    }

# -----------------------------
# 5. Run demo
# -----------------------------
query = "Explain Retrieval-Augmented Generation in simple terms."
response = chain_of_thought_response(query)

import json
print(json.dumps(response, indent=2))
