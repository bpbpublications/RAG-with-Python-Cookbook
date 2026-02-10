# chain_of_thought_guided_response.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers.pipelines import pipeline
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Load the document
loader = TextLoader("cpython hapter7_RAG.txt")   # replace with your docs
docs = loader.load()

# 2. Split the documents into chunks of 500 characters with 50 character overlap
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = splitter.split_documents(docs)

# 3. Create embeddings and vector store from documents using HuggingFaceEmbeddings and FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# 4. Generate candidate answers using local model with pure sampling
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    max_length=256,
    do_sample=True,            
    top_k=50,
    top_p=0.95,
    num_return_sequences=5,
    num_beams=1                
)

# 5. Define function to compute cosine similarity between two vectors a and b
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# 6. Define function chain_of_thought_response(query: str) that takes a query string as input
# and returns a structured response with the best answer, score, candidates considered,
# list of candidates with their scores, and sources of retrieved documents
def chain_of_thought_response(query: str):
    # retrieve context
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(query)
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

# 7. Query which will use chain of thought prompting to generate an answer
query = "Explain Retrieval-Augmented Generation in simple terms."
response = chain_of_thought_response(query)

import json
# 8. print the structured response in JSON format
print(json.dumps(response, indent=2))
