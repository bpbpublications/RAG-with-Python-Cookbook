# critical_verification_response.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers.pipelines import pipeline
import numpy as np
import json

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Load the  documents
loader = TextLoader("chapter7_RAG.txt")   # replace with your docs
docs = loader.load()

# 2. Split the document into chuncks with chunk size as 500 and overlap as 50
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = splitter.split_documents(docs)

# 3. Create embeddings and vector store from documents using HuggingFaceEmbeddings and FAISS.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# NLI model for claim verification
nli_model = pipeline("text-classification", model="facebook/bart-large-mnli")

# Claim Verification Function
def verify_claim(claim, retrieved_docs):
    scores = []
    for doc in retrieved_docs:
        text = doc.page_content[:500]  # check snippet
        result = nli_model(f"{claim} </s> {text}", truncation=True)[0]
        scores.append(result)

    # pick best evidence
    best = max(scores, key=lambda x: x["score"])
    return best["label"], float(f"{best['score']:.2f}")

# Critical Verification Response
def critical_verification_response(query: str, answer: str):
    # retrieve evidence
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(query)

    # break answer into claims (simple sentence split)
    claims = [c.strip() for c in answer.split(".") if c.strip()]

    results = []
    for claim in claims:
        label, score = verify_claim(claim, retrieved_docs)
        results.append({
            "claim": claim,
            "verdict": label,
            "confidence": score
        })

    # aggregate metrics
    support_ratio = sum(1 for r in results if r["verdict"] == "ENTAILMENT") / len(results)
    contradiction_ratio = sum(1 for r in results if r["verdict"] == "CONTRADICTION") / len(results)

    return {
        "question": query,
        "answer": answer,
        "verification": results,
        "support_ratio": float(f"{support_ratio:.2f}"),
        "contradiction_ratio": float(f"{contradiction_ratio:.2f}")
    }

# 4. Create query
query = "What is Retrieval-Augmented Generation?"
answer = "Retrieval-Augmented Generation is a method that combines document retrieval with text generation. It improves factual accuracy."

# 5. Perform critical verification
response = critical_verification_response(query, answer)

# 6. Print the response after performing critical verification
print(json.dumps(response, indent=2))
