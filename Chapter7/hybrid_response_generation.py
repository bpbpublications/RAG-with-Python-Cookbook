# hybrid_response_generation.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from transformers.pipelines import pipeline
import json

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Load the document
loader = TextLoader("chapter7_RAG.txt")   # replace with your docs
docs = loader.load()

# 2. Split the documents into chunks of 400 characters with 50 character overlap
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
documents = splitter.split_documents(docs)

# 3. Create embeddings and vector store from documents using HuggingFaceEmbeddings and FAISS
dense_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
dense_store = FAISS.from_documents(documents, dense_embeddings)

# 4. Create sparse retriever using BM25
sparse_retriever = BM25Retriever.from_documents(documents)

# 5. Generate pipeline using a local model
# This model will be used for generating responses based on retrieved context
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    max_length=256
)

# 6. Create a hybrid retriever method that combines dense and sparse retrieval
def hybrid_retrieve(query: str, k=3):
    dense_docs = dense_store.as_retriever(search_kwargs={"k": k}).invoke(query)
    sparse_docs = sparse_retriever.invoke(query)

    # merge unique docs by content
    seen, merged = set(), []
    for doc in dense_docs + sparse_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            merged.append(doc)
    return merged[:k]

# 7. Create hybrid response function to generate response using multiple prompts
def hybrid_response(query: str):
    retrieved_docs = hybrid_retrieve(query, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    prompts = [
        f"Question: {query}\nContext:\n{context}\n\nAnswer briefly and clearly:",
        f"Question: {query}\nContext:\n{context}\n\nGive a detailed explanation with reasoning:",
        f"Question: {query}\nContext:\n{context}\n\nExplain in simple terms, as if teaching a beginner:"
    ]

    responses = []
    for p in prompts:
        out = generator(p)[0]["generated_text"]
        responses.append(out)

    # merge responses
    final_answer = "\n\n---\n\n".join(responses)

    return {
        "question": query,
        "final_answer": final_answer,
        "retrieved_sources": [doc.metadata.get("source", "unknown") for doc in retrieved_docs],
        "num_prompts": len(prompts),
        "num_sources": len(retrieved_docs)
    }

# 8. Create a query and generate the hybrid response using hybrid_response function
query = "Explain Retrieval-Augmented Generation (RAG)."
response = hybrid_response(query)

# 9. Print the structured response in JSON format
print(json.dumps(response, indent=2))
