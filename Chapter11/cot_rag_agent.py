# cot_rag_agent.py
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from transformers.pipelines import pipeline

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Generate a pipeline for text generation using a local model (e.g., t5-small)
generator = pipeline("text2text-generation", model="t5-small", max_length=256)
llm = HuggingFacePipeline(pipeline=generator)

# 2. Create a sample document 
rag_docs = [
    Document(
        page_content="RAG combines retrieval and generation to improve answer accuracy. "
                     "It retrieves relevant documents before generating responses.",
        metadata={"topic": "RAG basics"}
    ),
    Document(
        page_content="Dense retrieval uses embeddings to find semantically similar text chunks. "
                     "FAISS is commonly used for fast vector retrieval.",
        metadata={"topic": "Dense Retrieval"}
    ),
    Document(
        page_content="Chain-of-Thought prompting encourages step-by-step reasoning for complex queries. "
                     "It helps the model produce more logical answers.",
        metadata={"topic": "CoT"}
    ),
]

# 3. Create vector store from documents using HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(rag_docs, embeddings)

# Chain-of-Thought Retrieval Agent
def cot_rag_agent(query: str) -> str:
    # Retrieve top 2 relevant RAG documents
    retrieved_docs = vectorstore.similarity_search(query, k=2)
    if not retrieved_docs:
        return "No relevant information found."

    # Chain-of-Thought reasoning prompt
    context_text = " ".join([doc.page_content for doc in retrieved_docs])
    prompt = (
        f"Think step by step and answer the question using the context from the RAG documents.\n\n"
        f"Context: {context_text}\n"
        f"Question: {query}\nAnswer:"
    )

    # Generate answer
    response = llm.invoke(prompt).strip()
    return response

if __name__ == "__main__":
    queries = [
        "What is RAG and how does it help?",
        "Explain dense retrieval in simple terms.",
        "Why use Chain-of-Thought prompting?"
    ]

# 4. Based on query, return response based on Chain-Of-Thoughts and print the responses
    for q in queries:
        print("\n--- User ---")
        print(q)
        answer = cot_rag_agent(q)
        print("\n--- Assistant ---")
        print(answer)
