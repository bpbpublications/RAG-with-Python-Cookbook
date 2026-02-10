# self_querying_agent.py
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from transformers import pipeline
import json

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Create a text generation pipeline using a small model for demonstration purposes 
generator = pipeline("text2text-generation", model="t5-small", max_length=256)
llm = HuggingFacePipeline(pipeline=generator)

# 2. Create a simple dataset with metadata for demonstration purposes 
docs = [
    Document(page_content="Intermittent fasting improves insulin sensitivity.", metadata={"topic": "health"}),
    Document(page_content="The capital of France is Paris.", metadata={"topic": "geography"}),
    Document(page_content="Python is a popular programming language for AI.", metadata={"topic": "technology"}),
]

# 3. Create a vector store from the documents using HuggingFace embeddings  
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# 4. Self-querying agent function which processes the question, retrieves documents, and generates an answer
def self_query_agent(question: str):
    # Step 1: Generate query + filters (JSON string)
    prompt = f"""
Convert the user question into JSON with fields:
- query: the search query text
- filters: metadata filters (dictionary, or {{}} if none)

User question: {question}
"""
    raw = llm.invoke(prompt)  # use predict() instead of LLMChain.invoke()
    try:
        parsed = json.loads(raw)
    except:
        start, end = raw.find("{"), raw.rfind("}")
        parsed = json.loads(raw[start:end+1]) if start != -1 else {"query": question, "filters": {}}

    query = parsed.get("query", question)
    filters = parsed.get("filters", {})

    retrieved_docs = vectorstore.similarity_search(query, k=2, filter=filters or None)

    context = "\n\n".join([f"{d.page_content}" for d in retrieved_docs])
    answer_prompt = f"Answer the question using the context below.\n\nQuestion: {question}\n\nContext:\n{context}"
    answer = llm.invoke(answer_prompt)

    return answer, retrieved_docs, parsed

# Demo Run
if __name__ == "__main__":
    # Example query
    q1 = "What do we know about health and insulin sensitivity?"
    ans1, docs1, parsed1 = self_query_agent(q1)

# 5. Print the results 
    print("\nParsed Query:", parsed1)
    print("\nRetrieved Docs:", [d.page_content for d in docs1])
    print("\nAnswer:", ans1)
