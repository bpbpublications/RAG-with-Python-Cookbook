# dynamic_reranking_agent.py
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from transformers.pipelines import pipeline
import re

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Load small LLM locally using HuggingFace pipeline for text generation 
generator = pipeline("text2text-generation", model="t5-small", max_length=256)
llm = HuggingFacePipeline(pipeline=generator)

# 2. Create a simple dataset with metadata for demonstration purposes 
docs = [
    Document(page_content="The capital of France is Paris.", metadata={"topic": "geography"}),
    Document(page_content="Python is a popular programming language for AI.", metadata={"topic": "technology"}),
    Document(page_content="Regular exercise helps improve insulin sensitivity.", metadata={"topic": "health"}),
]

# 3. Create a vector store from the documents using HuggingFace embeddings  
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# Document search and simple scoring
def filtered_similarity_search(query, k=2):
    results = vectorstore.similarity_search(query, k=k)
    query_words = set(query.lower().split())
    scored_docs = []
    for doc in results:
        score = len(set(doc.page_content.lower().split()) & query_words)
        scored_docs.append((score, doc))
    # sort descending by score
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    return scored_docs

# Calculator tool
def calculator_tool(query: str) -> str:
    query = query.lower()
    query = query.replace("multiplied by", "*").replace("times", "*")
    query = query.replace("plus", "+").replace("minus", "-")
    query = query.replace("divided by", "/")
    query = re.sub(r"[^0-9\+\-\*\/\.\(\) ]", "", query)
    try:
        return str(eval(query))
    except Exception as e:
        return f"Error: {e}"

def is_math_task(task: str) -> bool:
    return bool(re.search(r"\d", task)) or any(
        word in task.lower() for word in ["plus", "minus", "multiply", "times", "divided by", "add"]
    )

# Agent function
def agent(user_input: str) -> str:
    if is_math_task(user_input):
        return calculator_tool(user_input)
    
    # Retrieve and rank relevant documents
    scored_docs = filtered_similarity_search(user_input, k=2)
    
    print("\n--- Retrieved & Ranked Documents ---")
    for i, (score, doc) in enumerate(scored_docs, start=1):
        print(f"{i}. Score: {score}, Content: {doc.page_content}")
    
    # Take only top document for context to avoid repetition
    context = scored_docs[0][1].page_content if scored_docs else ""
    
    # Simple prompt: avoid repeating question in multiple places
    prompt = f"Context: {context}\nAnswer the following question concisely:\n{user_input}"
    
    response = llm.invoke(prompt).strip()
    
    # Remove any repeated question or instruction that T5 sometimes adds
    response = re.sub(r"^.*?Answer\s*[:\-]?", "", response, flags=re.IGNORECASE).strip()
    
    return response

# Example conversation
if __name__ == "__main__":
    inputs = [
        "What is the capital of France?",
        "Which programming language is popular for AI?",
        "What is 10 plus 5?"
    ]

# 4. Dynamically re-rank the retrieved doc and print the response based on ranking
    for user_input in inputs:
        print("\n--- User ---")
        print(user_input)
        answer = agent(user_input)
        print("\n--- Assistant ---")
        print(answer)
