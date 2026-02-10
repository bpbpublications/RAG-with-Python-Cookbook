# adaptive_summarization_agent.py
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from transformers.pipelines import pipeline
import re

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Generate a pipeline for text generation using a local model (e.g., t5-small)
generator = pipeline("text2text-generation", model="t5-small", max_length=256)
llm = HuggingFacePipeline(pipeline=generator)

# 2. Create a sample document store 
docs = [
    Document(page_content="The capital of France is Paris.", metadata={"topic": "geography"}),
]

# 3. Create vector store from documents using HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

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

# Adaptive Summarization Agent
def summarize_query(query: str) -> str:
    retrieved_docs = vectorstore.similarity_search(query, k=3)
    if not retrieved_docs:
        return "No relevant information found."

    # If top document seems highly relevant, return it directly
    top_doc = retrieved_docs[0]
    if len(retrieved_docs) == 1 or query.lower() in top_doc.page_content.lower():
        return top_doc.page_content

    # Otherwise, summarize top documents
    context = " ".join([doc.page_content for doc in retrieved_docs])
    prompt = f"Summarize the following information in 1-2 concise English sentences:\n{context}"
    response = generator(prompt)[0]['generated_text'].strip()
    return response

# Combined agent with math handling
def agent(user_input: str) -> str:
    if is_math_task(user_input):
        return calculator_tool(user_input)
    else:
        return summarize_query(user_input)

# Example conversation
if __name__ == "__main__":
# 4. Sample inputs to test the agent with adaptive summarization and math handling
    inputs = [
        "What is the capital of France?",
        "What is 10 plus 5?"
    ]
# 5. Run the agent with example inputs and print responses 
    for user_input in inputs:
        print("\n--- User ---")
        print(user_input)
        answer = agent(user_input)
        print("\n--- Assistant ---")
        print(answer)
