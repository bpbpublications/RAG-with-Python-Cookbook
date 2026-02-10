# tool_augmented_rag_chain.py
# This code demonstrates a tool-augmented retrieval-augmented generation (RAG) chain
import re
from langchain_huggingface import HuggingFacePipeline
from transformers.pipelines import pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Initialize the local LLM using HuggingFace's transformers pipeline
local_pipe = pipeline("text2text-generation", model="google/flan-t5-base")
llm = HuggingFacePipeline(pipeline=local_pipe)

# Define the calculator tool
def calculator_tool(query: str) -> str:
    expr_match = re.findall(r"[0-9\.\+\-\*\/\(\)]+", query)
    if not expr_match:
        return "No math expression found."
    expr = "".join(expr_match)
    allowed_names = {"abs": abs, "round": round, "pow": pow, "min": min, "max": max}
    try:
        return str(eval(expr, {"__builtins__": {}}, allowed_names))
    except Exception as e:
        return f"Error: {e}"

# 2. Sample documents for RAG Tool 
# In a real-world scenario, these would be replaced with a larger and more relevant dataset.
docs = [
    Document(page_content="Alice wrote about intermittent fasting and health."),
    Document(page_content="The 2008 financial crisis impacted global markets."),
    Document(page_content="The French Revolution began in 1789.")
]

# 3. Initialize embeddings and create the vector store for RAG Tool 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding=embeddings)

# RAG Tool function to retrieve relevant documents and generate a response
def rag_tool(query: str) -> str:
    # return only the most relevant document
    results = vectorstore.similarity_search(query, k=1)  
    if not results:
        return "No relevant documents found."
    return results[0].page_content

# Combined tool agent function
def tool_agent(query: str) -> str:
    if any(op in query for op in ["+", "-", "*", "/", "**", "(", ")"]):
        return calculator_tool(query)
    else:
        return rag_tool(query)

# 4. Example query of the tool-augmented RAG chain
queries = [
    "What is 25 * 12?",
    "Calculate 100 / 4 + 7",
    "Tell me about financial events after 2005",
    "History events before 1900"
]

# 5. Execute the tool-augmented RAG chain with example queries and print the results 
for q in queries:
    print("\nQuery:", q)
    answer = tool_agent(q)
    print("Answer:", answer)
