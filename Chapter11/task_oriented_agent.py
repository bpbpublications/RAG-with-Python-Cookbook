# task_oriented_agent.py
# A task-oriented agent that can handle both document retrieval and simple math calculations.
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from transformers import pipeline
import re
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Load small LLM locally using HuggingFace pipeline for text generation 
generator = pipeline("text2text-generation", model="t5-small", max_length=256)
llm = HuggingFacePipeline(pipeline=generator)

# 2. Create a simple dataset with metadata for demonstration purpose
docs = [
    Document(page_content="The capital of France is Paris.", metadata={"topic": "geography"}),
    Document(page_content="Python is a popular programming language for AI.", metadata={"topic": "technology"}),
    Document(page_content="Intermittent fasting improves insulin sensitivity.", metadata={"topic": "health"}),
]

# 3. Create a vector store from the documents using HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# Tool for simple math calculations
def calculator_tool(query: str) -> str:
    """Evaluate simple math expressions."""
    query = query.lower()
    query = query.replace("multiplied by", "*")
    query = query.replace("times", "*")
    query = query.replace("plus", "+")
    query = query.replace("minus", "-")
    query = query.replace("divided by", "/")
    query = re.sub(r"[^0-9\+\-\*\/\.\(\) ]", "", query)
    try:
        return str(eval(query))
    except Exception as e:
        return f"Error: {e}"

# Tool for document retrieval and summarization
def doc_search_tool(query: str) -> str:
    """Retrieve the most relevant document and summarize it."""
    # Retrieve only the top document
    retrieved_docs = vectorstore.similarity_search(query, k=1)
    if not retrieved_docs:
        return "No relevant information found."
    context = retrieved_docs[0].page_content
    # Generate a clean, full-sentence answer
    prompt = f"Answer the question using the following context. Make your answer concise and in full sentences.\n\nQuestion: {query}\nContext: {context}"
    return llm.invoke(prompt)

# Simple computing to determine if the task is a math problem or a document retrieval
def is_math_task(task: str) -> bool:
    task_lower = task.lower()
    if re.search(r"\d", task_lower):
        return True
    math_words = ["calculate", "plus", "minus", "multiply", "times", "divided by", "add"]
    return any(word in task_lower for word in math_words)

# Task-oriented agent function
def task_oriented_agent(task: str) -> str:
    if is_math_task(task):
        return calculator_tool(task)
    else:
        return doc_search_tool(task)

if __name__ == "__main__":
    # 4. Task examples to demonstrate retrieval and calculation capabilities of the agent
    tasks = [
        "What is 25 multiplied by 4?",
        "Tell me about health and insulin sensitivity.",
        "What is the capital of France?"
    ]

# 5. Run the loop for each task to demonstrate the agent's capabilities of retrieval and calculation and print the results
    for task in tasks:
        print("\n--- Task ---")
        print(task)
        result = task_oriented_agent(task)
        print("\nAnswer:", result)
