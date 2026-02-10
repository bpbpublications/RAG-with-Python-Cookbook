# context_aware_conversational_agent.py
# A context-aware conversational agent that uses a local LLM, a document store,
# and a calculator tool to answer user queries with conversation memory.
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from transformers import pipeline
import re
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Create a pipeline for text generation using a local model (e.g., t5-small) 
generator = pipeline("text2text-generation", model="t5-small", max_length=256)
llm = HuggingFacePipeline(pipeline=generator)

# 2. Create a sample document store
docs = [
    Document(page_content="The capital of France is Paris.", metadata={"topic": "geography"}),
    Document(page_content="Python is a popular programming language for AI.", metadata={"topic": "technology"}),
    Document(page_content="Intermittent fasting improves insulin sensitivity.", metadata={"topic": "health"}),
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

# Document search tool 
def doc_search_tool(query: str) -> str:
    retrieved_docs = vectorstore.similarity_search(query, k=1)
    if not retrieved_docs:
        return "No relevant information found."
    context = retrieved_docs[0].page_content
    prompt = (
        f"Answer the question concisely in full sentences using the following context if relevant.\n\n"
        f"Context: {context}\n"
        f"Question: {query}\nAnswer:"
    )
    response = llm.invoke(prompt)
    # Remove instruction echoing
    response = re.sub(r"(Answer concisely.*|Benutzer:)", "", response, flags=re.IGNORECASE).strip()
    return response

# Conversation history to maintain context
conversation_history = []

# Context-aware conversational agent with memory and repetition avoidance 
def context_aware_agent(user_input: str) -> str:
    """
    Generate a response considering conversation history and relevant documents.
    Repetition-free.
    """
    retrieved_info = doc_search_tool(user_input)

    # Minimal prompt with context only
    prompt_for_llm = (
        f"Using the following context, answer the question concisely in full sentences.\n\n"
        f"Context: {retrieved_info}\n"
        f"Question: {user_input}\nAnswer:"
    )

    response = llm.invoke(prompt_for_llm)
    # Remove repeated input or instruction echo
    response = re.sub(rf"{re.escape(user_input)}", "", response, flags=re.IGNORECASE).strip()
    response = re.sub(r"(Answer concisely.*|User:.*|Benutzer:)", "", response, flags=re.IGNORECASE).strip()

    # Update conversation history
    conversation_history.append(f"User: {user_input}")
    conversation_history.append(f"Assistant: {response}")

    return response

# Math-aware conversational agent   
def agent(user_input: str) -> str:
    if is_math_task(user_input):
        return calculator_tool(user_input)
    else:
        return context_aware_agent(user_input)

# Example usage
if __name__ == "__main__":
    # 3. Example inputs to test the agent with context awareness and math handling 
    inputs = [
        "Hello! Can you tell me about health and insulin sensitivity?",
        "What is the capital of France?",
        "Also, what programming language is popular for AI?",
        "Thanks! And what is 10 plus 5?"
    ]

    # 4. Run the agent with example inputs and print responses
    for user_input in inputs:
        print("\n--- User ---")
        print(user_input)
        answer = agent(user_input)
        print("\n--- Assistant ---")
        print(answer)
