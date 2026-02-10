# streaming_retrieval_agent.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Sample docs to showcase streaming retrieval
docs = [
    "RAG improves answers by grounding them in retrieved documents.",
    "FAISS is a library for fast similarity search in vector databases.",
    "Python is widely used in artificial intelligence."
]

# 2. Create a vector store from the documents using HuggingFace embeddings  
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(docs, embedding=embeddings)

# 3. Load small local model
model_id = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=128)

# 4. Prepare a query for streaming
query = "How does RAG work?"
print("\n--- User ---")
print(query)

retriever = vectorstore.as_retriever()
docs = retriever.invoke(query)   # new method (instead of get_relevant_documents)
context = " ".join([d.page_content for d in docs])

# 5. Create a prompt for streaming
prompt = f"Answer concisely using context: {context}. Question: {query}"

# 6. Get the output from streaming agent and print
print("\n--- Assistant (streaming) ---")
output = pipe(prompt, max_new_tokens=50, clean_up_tokenization_spaces=True)[0]["generated_text"]

# simulate streaming
for word in output.split():
    print(word, end=" ", flush=True)

