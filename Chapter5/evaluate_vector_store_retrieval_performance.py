# evaluate_vector_store_retrieval_performance.py
import time
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. define constants
# This is a persistent directory created in last recipe
PERSIST_DIR = "chapter5_chroma_store"
EMBED_MODEL = "all-MiniLM-L6-v2"
K = 3

# 2. Define evaluation data
# This is a list of queries and expected answers to evaluate the vector store's retrieval performance
EVAL_DATA = [
    {"query": "What is RAG?", "answer": "RAG"},
    {"query": "What is the purpose of vector stores?", "answer": "vector stores"},
    ]

# 3. Load the vector store
# This initializes the Chroma vector store with the specified directory and embedding model
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

# 4. Evaluate the vector store's retrieval performance
# This iterates through the evaluation data, performs similarity searches, and checks if the expected answer
# is present in the retrieved results
correct = 0
latencies = []

for example in EVAL_DATA:
    query, expected = example["query"], example["answer"]

    start = time.time()
    results = db.similarity_search(query, k=K)
    latency = time.time() - start
    latencies.append(latency)

    retrieved_texts = " ".join([doc.page_content for doc in results])

    if expected.lower() in retrieved_texts.lower():
        correct += 1

    print(f"\nQuery: {query}")
    print(f"Expected answer keyword: {expected}")
    print(f"Top-{K} Retrieved: {[doc.page_content[:50]+'...' for doc in results]}")
    print(f"Correct? {'✅' if expected.lower() in retrieved_texts.lower() else '❌'}")
    print(f"Latency: {latency:.3f}s")

# 5. Print evaluation summary
# This calculates and prints the accuracy of the retrieval performance and the average latency
accuracy = correct / len(EVAL_DATA)
avg_latency = sum(latencies) / len(latencies)

print("\n==== Evaluation Summary ====")
print(f"Accuracy@{K}: {accuracy:.2f}")
print(f"Average Latency: {avg_latency:.3f}s")
