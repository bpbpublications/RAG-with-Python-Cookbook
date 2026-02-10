# multivector_retrieval_dense_sparse.py

from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel
from operator import itemgetter

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# -----------------------------
# 1. Define documents
# -----------------------------
docs = [
    "Artificial intelligence is the simulation of human intelligence by machines.",
    "Deep learning is a subset of machine learning using neural networks.",
    "Machine learning enables computers to learn from data without explicit programming.",
    "Reinforcement learning is based on rewards and punishments.",
]

# -----------------------------
# 2. Dense Retriever (Chroma + HF Embeddings)
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

dense_store = Chroma.from_texts(
    docs,
    embedding=embeddings,
    persist_directory="./chroma_multi"
)

dense_retriever = dense_store.as_retriever(search_kwargs={"k": 2})

# -----------------------------
# 3. Sparse Retriever (BM25)
# -----------------------------
sparse_retriever = BM25Retriever.from_texts(docs)
sparse_retriever.k = 2

# -----------------------------
# 4. Ensemble Retriever using LCEL
# -----------------------------
parallel_retriever = RunnableParallel(
    dense=itemgetter("query") | dense_retriever,
    sparse=itemgetter("query") | sparse_retriever
)

def combine_results(results, weights=(0.5, 0.5)):
    dense_docs = results["dense"]
    sparse_docs = results["sparse"]

    # Add weighted scores in metadata
    for d in dense_docs:
        d.metadata["score"] = weights[0]

    for s in sparse_docs:
        s.metadata["score"] = weights[1]

    merged = dense_docs + sparse_docs

    # Deduplicate using document content
    unique = {}
    for doc in merged:
        key = doc.page_content.strip()
        if key not in unique:
            unique[key] = doc
        else:
            # Keep document with higher score
            if doc.metadata["score"] > unique[key].metadata["score"]:
                unique[key] = doc

    # Convert back to list and sort by score
    return sorted(unique.values(), key=lambda x: x.metadata["score"], reverse=True)



# -----------------------------
# 5. Query
# -----------------------------
query = "What are neural networks?"

results = parallel_retriever.invoke({"query": query})
final_docs = combine_results(results)

# -----------------------------
# 6. Print Results
# -----------------------------
print(f"Query: {query}\n")
for i, doc in enumerate(final_docs, start=1):
    print(f"Result {i}: {doc.page_content}")
