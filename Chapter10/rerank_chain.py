# rerank_chain.py
# Example of a rerank chain using FAISS and CrossEncoder for reranking
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Load or create FAISS index with local embeddings model 
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. If FAISS index already exists, load it
# Otherwise, build from a small sample corpus
try:
    vectorstore = FAISS.load_local("faiss_index", embedding_model)
except:
    texts = [
        "Intermittent fasting improves insulin sensitivity.",
        "It may help with weight loss by reducing calorie intake.",
        "Fasting can reduce inflammation and support cellular repair.",
        "Drinking water during fasting helps with hydration.",
        "Exercise combined with intermittent fasting can boost fat loss."
    ]
    vectorstore = FAISS.from_texts(texts, embedding_model)
    vectorstore.save_local("faiss_index")

# 3. Create retriever from FAISS index   
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 4. Initialize CrossEncoder for reranking 
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Rerank function using CrossEncoder
def rerank_results(query, docs):
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    for i, doc in enumerate(docs):
        doc.metadata["score"] = float(scores[i])
    reranked = sorted(docs, key=lambda d: d.metadata["score"], reverse=True)
    return reranked

# 5. Execute the rerank chain with an example query and print results 
def run_rerank_chain(query):
    docs = retriever.invoke(query)
    reranked_docs = rerank_results(query, docs)

    print("\n=== Top Reranked Documents ===")
    for d in reranked_docs:
        print(f"Score: {d.metadata['score']:.4f} | {d.page_content}")
    return reranked_docs

# Example query
run_rerank_chain("What are the benefits of intermittent fasting?")
