# hybrid_retrieval_agent.py
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
import re
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Create a sample document store 
docs = [
    Document(page_content="RAG stands for Retrieval-Augmented Generation.", metadata={"topic": "RAG"}),
    Document(page_content="Dense embeddings capture semantic meaning of text.", metadata={"topic": "RAG"}),
    Document(page_content="BM25 is a sparse retrieval method based on keyword matching.", metadata={"topic": "RAG"}),
    Document(page_content="Hybrid retrieval combines dense and sparse methods for better results.", metadata={"topic": "RAG"}),
]

# 2. Create vector store from documents using HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

def dense_retrieval(query, k=2):
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

# Sparse retrieval (BM25 / TF-IDF)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([doc.page_content for doc in docs])

def sparse_retrieval(query, k=2):
    query_vec = tfidf_vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_indices = scores.argsort()[::-1][:k]
    return [docs[i].page_content for i in top_indices]

# Hybrid retrieval (dense + sparse)
def hybrid_retrieval(query, k=2):
    dense_results = dense_retrieval(query, k=5)
    sparse_results = sparse_retrieval(query, k=5)
    # Combine results and remove duplicates
    combined = list(dict.fromkeys(dense_results + sparse_results))
    return combined[:k]


# Agent
def agent(query):
    # Simple math detection
    if re.search(r"\d", query) or any(op in query for op in ["plus","minus","multiply","divided"]):
        try:
            return str(eval(query.replace("plus","+").replace("minus","-").replace("multiply","*").replace("divided","/")))
        except:
            return "Error in calculation"
    # Otherwise do hybrid retrieval
    results = hybrid_retrieval(query)
    return "Relevant info:\n" + "\n".join(results)

# Example
if __name__ == "__main__":
# 3. Prepare a query for the program
    queries = [
        "What is RAG?",
        "Explain hybrid retrieval",
        "10 plus 5"
    ]

# 4. Generate a response using hybrid retrieval which uses both dense and sparse techniques
    for q in queries:
        print("\n--- User ---")
        print(q)
        print("\n--- Assistant ---")
        print(agent(q))
