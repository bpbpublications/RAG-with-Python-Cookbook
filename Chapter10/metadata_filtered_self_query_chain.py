# metadata_filtered_self_query_chain.py
# Example of metadata-filtered retrieval using FAISS and local embeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI  
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Sample Documents with metadata for filtering 
docs = [
    Document(page_content="Alice wrote about intermittent fasting and health.", metadata={"category": "health"}),
    Document(page_content="The 2008 financial crisis impacted global markets.", metadata={"category": "finance"}),
    Document(page_content="The French Revolution began in 1789.", metadata={"category": "history"})
]

# 2. Create local embeddings using SentenceTransformers
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Create FAISS vector store from documents with metadata  
vectorstore = FAISS.from_documents(docs, embedding)

# filter function by metadata
def filter_by_category(category):
    return vectorstore.similarity_search(
        query="",
        filter={"category": category},
        k=3
    )

# 4. Example Queries with expected categories 
queries = [
    ("Tell me about health", "health"),
    ("Financial events after 2005", "finance"),
    ("History events before 1900", "history")
]

# 5. Execute queries and print results with metadata filtering
for q_text, category in queries:
    results = filter_by_category(category)
    print(f"Query: {q_text}")
    for res in results:
        print(f"Answer: {res.page_content} (Source category: {res.metadata['category']})")
        print("\n" + "-"*50 + "\n")
