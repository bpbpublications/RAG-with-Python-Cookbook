# query_faiss_vectorstore.py
# This code snippet demonstrates how to query a FAISS vector store using LangChain and Hugging Face embeddings.
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Load the FAISS vector store
# Ensure the FAISS index is already created and saved in the specified directory.
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(folder_path="chapter5_faiss_store", embeddings=embedding_model, allow_dangerous_deserialization=True)

# 2. Define your query
# This is the question you want to answer using the retrieved context.
query = "What is LangChain?"
results = vectorstore.similarity_search(query, k=2)

# 3. Output the results
# This will print the content of the retrieved documents.
print("\nTop Matches:")
for i, doc in enumerate(results, start=1):
    print(f"{i}. {doc.page_content}")
