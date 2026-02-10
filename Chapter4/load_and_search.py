# load_and_search.py
# This script demonstrates how to load a saved FAISS index and perform a similarity search.
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Initialize the embedding model
# This uses the same embedding model as in the previous script to ensure compatibility.
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Load the FAISS index from the saved path
# This assumes the index was saved in the previous step.
faiss_index_path = "embed_faiss_index"
vectorstore = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)

# 3. Perform a similarity search
# This query will find documents similar to the input text based on their embeddings.
results = vectorstore.similarity_search("What does Generative models do?", k=1)
print("\nBest Match:", results[0].page_content)
