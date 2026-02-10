# embedding_dimensionality_reduction.py
# This script demonstrates how to reduce the dimensionality of embeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from sklearn.decomposition import TruncatedSVD
import numpy as np
import faiss

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Load your text document
# Ensure you have a text file named "chapter6_sample_doc.txt" in the same directory
loader = TextLoader("chapter6_sample_doc.txt")
docs = loader.load()

# 2. Split the documents into chunks
# Using CharacterTextSplitter to handle different chunk sizes
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Initialize the embedding model
# Using a HuggingFace model for embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
original_embeddings = embeddings.embed_documents([c.page_content for c in chunks])
print(f"Original dim: {len(original_embeddings[0])}, Chunks: {len(original_embeddings)}")

# 4. Reduce dimensionality using SVD
# Set target dimension, ensuring it does not exceed the number of samples or features
target_dim = 128
max_possible_dim = min(len(original_embeddings), len(original_embeddings[0]))  # cannot exceed samples/features
safe_dim = min(target_dim, max_possible_dim)

svd = TruncatedSVD(n_components=safe_dim, random_state=42)
reduced_embeddings = svd.fit_transform(original_embeddings)

actual_dim = reduced_embeddings.shape[1]
print(f"Reduced dim: {actual_dim}")

# 5. Create a FAISS index with the reduced embeddings
# FAISS requires float32 type for indexing
index = faiss.IndexFlatL2(actual_dim)  # match the actual reduced dimension
index.add(np.array(reduced_embeddings).astype("float32")) # type: ignore
print(f"FAISS Index contains {index.ntotal} vectors.")

# 6. Perform a similarity search
# This will retrieve the top k chunks similar to a given query
query = "Applications of AI in healthcare"
query_emb = embeddings.embed_query(query)
query_emb_reduced = svd.transform([query_emb])

D, I = index.search(np.array(query_emb_reduced).astype("float32"), k=2) # type: ignore

# 7. Print the results
print("\nQuery:", query)
print("Retrieved Chunks:")
for idx in I[0]:
    print("-", chunks[idx].page_content[:150], "...")
