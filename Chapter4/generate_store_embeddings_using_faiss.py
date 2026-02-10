# generate_store_embeddings_using_faiss.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Load the pre-trained model
# The model 'all-MiniLM-L6-v2' is suitable for generating embeddings.
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Define a list of documents to be embedded
# Each document is a string that will be transformed into a vector representation.
documents = [
    "RAG combines retrieval with generative models.",
    "FAISS enables fast similarity search.",
    "Embedding turns text into numeric vectors.",
    "Transformers are powerful for NLP tasks."
]

# 3. Embed the documents
# The model.encode method generates embeddings for the list of documents.
embeddings = model.encode(documents, convert_to_numpy=True)

# 4. Create a FAISS index
# The index will store the embeddings and allow for efficient similarity search.
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# 5. Add embeddings to the FAISS index
# The embeddings are added to the index for later retrieval.
index.add(embeddings) # type: ignore

# 6. Example query to search the FAISS index
query = "How to search a vector embeddings?"

query_vector = model.encode([query])
k = 2  # top-k matches

# 7. Search the index
# The search method retrieves the k nearest neighbors for the query vector.
distances, indices = index.search(query_vector, k) # type: ignore

# 8. Print the results
# The results show the documents that are most similar to the query along with their distances.
print(f"\nQuery: {query}")
for i, idx in enumerate(indices[0]):
    print(f"Match {i+1}: '{documents[idx]}' (distance: {distances[0][i]:.4f})")
