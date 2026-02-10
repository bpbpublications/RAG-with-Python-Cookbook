# noiseresistantembeddings_with_prenormalization.py
# This script demonstrates how to create noise-resistant embeddings using pre-normalization.
from sentence_transformers import SentenceTransformer
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Load a pre-trained SentenceTransformer model
# This model is used to generate embeddings for sentences.
# The model is chosen for its balance between performance and speed.
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. Define a set of sentences with noise (punctuation, extra spaces, etc.)
# These sentences will be used to generate embeddings.
# The noise includes punctuation variations, extra spaces, and capitalization differences.
sentences = [
    "Retrieval Augmented Generation (RAG) is an architecture!",
    "Retrieval Augmented Generation (RAG) is an architecture!!!   ",  # extra punctuation & spaces
    "RETRIEVAL AUGMENTED GENERATION (RAG) IS AN ARCHITECTURE",  # all caps
]

# 3. Define a normalization function
# This function normalizes the embeddings to unit length.
def normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms

# 4. Generate embeddings for the sentences
# The model encodes the sentences into dense vector representations.
embeddings = model.encode(sentences)

# 5. Normalize the embeddings
# This scales each embedding vector to unit length (L2 norm = 1),
# ensuring consistent magnitude and stable similarity comparisons.
normalized_embeddings = normalize(embeddings)

# 6. Print the normalized embeddings
# Displaying the first 5 dimensions of each normalized embedding for brevity
for i, vec in enumerate(normalized_embeddings):
    print(f"Sentence {i+1} (norm={np.linalg.norm(vec):.2f}):")
    print(vec[:5], "...")  # first 5 dims
    print()
