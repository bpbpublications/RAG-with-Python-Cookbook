# embeddings_offline_using_sentencetransformers.py
# This script demonstrates how to use the SentenceTransformers library to generate text embeddings offline.
from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Load an offline, pre-trained SentenceTransformer model
# This lightweight open-source model ("all-MiniLM-L6-v2") runs locally,
# offering a good balance between speed and accuracy.
model = SentenceTransformer("all-MiniLM-L6-v2") 

# 2. Sample texts to generate embeddings
texts = [
    "LangChain enables RAG pipelines.",
    "Embedding converts text into vectors.",
    "You can run models offline with SentenceTransformers."
]

# 3. Generate embeddings for sample texts
# The model.encode method processes the texts and returns their embeddings
embeddings = model.encode(texts)

# 4. Print the generated embeddings
for i, emb in enumerate(embeddings):
    print(f"\nText {i+1}: {texts[i]}")
    print(f"Embedding shape: {emb.shape}")
    print(f"First 5 dimensions: {emb[:5]}")
