# embedding_chunk_graphs.py
# This script demonstrates how to create a graph of text chunks based on their embeddings.
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import networkx as nx

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Load text document
text = """
RAG (Retrieval-Augmented Generation) is a technique to improve LLM responses
by retrieving relevant context from a knowledge base.
It involves steps like loading documents, splitting them into chunks, embedding,
and storing in a vector database.

Chunk embeddings help in semantic search.
You can also build a graph of chunks based on similarity,
which can be used to find related topics and context paths.
"""

# 2. Split text into chunks
# Using RecursiveCharacterTextSplitter to create manageable text chunks.
splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=20)
chunks = splitter.split_text(text)

# 3. Load a pre-trained SentenceTransformer model
# This model is used to generate embeddings for the text chunks.
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# 4. Create a graph of text chunks based on cosine similarity
# Using NetworkX to create a graph where nodes are chunks and edges represent similarity.
def cosine_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

threshold = 0.7  # similarity cutoff
G = nx.Graph()

for i, chunk in enumerate(chunks):
    G.add_node(i, text=chunk)

for i in range(len(chunks)):
    for j in range(i+1, len(chunks)):
        sim = cosine_sim(embeddings[i], embeddings[j])
        if sim >= threshold:
            G.add_edge(i, j, weight=sim)

# 5. Print the graph edges
# Displaying edges with similarity above the threshold.
print("Graph edges based on similarity >= 0.7:")
for u, v, data in G.edges(data=True):
    print(f"Chunk {u} ↔ Chunk {v} | similarity: {data['weight']:.2f}")

# 6. Print the chunks
# Displaying the text of each chunk for reference.
print("\nChunks:")
for idx, c in enumerate(chunks):
    print(f"{idx}: {c}")
