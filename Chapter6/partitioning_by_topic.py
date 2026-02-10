# partitioning_by_topic.py
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.cluster import KMeans
import faiss
import numpy as np

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
chunk_texts = [c.page_content for c in chunks]
chunk_embeddings = embeddings.embed_documents(chunk_texts)

print(f"Created {len(chunks)} chunks with {len(chunk_embeddings[0])}-dim embeddings")

# 4. Cluster embeddings using KMeans
# Choose number of topics based on your data; this can be tuned
num_topics = 3   # tune this based on data size
kmeans = KMeans(n_clusters=num_topics, random_state=42, n_init=10)
labels = kmeans.fit_predict(chunk_embeddings)

# 5. Partition chunks by topic
# Create a dictionary to hold chunks for each topic
partitions = {i: [] for i in range(num_topics)}
for idx, label in enumerate(labels):
    partitions[label].append(idx)

print("\nPartition sizes:")
for topic, idxs in partitions.items():
    print(f"Topic {topic}: {len(idxs)} chunks")

# 6. Create FAISS indices for each topic partition
# This allows efficient searching within each topic
topic_indices = {}
for topic, idxs in partitions.items():
    topic_embs = np.array([chunk_embeddings[i] for i in idxs]).astype("float32")
    index = faiss.IndexFlatL2(topic_embs.shape[1])
    index.add(topic_embs) # type: ignore
    topic_indices[topic] = (index, idxs)

# 7. Perform a similarity search for a query
# This will retrieve the top k chunks similar to a given query
query = "How is AI used in healthcare?"
query_emb = embeddings.embed_query(query)

# 8. Determine the topic of the query using KMeans
centroids = kmeans.cluster_centers_
topic_id = np.argmin(np.linalg.norm(centroids - query_emb, axis=1))
print(f"\nQuery routed to Topic {topic_id}")

# 9. Search within the topic partition
# Retrieve the FAISS index and corresponding chunk indices for the identified topic
index, idxs = topic_indices[topic_id]
D, I = index.search(np.array([query_emb]).astype("float32"), k=2)

# 10. Print the results
print("\nQuery:", query)
print("Retrieved Chunks:")
for rank, idx in enumerate(I[0]):
    chunk_id = idxs[idx]
    print(f"- {chunk_texts[chunk_id][:150]}...")
