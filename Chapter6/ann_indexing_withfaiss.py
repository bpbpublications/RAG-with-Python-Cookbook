# ann_indexing_withfaiss.py
import numpy as np
import faiss                   # pip install faiss-cpu

# 1. Generate random vectors for database and queries
# This simulates a scenario where we have a database of vectors and we want to query them
d = 64                         # dimension of vectors
nb = 1000                      # database size
nq = 5                         # number of queries

np.random.seed(42)
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# 2. Create a FAISS index
# Using a flat index for simplicity, which is suitable for small datasets
index = faiss.IndexFlatL2(d)   
print("Is trained:", index.is_trained)

# 3. Add vectors to the index
# This step indexes the vectors in the database
index.add(xb)                  # type: ignore
print("Total vectors indexed:", index.ntotal)

# 4. Perform a search
k = 5                         # number of nearest neighbors
distances, indices = index.search(xq, k)   # type: ignore

# 5. Print the results
# This will show the indices of the nearest neighbors and their distances for each query vector
print("Query Results (indices):")
print(indices)
print("Query Results (distances):")
print(distances)
