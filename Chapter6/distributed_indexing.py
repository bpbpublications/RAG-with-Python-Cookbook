# distributed_indexing.py
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Initialize the embedding model
# Using HuggingFaceEmbeddings to convert text into vector embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Split documents into chunks
# Using RecursiveCharacterTextSplitter to handle different chunk sizes
data_files = ["chapter6_data_part1.txt", "chapter6_data_part2.txt", "chapter6_data_part3.txt"]

vectorstores = []

# 3. Create a FAISS vector store for each document
# Each worker/node will build its own FAISS index
for file in data_files:
    loader = TextLoader(file) # Load each document
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) 
    chunks = splitter.split_documents(docs) # Split the documents into chunks

    # Create a FAISS vector store with the chunks
    # This will index the chunks for similarity search
    vs = FAISS.from_documents(chunks, embedding_model)
    vectorstores.append(vs)

# 4. Merge the vector stores into a distributed index
# This simulates a distributed indexing scenario
main_index = vectorstores[0]
for vs in vectorstores[1:]:
    main_index.merge_from(vs)

# 5. Perform a similarity search on the distributed index
# This will retrieve the top k chunks similar to a given query
query = "Explain the applications of Artificial Intelligence."
results = main_index.similarity_search(query, k=3)

# 6. Print the results from the distributed index
# Display the top results from the distributed index
print("\nTop Results from Distributed Index:")
for r in results:
    print("-", r.page_content[:100], "...")
