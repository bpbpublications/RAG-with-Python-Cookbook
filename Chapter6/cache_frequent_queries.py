# cache_frequent_queries.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Load your text document
# Ensure you have a text file named "chapter6_sample_doc.txt" in the same directory
loader = TextLoader("chapter6_sample_doc.txt")
documents = loader.load()

# 2. Initialize the embedding model
# Using a HuggingFace model for embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Split the documents into chunks
# Using RecursiveCharacterTextSplitter to handle different chunk sizes
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# 4. Create a FAISS vector store with the chunks
# This will index the chunks for similarity search
vectorstore = FAISS.from_documents(chunks, embedding_model)

# 5. Implement caching for frequent queries
# Using a simple dictionary to cache results
query_cache = {}   # dictionary to store {query: results}

# 6. Function to perform search with caching
def search_with_cache(query, k=3):
    # 6a. Check if the query is in cache
    if query in query_cache:
        print(f"Cache Hit: Returning cached results for '{query}'")
        return query_cache[query]
    
    # 6b. If not in cache, perform the search
    # This will retrieve the top k chunks similar to the given query
    print(f"Cache Miss: Running similarity search for '{query}'")
    results = vectorstore.similarity_search(query, k=k)
    
    # 6c. Store the results in the cache
    query_cache[query] = results
    return results

# 7. Test the caching mechanism with some queries
# You can replace these queries with any relevant to your document
queries = [
    "What is machine learning?",
    "What is AI?",
    "What is machine learning?",   # repeated
    "Applications of machine learning",
    "What is AI?"                  # repeated
]

# 8. Loop through the queries and print the results
for q in queries:
    results = search_with_cache(q)
    print(f"Top Result: {results[0].page_content[:100]}...\n")
