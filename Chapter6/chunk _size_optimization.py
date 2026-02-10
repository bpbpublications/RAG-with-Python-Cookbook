# chunk_size_optimization.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Load your text document
# Ensure you have a text file named "RAG.txt" in the same directory
loader = TextLoader("RAG.txt")
documents = loader.load()

# 2. Initialize the embedding model
# Using a HuggingFace model for embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Test different chunk sizes
# This will help us find the optimal chunk size for our use case
chunk_sizes = [200, 500, 1000]

# 4. Loop through different chunk sizes to see their impact on retrieval
for size in chunk_sizes:
    print(f"\n🔹 Testing with chunk size = {size}")
    
    # 4a. Split the documents into chunks
    # Using RecursiveCharacterTextSplitter to handle different chunk sizes
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=50  # keep some overlap for context
    )
    chunks = splitter.split_documents(documents)
    print(f"Total Chunks Created: {len(chunks)}")
    
    # 4b. Create a FAISS vector store with the chunks
    # This will index the chunks for similarity search
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    
    # 4c. Perform a similarity search
    # This will retrieve the top k chunks similar to a given query
    query = "What is RAG?"
    results = vectorstore.similarity_search(query, k=3)
    
    print("Top Results:")

    # 5. Print the top results for the query
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.page_content[:150]}...")
