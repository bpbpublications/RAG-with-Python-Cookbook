# reranking_with_crossencoders.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from sentence_transformers import CrossEncoder

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

# 5. Perform a similarity search
# This will retrieve the top k chunks similar to a given query
query = "What are the applications of machine learning?"
initial_results = vectorstore.similarity_search(query, k=5)

# 6. Print initial results
# These are the results before re-ranking        
print("\nInitial FAISS Results:")
for r in initial_results:
    print("-", r.page_content[:100], "...")

# 7. Re-Rank using Cross-Encoder
# Initialize the Cross-Encoder model
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# 8. Prepare pairs for Cross-Encoder
# Create pairs of (query, document) for re-ranking
pairs = [(query, r.page_content) for r in initial_results]

# 9. Get scores from Cross-Encoder
# This will score each document based on its relevance to the query
scores = cross_encoder.predict(pairs)

# 10. Combine results with scores
# Zip the initial results with their corresponding scores
scored_results = list(zip(initial_results, scores))

# 11. Re-Rank the results based on scores
# Sort the results by score in descending order
reranked = sorted(scored_results, key=lambda x: x[1], reverse=True)

# 12. Print the re-ranked results
# Display the top results after re-ranking
print("\nRe-Ranked Results (Cross-Encoder):")
for doc, score in reranked:
    print(f"Score: {score:.4f} | {doc.page_content[:100]}...")
