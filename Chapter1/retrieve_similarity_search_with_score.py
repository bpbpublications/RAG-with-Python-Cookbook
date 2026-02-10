# This code demonstrates how to perform a similarity search with scores using LangChain and FAISS.
# It loads a text document, splits it into chunks, creates embeddings, and performs a similarity
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load the document
# Replace 'RAG.txt' with your actual text file path.
# Ensure the file exists in the specified path.
loader = TextLoader("RAG.txt")  # Replace with your file path
documents = loader.load()

# 2. Split the documents into smaller chunks
# This is important for efficient processing and retrieval.
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(documents)

# 3. Create embeddings for the document chunks
# Using HuggingFace embeddings for semantic understanding.
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # Lightweight, good-quality model
)

# 4. Create a FAISS vector store from the document chunks and embeddings
# This allows for efficient similarity search.
faiss_index = FAISS.from_documents(docs, embedding_model)

# 5. Perform a similarity search with scores
# This will find the most relevant chunks based on a query and return their similarity scores.
query = "What is RAG?"
results_with_score = faiss_index.similarity_search_with_score(query, k=3)

# 6. Output the results with scores
# This will print the top matching chunks and their similarity scores to the console.
print("\nTop Matches with Similarity Scores:")
for i, (doc, score) in enumerate(results_with_score, 1):
    print(f"\nResult {i}:")
    print(f"Score: {score:.4f}")
    print("Content:")
    print(doc.page_content)
