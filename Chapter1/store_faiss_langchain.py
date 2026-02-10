# This code demonstrates how to create a FAISS vector store using LangChain and HuggingFace embeddings.
# It loads a text document, splits it into chunks, creates embeddings, and stores them in
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load the document
# Replace 'RAG.txt' with your actual text file path.
loader = TextLoader("RAG.txt")  # Replace with your file
documents = loader.load()

# 2. Split the documents into smaller chunks
# This is important for efficient processing and retrieval.
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(documents)

# 3. Create embeddings for the document chunks
# Using HuggingFace embeddings for semantic understanding.
embeddings = HuggingFaceEmbeddings()

# 4. Create a FAISS vector store from the document chunks and embeddings
# This allows for efficient similarity search.
faiss_index = FAISS.from_documents(docs, embeddings)

# 5. Perform a similarity search
# This will find the most relevant chunks based on a query.
query = "What is RAG?"
results = faiss_index.similarity_search(query, k=3)

# 6. Output the results
# This will print the top matching chunks to the console.
print("\n Top Matches:")
for i, doc in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(doc.page_content)


