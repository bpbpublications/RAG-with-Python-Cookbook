# semantic_search_auto_summarized_chunks.py
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Choose your file (PDF or TXT)
file_path = "RAG.pdf"  # Change to your file

# 2. Load the document
# Depending on the file type, we use different loaders.
if file_path.lower().endswith(".pdf"):
    loader = PyPDFLoader(file_path)
elif file_path.lower().endswith(".txt"):
    loader = TextLoader(file_path)
else:
    raise ValueError("Unsupported file type. Use PDF or TXT.")

documents = loader.load()
print(f"Loaded {len(documents)} pages from {file_path}")

# 3. Split the document into chunks
# This will create smaller chunks of text for better processing and storage.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")

# 4. Initialize the summarization pipeline
# This will summarize each chunk of text.
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# 5. Create a list to hold summarized documents
# Each summarized chunk will be stored as a Document object.
summarized_docs = []

# 6. Summarize each chunk and create Document objects
for chunk in chunks:
    try:
        summary_text = summarizer(chunk.page_content, max_length=40, min_length=10, do_sample=False)[0]['summary_text']
    except Exception:
        summary_text = "Summary not available"
    summarized_docs.append(
        Document(
            page_content=chunk.page_content,
            metadata={
                "summary": summary_text,
                "source": chunk.metadata.get("source", "unknown")
            }
        )
    )
print("Summaries added to chunks.")

# 7. Load SentenceTransformer embeddings
# This model is used to generate embeddings for the summarized chunks.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 8. Initialize ChromaDB as a vector store
# This will create a local ChromaDB instance to store and retrieve embeddings.
persist_directory = "./chroma_db_summarized"
vectorstore = Chroma.from_documents(
    documents=summarized_docs,
    embedding=embeddings,
    persist_directory=persist_directory
)

# 9. Add summarized documents to the vector store
# This will store the embeddings in ChromaDB.
print("Stored summarized chunks in ChromaDB.")

# 10. Query the vector store
query = "Explain RAG"
results = vectorstore.similarity_search(query, k=3)

# 11. Display search results
print("\nSearch Results:")
for i, doc in enumerate(results, start=1):
    print(f"{i}. SUMMARY: {doc.metadata.get('summary')}")
    print(f"   CONTENT: {doc.page_content.strip()[:200]}...\n")
