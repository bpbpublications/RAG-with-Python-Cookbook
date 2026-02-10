# batch_embedding_large_documents.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Define the embedding model
# Using SentenceTransformer for batch processing
class BatchedSentenceTransformerEmbedding(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def embed_documents(self, texts):
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding Batches"):
            batch = texts[i: i + self.batch_size]
            batch_embeddings = self.model.encode(batch)
            embeddings.extend(batch_embeddings)
        return embeddings

    def embed_query(self, text):
        return self.model.encode([text])[0]

# 2. Create a large document
# Simulating a large document by repeating a smaller text
large_text = "LangChain is powerful.\n" * 50  # Simulate repetition

# 3. Split the large document into manageable chunks
# RecursiveCharacterTextSplitter ensures that the document is broken into
# overlapping chunks small enough for the embedding model to handle efficiently.
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_text(large_text)
documents = [Document(page_content=chunk) for chunk in chunks]

# 4. Create the vector store with batched embeddings
# Using FAISS for efficient similarity search
embedding_model = BatchedSentenceTransformerEmbedding()
vectorstore = FAISS.from_documents(documents, embedding=embedding_model)

# 5. Perform a similarity search
# Searching for a specific query in the vector store
results = vectorstore.similarity_search("What is LangChain?", k=1)
print("\nBest Match:", results[0].page_content)
