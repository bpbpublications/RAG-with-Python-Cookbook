# batch_embedding_large_corpus.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Define a custom embedding class for batching
class BatchedSentenceTransformerEmbedding(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def embed_documents(self, texts):
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding Batches"):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        return embeddings

    def embed_query(self, text):
        return self.model.encode([text])[0]

# 2. Prepare your large corpus of text documents
# This is a simulated large dataset for demonstration purposes.
corpus = ["LangChain enables LLM applications.", "FAISS performs fast vector searches."] * 10  # Simulated large dataset
documents = [Document(page_content=doc) for doc in corpus]

# 3. Split Documents into Chunks
# This step is crucial for large documents to ensure efficient embedding and indexing.
splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=30)
chunked_docs = []
for doc in documents:
    chunks = splitter.split_text(doc.page_content)
    chunked_docs.extend([Document(page_content=chunk) for chunk in chunks])

# 4. Create the FAISS vector store with batched embeddings
# This will index the documents using the custom embedding model.
embedding_model = BatchedSentenceTransformerEmbedding()
vectorstore = FAISS.from_documents(chunked_docs, embedding=embedding_model)

# 5. Save the vector store to disk
# This allows for later retrieval without needing to re-embed the documents.
vectorstore.save_local("faiss_large_content_index")

# 6. Load the vector store and perform a similarity search
# This demonstrates how to retrieve documents similar to a query.
vectorstore = FAISS.load_local("faiss_large_content_index", embedding_model,allow_dangerous_deserialization=True)
results = vectorstore.similarity_search("What does FAISS do?", k=2)

# print the search results
for r in results:
     print(r.page_content)
