# custom_embedding_function_using_sentencetransformers.py
# This code demonstrates how to create a custom embedding function using the Sentence Transformers library.
from langchain.embeddings.base import Embeddings
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Custom embedding class that uses Sentence Transformers
# This class implements the LangChain Embeddings interface.
class CustomSentenceTransformerEmbedding(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()


# 2. Define a list of documents to be embedded
# Each document is a string that will be transformed into a vector representation.
docs = [
    Document(page_content="LangChain is powerful for building RAG."),
    Document(page_content="Embeddings turn text into vectors."),
]

# 3. Create an instance of the custom embedding function
# This initializes the SentenceTransformer model for generating embeddings.
embedding_fn = CustomSentenceTransformerEmbedding()

# 4. Create a FAISS vector store using the custom embedding function
# The vector store will use the embeddings to allow for efficient similarity search.
vector_store = FAISS.from_documents(docs, embedding=embedding_fn)

# 5. Example query to search the vector store
# This query will find documents similar to the input text based on their embeddings.
query = "What is LangChain used for?"
results = vector_store.similarity_search(query, k=1)

# 6. Print the results
# The results show the documents that are most similar to the query.
for res in results:
    print("Query:", query)
    print("Match:", res.page_content)
