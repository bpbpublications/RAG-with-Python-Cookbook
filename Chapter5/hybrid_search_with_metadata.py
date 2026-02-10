# hybrid_search_with_metadata.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.documents import Document as LangchainDocument

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 2. Prepare your text documents
# These documents will be indexed in the FAISS vector store.
# Each document can have metadata for filtering during search.
docs = [
    LangchainDocument(page_content="LangChain is a tool to build LLM applications.", metadata={"source": "intro", "topic": "LLM"}),
    LangchainDocument(page_content="FAISS is used for vector similarity search.", metadata={"source": "db", "topic": "Vector DB"}),
    LangchainDocument(page_content="OpenAI provides GPT models.", metadata={"source": "api", "topic": "LLM"}),
    LangchainDocument(page_content="FAISS and Chroma are vector databases.", metadata={"source": "db", "topic": "Vector DB"}),
]

# 2. Load the embedding model
# You can choose a different model if needed, but this one is efficient for many tasks.
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Create the FAISS vector store
# This will index the documents using the specified embedding model.
vectorstore = FAISS.from_documents(docs, embedding=embedding_model)

# 5. Perform a hybrid search with metadata filtering
# This will find the most relevant documents for a given query, considering both semantic similarity and metadata
query = "Which tools help with semantic search?"
results = vectorstore.similarity_search(
    query, 
    k=2, 
    filter={"topic": "Vector DB"}  # Hybrid filtering
)

# 6. Output the results
# This will print the content of the retrieved documents along with their metadata.
print("\n Hybrid Search Results (Semantic + Metadata):")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content} | Metadata: {doc.metadata}")
