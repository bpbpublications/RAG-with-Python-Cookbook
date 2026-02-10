# This code demonstrates how to batch small documents before splitting them into chunks using LangChain's text splitter.
# It combines small documents into batches and then splits them into manageable chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. Sample small documents
docs = [
    Document(page_content="What is RAG?", metadata={"source": "faq1"}),
    Document(page_content="RAG combines retrieval and generation.", metadata={"source": "faq2"}),
    Document(page_content="LangChain is useful for RAG.", metadata={"source": "faq3"}),
    Document(page_content="Use FAISS for similarity search.", metadata={"source": "faq4"}),
    Document(page_content="Embeddings map text into vectors.", metadata={"source": "faq5"}),
]

# 2. Batch small documents before splitting
# Here, we will batch documents into groups of 2
batch_size = 2
batched_docs = []
for i in range(0, len(docs), batch_size):
    combined_text = "\n".join([doc.page_content for doc in docs[i:i+batch_size]])
    metadata = {"batched_from": [doc.metadata["source"] for doc in docs[i:i+batch_size]]}
    batched_docs.append(Document(page_content=combined_text, metadata=metadata))

# 3. Split the batched documents into chunks
# Using RecursiveCharacterTextSplitter to split the batched documents into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
split_docs = splitter.split_documents(batched_docs)

# 4. Print the resulting chunks with their metadata
# Each chunk will retain the metadata from the original documents
for i, chunk in enumerate(split_docs):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Metadata: {chunk.metadata}")
    print(f"Content:\n{chunk.page_content}")
