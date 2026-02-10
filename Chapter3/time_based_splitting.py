# time_based_splitting.py
# This script demonstrates how to split documents based on time metadata.
from langchain.schema import Document
from datetime import timedelta

# 1. Sample transcript with time metadata (in seconds)
docs = [
    Document(page_content="Intro to RAG and its use cases.", metadata={"timestamp": 0}),
    Document(page_content="LangChain framework and components.", metadata={"timestamp": 60}),
    Document(page_content="Loading and splitting documents.", metadata={"timestamp": 130}),
    Document(page_content="Creating embeddings using HuggingFace.", metadata={"timestamp": 190}),
    Document(page_content="Indexing and retrieval techniques.", metadata={"timestamp": 250}),
    Document(page_content="Building a full RAG pipeline.", metadata={"timestamp": 310}),
]

# 2. Define the chunk duration in seconds
# Here, we will create chunks of 2 minutes (120 seconds)
chunk_duration = 120
chunks = []
current_chunk = []
current_start_time = 0

# 3. Iterate through the documents and group them into chunks based on the defined duration
for doc in docs:
    if doc.metadata["timestamp"] < current_start_time + chunk_duration:
        current_chunk.append(doc)
    else:
        # Create one chunk
        combined_text = "\n".join([d.page_content for d in current_chunk])
        chunks.append(Document(page_content=combined_text, metadata={"start_time": str(timedelta(seconds=current_start_time))}))
        # Start a new chunk
        current_start_time += chunk_duration
        current_chunk = [doc]

# Add the last chunk if any
if current_chunk:
    combined_text = "\n".join([d.page_content for d in current_chunk])
    chunks.append(Document(page_content=combined_text, metadata={"start_time": str(timedelta(seconds=current_start_time))}))

# 4. Print the resulting chunks with their start time metadata
# Each chunk will retain the start time metadata for context
for i, chunk in enumerate(chunks, 1):
    print(f"\n--- Chunk {i} (Start Time: {chunk.metadata['start_time']}) ---")
    print(chunk.page_content)
