# markdown_header_splitting.py
# This script demonstrates how to split markdown text into chunks based on headers using regex.
# It is useful for processing structured text files.
import re

# 1. This is a sample markdown text with headers that we want to split into chunks.
markdown_text = """
# RAG with Python
This explains use of RAG

## What is RAG?
RAG (Retrieval-Augmented Generation) enhances LLMs by injecting external information into prompts.

## Components
Components of RAG

### Retriever
This fetches relevant documents from a knowledge base.

### Generator
The LLM uses retrieved docs to answer the query.

## Use Cases
- Customer support bots
- Legal document assistants
- Research assistants
"""

# 2. Use regex to split the markdown text into chunks based on headers.
# The pattern matches headers starting with '##' or '###'.
header_pattern = r'(?=^#{2,3} .*)'
chunks = re.split(header_pattern, markdown_text, flags=re.MULTILINE)

# 3. Post-process the chunks to clean up whitespace and ensure they are not empty.
for i, chunk in enumerate(chunks, 1):
    cleaned = chunk.strip()
    # 4. Only print non-empty chunks
    if cleaned:
        print(f"\n--- Chunk {i} ---\n{cleaned}")
