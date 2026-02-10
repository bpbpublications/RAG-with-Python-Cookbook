# regex_based_splitting.py
# This script demonstrates how to split text into chunks based on regex patterns.
import re

# 1. Sample text to be split into sections based on headers
# This text contains multiple sections that we want to identify and split
text = """
## What is RAG?
RAG stands for Retrieval-Augmented Generation. It enhances language models by retrieving relevant information from external sources before generating responses.

## Components of RAG
- Retriever: Finds relevant documents.
- Generator: Uses the retrieved context to generate answers.
- Vector Store: Stores document embeddings for efficient search.

## Benefits
- Improved accuracy
- Current information access
- Cost-effective context handling
"""

# 2. Define a regex pattern to match headers
# This pattern matches lines that start with '## ' followed by the header text
pattern = r"^##\s+(.*)$"
sections = re.split(pattern, text, flags=re.MULTILINE)

# 3. Create chunks by pairing headers with their content
# This will create a list of chunks where each chunk contains a header and its corresponding content
chunks = []
for i in range(1, len(sections), 2):
    header = sections[i].strip()
    content = sections[i + 1].strip()
    chunks.append(f"{header}\n{content}")

# 4. Print the resulting chunks
# Each chunk is printed with a header indicating its index
for i, chunk in enumerate(chunks, 1):
    print(f"\n--- Chunk {i} ---\n{chunk}")
