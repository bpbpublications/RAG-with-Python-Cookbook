# This script demonstrates how to split a long text into smaller chunks # using TokenTextSplitter from Langchain
from langchain_text_splitters import TokenTextSplitter

# Example text to be split
text = """
Retrieval Augmented Generation (RAG) is an architecture that combines the ability of large language models (LLMs) with a retrieval system to enhance the factual accuracy, contextual relevance, and quality of generated response against the query raised by user to a RAG system.
"""

# 1. Initialize the TokenTextSplitter
# This will split the text into chunks based on token count
splitter = TokenTextSplitter(
    chunk_size=30,        # max tokens per chunk
    chunk_overlap=10      # token overlap between chunks
)

# 2. Split the text into chunks
# The split_text method will return a list of text chunks   
chunks = splitter.split_text(text)

# 3. Display the number of chunks created and preview the chunks
# This will print the total number of chunks and the content of each
# chunk
print(f"Total Chunks: {len(chunks)}\n")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print()
