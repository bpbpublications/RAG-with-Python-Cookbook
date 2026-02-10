# ngram_splitter_usage_example.py
# It demonstrates how to use the NGramTextSplitter to split text into n-grams.
from ngram_splitter import NGramTextSplitter

# 1. Sample text to be split into n-grams
sample_text = (
    """Retrieval Augmented Generation (RAG) is an architecture that combines
the ability of large language models (LLMs) with a retrieval system to enhance
the factual accuracy, contextual relevance, and quality of generated response
against the query raised by user to a RAG system."""
)

# 2. Initialize the n-gram splitter with desired parameters
# Here, n=8 means each chunk will have 8 words, and overlap=3
splitter = NGramTextSplitter(n=8, overlap=3)

# 3. Split the sample text into n-grams
# The create_documents method will return a list of Document objects
# Each Document will contain a chunk of text as specified by the n-gram parameters
docs = splitter.create_documents([sample_text])

# 4. Print the resulting n-gram chunks
# Each chunk will be printed with its content
for i, doc in enumerate(docs, 1):
    print(f"\n--- Chunk {i} ---\n{doc.page_content}")
