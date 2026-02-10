# 1. Mention sample multi-paragraph text
text = """
Retrieval-Augmented Generation (RAG) is a powerful technique that combines information retrieval with text generation.

It allows a language model to access external knowledge sources, improving its ability to answer questions accurately.

This is especially useful when the model has not been fine-tuned on a specific dataset or domain.

The retriever component fetches the most relevant documents based on a user query.

The generator then produces a response using both the query and the retrieved documents.

This creates a dynamic pipeline where answers can reflect the most up-to-date information available.
"""

# 2. Split the text into paragraphs
paragraphs = [p.strip() for p in text.strip().split('\n\n') if p.strip()]

# 3. Create fixed-size chunks from the paragraphs
# Here, we define a chunk size of 2 paragraphs per chunk
chunk_size = 2
chunks = [
    "\n\n".join(paragraphs[i:i+chunk_size])
    for i in range(0, len(paragraphs), chunk_size)
]

# 4. Print the resulting chunks
# Each chunk will contain 2 paragraphs, or fewer if at the end of the list
for i, chunk in enumerate(chunks, 1):
    print(f"\n--- Chunk {i} ---\n{chunk}")
