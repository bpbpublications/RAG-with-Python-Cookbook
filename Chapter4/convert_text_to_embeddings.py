# convert_text_to_embeddings.py
# This code demonstrates how to convert text into embeddings using the HuggingFaceEmbeddings class from LangChain.
# It initializes the embedding model, provides a sample text, and generates the corresponding embedding vector.
from langchain_huggingface import HuggingFaceEmbeddings


import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Initialize the embedding model
# The model 'all-MiniLM-L6-v2' is a lightweight model suitable for generating embeddings.
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Define the text to be converted into embeddings
# This text will be transformed into a vector representation.
text = "Retrieval-Augmented Generation is a powerful technique for combining retrieval with language models."

# 3. Generate the embedding vector for the text
# The embed_query method converts the text into a numerical vector.
embedding = embedding_model.embed_query(text)

# 4. Print the resulting embedding vector
# The embedding is a list of floating-point numbers representing the text in a high-dimensional space.
print(f"Embedding dimension: {len(embedding)}")
print(f"First 10 values of the embedding:\n{embedding[:10]}")
