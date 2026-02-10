# This code demonstrates how to split a text document into smaller chunks using LangChain's NLTKTextSplitter.
# It uses the NLTK library for tokenization and is suitable for processing large text files
import nltk
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.document_loaders import TextLoader

# Ensure you have the NLTK tokenizer downloaded
nltk.download("punkt")

# Step 1: Load the document
# Replace 'RAG.txt' with your actual text file path.
# Ensure the file exists in the specified path.
loader = TextLoader("RAG.txt")  # Replace with your actual text file
documents = loader.load()

# Step 2: Initialize the NLTKTextSplitter
# This will split the text into chunks based on specified parameters.
text_splitter = NLTKTextSplitter(
    chunk_size=300,        # Max characters per chunk
    chunk_overlap=50       # Overlap between chunks
)

# Step 3: Split the documents into chunks
# This will create smaller text segments for better processing and retrieval.
split_docs = text_splitter.split_documents(documents)

# Step 4: Output the results
# This will print each chunk to the console.
for i, chunk in enumerate(split_docs):
    print(f"\n=== Chunk {i + 1} ===")
    print(chunk.page_content)
