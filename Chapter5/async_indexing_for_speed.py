# async_indexing_for_speed.py
import os
import glob
import asyncio
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Define constants for the script
# These constants include directories for input files, the vector store, and the embedding model
INPUT_DIR = "documents"            # Folder containing text files
PERSIST_DIR = "chapter5_chroma_store"  # Chroma DB folder
EMBED_MODEL = "all-MiniLM-L6-v2"  # Fast embedding model
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 2. Function to chunk text into smaller pieces
# This function takes a string and splits it into chunks of a specified size with some overlap
def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Split text into chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

# 3. Asynchronous file reading function
# This function reads a file asynchronously using an event loop and returns its content as a string
async def read_file(path: str) -> str:
    """Read file asynchronously."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: open(path, "r", encoding="utf-8").read())

# 4. Asynchronous processing of files
# This function reads a file and chunks its content
async def process_file(path: str) -> List[str]:
    """Read and chunk file."""
    text = await read_file(path)
    return chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

# 5. Main function to gather all files and process them asynchronously
# It collects all text files, processes them, and stores the resulting chunks in a vector store
async def main():
    files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    if not files:
        print("No supported files found.")
        return

# 6. Print the number of files found and start processing them asynchronously
    print(f"Found {len(files)} files. Processing asynchronously...")
    results = await asyncio.gather(*(process_file(f) for f in files))
    
# 7. Flatten the list of lists into a single list of chunks
# This combines all the chunks from each file into one list
    all_chunks = [chunk for chunks in results for chunk in chunks]

# 8. Print the total number of chunks created
    print(f"Total chunks: {len(all_chunks)}")
    
# 9. Create a vector store using Chroma and the HuggingFace embeddings
# This initializes the vector store with the processed text chunks
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    db.add_texts(all_chunks)

# 10. Print a success message indicating the vector store has been created
    print("Vector store created successfully.")

# Entry point for the script
# This ensures the main function runs when the script is executed directly
if __name__ == "__main__":
    asyncio.run(main())
