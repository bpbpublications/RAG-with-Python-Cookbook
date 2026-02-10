# batching_before_splitting.py
# This script demonstrates how to batch small text files into logical groups
# based on token count, and then split those batches into smaller chunks for embedding.
import os
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tiktoken

# 1. Initialize the tokenizer
# You can use any tokenizer compatible with your embedding model
tokenizer = tiktoken.get_encoding("cl100k_base")  # or use tokenizer.encode(text)

# 2. Load text documents from a folder
# This function loads all text files from a specified folder and returns a list of Document objects
def load_text_documents(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(folder_path, file))
            documents.extend(loader.load())
    return documents

# 3. Batch documents based on token count
# This function groups documents into batches based on a maximum token count
def batch_documents_by_tokens(documents, max_tokens=200):
    batches = []
    current_batch = ""
    for doc in documents:
        content = doc.page_content.strip()
        if not content:
            continue
        if len(tokenizer.encode(current_batch + content)) <= max_tokens:
            current_batch += "\n" + content
        else:
            batches.append(Document(page_content=current_batch))
            current_batch = content
    if current_batch:
        batches.append(Document(page_content=current_batch))
    return batches

# 4. Split batches into smaller chunks
# This function splits each batch into smaller chunks using RecursiveCharacterTextSplitter
# It allows you to specify chunk size and overlap for better context retention
def split_batches(batched_docs, chunk_size=200, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(batched_docs)

# 5. main method to execute the batching and splitting
# This script will load small text files, batch them into logical groups based on token count,
# and then split those batches into smaller chunks ready for embedding.
if __name__ == "__main__":
    folder = "text_file_batch/"  # your folder of small files
    raw_docs = load_text_documents(folder)
    
    print(f" 1) Loaded {len(raw_docs)} small documents")

    batched_docs = batch_documents_by_tokens(raw_docs, max_tokens=300)
    print(f" 2) Grouped into {len(batched_docs)} logical batches")

    split_docs = split_batches(batched_docs)
    print(f" 3) Split into {len(split_docs)} chunks ready for embedding")

    

    
