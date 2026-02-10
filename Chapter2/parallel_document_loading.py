# parallel_document_loading.py
# This code demonstrates how to load documents in parallel using LangChain's document loaders.
# It supports loading PDF, TXT, and DOCX files concurrently to improve efficiency.
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 1. Get the appropriate loader based on file extension
# This function returns the correct loader for the given file type.
def get_loader(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path)
    elif ext in [".docx", ".doc"]:
        return UnstructuredWordDocumentLoader(file_path)
    else:
        return None

# 2. Load documents in parallel
    # This function uses ThreadPoolExecutor to load documents concurrently.
    # In the directory, it will look for files with supported extensions and load them.
    # In the directory, there are 3 files with supported extensions like PDF, TXT, and DOCX.
def load_documents_parallel(file_paths, max_workers=4):
    documents = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_loader(fp).load): fp # type: ignore
            for fp in file_paths if get_loader(fp)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading"):
            try:
                result = future.result()
                documents.extend(result)
            except Exception as e:
                print(f"Failed to load {futures[future]}: {e}")
    return documents

# 3. Get file paths from a directory
# This function retrieves all supported file paths from a specified directory.
def get_file_paths(directory):
    supported_exts = {".pdf", ".txt", ".docx"}
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in supported_exts
    ]

# Main execution block - 
# This part of the code runs when the script is executed directly.
# It retrieves file paths from a specified directory and loads the documents in parallel.
# Change 'documents/' to the directory containing your files.   
if __name__ == "__main__":
    folder_path = "documents/"  # change this
    files = get_file_paths(folder_path)
    all_docs = load_documents_parallel(files)
    # 4. Print the number of loaded documents
    # This will output the total number of documents loaded from the specified directory.
    print(f"\nLoaded {len(all_docs)} documents.")
