# Load_excel_file.py
# Load an Excel file using LangChain's UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredExcelLoader

# 1. Specify the path to your Excel file
file_path = "sample_excel_file.xlsx"

# 2. Load the Excel file
# This will create a loader instance and load the content of the file
loader = UnstructuredExcelLoader(file_path)

# 3. Load the documents from the Excel file
# This will parse the Excel and create a list of documents
documents = loader.load()

# 4. Print the content of the loaded documents
# Each document corresponds to a sheet in the Excel file
for i, doc in enumerate(documents):
    print(f"\n--- Document {i+1} ---")
    print(doc.page_content)
