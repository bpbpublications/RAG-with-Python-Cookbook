# load_csv_file.py
# Load a CSV file using LangChain's CSVLoader
from langchain_community.document_loaders import CSVLoader


# 1. Specify the path to your CSV file
# Make sure to replace 'sample_csv_file.csv' with the actual path to your CSV file
file_path = "sample_csv_file.csv"

# 2. Load the CSV file
# This will create a loader instance and load the content of the file
loader = CSVLoader(file_path)

# 3. Load the documents from the CSV file
# This will parse the CSV and create a list of documents
documents = loader.load()

# 4. Print the content of the loaded documents
# Each document corresponds to a row in the CSV file
for i, doc in enumerate(documents):
    print(f"\n--- Document {i+1} ---")
    print(doc.page_content)
