# preprocess_during_loading.py
# Preprocess During Document Loading
# This script demonstrates how to preprocess text documents during loading using LangChain
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
import nltk
from nltk.corpus import stopwords
import re

# 1. Download NLTK stopwords if not already downloaded
nltk.download("stopwords")

# 2. Load a text file using LangChain's TextLoader
loader = TextLoader("RAG.txt")

# 3. Load the documents from the text file using the loader
# This will create a loader instance and load the content of the file
docs = loader.load()

# 4. Define stopwords
# This will be used to filter out common words that do not contribute to the meaning
stop_words = set(stopwords.words("english"))

# 5. Define a preprocessing function
# This function will clean the text by lowercasing, removing stopwords, and normalizing
def preprocess(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove stopwords
    words = text.split()
    filtered = [word for word in words if word not in stop_words]
    text = ' '.join(filtered)
    
    # 3. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 6. Apply preprocessing to each document
# This will create a new list of documents with cleaned content
processed_docs = []
for doc in docs:
    cleaned = preprocess(doc.page_content)
    processed_doc = Document(
        page_content=cleaned,
        metadata=doc.metadata  # Preserve metadata
    )
    processed_docs.append(processed_doc)

# 7. Print the content of the cleaned documents
# Each document will have the preprocessed content
for doc in processed_docs:
    print("\nCleaned Document:\n", doc.page_content[:300])
