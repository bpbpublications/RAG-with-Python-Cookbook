# summarization_prompting.py

from transformers import pipeline
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load the document to be summarized
loader = TextLoader("chapter8_RAG.txt")   # Replace with your document
docs = loader.load()

# 2. Split into chunks to manage context length
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# 3. Merge chunks to give the model enough context
full_text = " ".join([chunk.page_content for chunk in splits])

# 4. Initialize a summarization pipeline using a dedicated summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# 5. User query to control summarization focus
query = "Summarize this text focusing on how RAG reduces hallucinations."

# 6. Combine query and text in a clear prompt for summarization
prompt_text = f"{query}\n\n{full_text}"

# 7. Generate the summary with adjusted parameters for multi-sentence output
summary = summarizer(
    prompt_text,
    max_length=84,    # Adjust max length for multi-sentence summaries
    min_length=80,     # Ensure at least a few sentences
    do_sample=False
)

final_summary = summary[0]["summary_text"]

# 8. Print the query and final summary results
print("\n--- USER QUERY ---")
print(query)
print("\n--- DOCUMENT SUMMARY ---")
print(final_summary)
