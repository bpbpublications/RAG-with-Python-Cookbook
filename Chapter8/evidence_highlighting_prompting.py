# evidence_highlighting_prompting.py
# Demonstrates evidence highlighting in a document using a small LLM and keyword filtering.
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers.pipelines import pipeline
import re

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Load the document to be analyzed for evidence highlighting
loader = TextLoader("chapter8_RAG.txt")  # Replace with your document
docs = loader.load()

# 2. Split into chunks using character-based splitter with chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# 3. Merge chunks to give the model enough context for filtering
full_text = " ".join([chunk.page_content for chunk in splits])

# 4. Clean text to fix common encoding issues
clean_text = full_text.replace("â€TM", "'").replace("â€“", "-")

# 5. Split text into sentences for finer filtering
sentences = re.split(r'(?<=[.!?])\s+', clean_text)

# 6. Define keywords for filtering relevant evidence
keywords_all = ['RAG', 'hallucinations']             # Must include
keywords_any = ['reduces', 'grounding', 'factual accuracy']  # Optional but helpful

# 7. Filter sentences containing 'all' keywords and 'any' keywords
filtered_evidence = [
    s for s in sentences
    if all(k.lower() in s.lower() for k in keywords_all) 
       or any(k.lower() in s.lower() for k in keywords_any)
]

# 8. Initialize a summarization pipeline using a dedicated summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# 9. Summarize the filtered evidence to create concise highlights
if filtered_evidence:
    evidence_text = " ".join(filtered_evidence)
    summary = summarizer(
        evidence_text,
        max_length=11,
        min_length=5,
        do_sample=False
    )
    summary_text = summary[0]['summary_text']
else:
    summary_text = "No relevant evidence found."

# 10. Further filter summary sentences to ensure relevance
final_sentences = [
    s for s in re.split(r'(?<=[.!?])\s+', summary_text)
    if any(k.lower() in s.lower() for k in keywords_all + keywords_any)
]

bullet_points = "\n".join(f"- {s.strip()}" for s in final_sentences)

# 11. Print the user query and highlighted evidence
query = "Highlight sentences in the text that provide evidence on how RAG reduces hallucinations."
print("\n--- USER QUERY ---")
print(query)
print("\n--- HIGHLIGHTED EVIDENCE ---")
print(bullet_points)
