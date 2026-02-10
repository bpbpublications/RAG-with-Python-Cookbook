# progressive_discosure_response.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import time

# 1. Load model and tokenizer (local, no API key needed)
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 2. Sample document text
# This text can be replaced with any relevant document.
document_text = """
Retrieval-Augmented Generation (RAG) is an AI technique that combines information retrieval with
natural language generation to improve the accuracy of answers. In RAG, when a question is asked,
the system first searches for relevant documents or data sources, then uses a language model
to generate a response based on the retrieved information. This method reduces hallucinations
by grounding the model's output in real data. RAG is widely used in chatbots, customer support,
and knowledge management systems to provide factual and context-aware responses.
"""

# 3. Define the method to retrieve context from the document
def retrieve_context(document: str, query: str, window: int = 200) -> str:
    query_words = query.lower().split()
    doc_lower = document.lower()
    for word in query_words:
        idx = doc_lower.find(word)
        if idx != -1:
            start = max(idx - window, 0)
            end = min(idx + window, len(document))
            return document[start:end]
    return document[:400]

# 4. Define the method to get model response with context
def get_model_response_with_context(query: str, context: str) -> str:
    prompt = f"Answer the question using the following context:\n{context}\n\nQuestion: {query}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=600,
        do_sample=True,
        top_p=0.95,
        temperature=0.7
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if not response:
        response = "Sorry, I could not generate a response based on the document."
    return response

# 5. Method to progressively disclose response
def progressive_disclosure_auto(response: str, chunk_size: int = 2, delay: float = 2):
    """
    Automatically reveals response in chunks with a time delay.
    """
    sentences = re.split(r'(?<=[.!?]) +', response)
    total = len(sentences)
    start = 0

    while start < total:
        end = min(start + chunk_size, total)
        chunk = " ".join(sentences[start:end])
        print("\n" + chunk + "\n")
        start = end
        time.sleep(delay)  # pause before next chunk

# 6. Example queries
queries = [
    "Explain RAG in AI",
    "How does RAG reduce hallucinations?"
]

# 7. Run the progressive disclosure for each query and print the response
for query in queries:
    print(f"\nQuery: {query}\n")
    context = retrieve_context(document_text, query)
    full_response = get_model_response_with_context(query, context)
    print("Bot is revealing response automatically:\n")
    progressive_disclosure_auto(full_response, chunk_size=2, delay=2)
    print("\n--- End of response ---\n")
