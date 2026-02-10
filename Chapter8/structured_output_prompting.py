# structure_output_prompting.py
# Demonstrates structured output prompting using a small LLM and strict JSON format.
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Load model and tokenizer using a small model flan-t5-base for demonstration
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="cpu")

# 2. Question and retrieved contexts
query = "How does RAG reduce hallucinations?"
contexts = [
    "It reduces hallucinations by grounding answers in retrieved documents instead of relying only on the model's memory.",
    "Retrieval-Augmented Generation (RAG) is a technique that improves AI responses by combining document retrieval with language generation."
]
# 3. Combine contexts into a single string
context_text = "\n".join(contexts)

# 4. Create a prompt enforcing JSON output
prompt = f"""
You are a helpful assistant.
Answer the question strictly in the following JSON format:

{{
  "query": "...",
  "retrieved_context": ["...", "..."],
  "answer": "..."
}}

Question: {query}
Context:
{context_text}

Answer:
"""

# 5. Generate the response
inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
outputs = model.generate(**inputs, max_new_tokens=200)
raw_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# 6. Parse the JSON response
# Attempt to parse the response; if it fails, return raw text
try:
    parsed = json.loads(raw_answer)
except json.JSONDecodeError:
    parsed = {
        "query": query,
        "retrieved_context": contexts,
        "answer": raw_answer
    }

# 7. Print structured JSON output
print(json.dumps(parsed, indent=2))
