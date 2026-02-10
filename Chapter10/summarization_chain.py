
# Import required libraries
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# 1. Choose a factual summarization model
# 'facebook/bart-large-cnn' is known for concise, accurate summaries.
model_name = "facebook/bart-large-cnn"

# 2. Load tokenizer and model
# Explicitly set 'model_max_length' to prevent truncation warnings.
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    model_max_length=1024,   # Maximum token length the model can handle
    truncation=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 3. Select device (GPU if available, else CPU)
device = 0 if torch.cuda.is_available() else -1
print(f"Device set to: {'GPU' if device == 0 else 'CPU'}")

# 4. Create the summarization pipeline
# 'pipeline' handles tokenization, model inference, and decoding.
summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=device,
    truncation=True
)

# 5. Input text to summarize
text = """Retrieval-Augmented Generation (RAG) is a technique
that reduces hallucinations in large language models by grounding answers
in external evidence. Instead of generating responses purely from
parameters, RAG retrieves relevant context from a knowledge base and
conditions the generation on it. This improves factual accuracy and
reduces the chances of fabricated information."""

# 6. Clean and normalize the input text
# Removes unnecessary newlines or spaces for better summarization
text = " ".join(line.strip() for line in text.splitlines() if line.strip())

# 7. Dynamically calculate summary length limits
# Ensures output is proportional to input size for consistent results.
input_length = len(text.split())
max_len = min(100, max(30, int(input_length * 0.6)))  # upper limit for summary tokens
min_len = int(max_len * 0.4)                          # lower limit for summary tokens
print(f"Dynamic length settings: min={min_len}, max={max_len}")

# 8. Generate summary
# Uses deterministic decoding (beam search) to ensure factual output.
summary = summarizer(
    text,
    max_length=max_len,
    min_length=min_len,
    truncation=True,
    do_sample=False,     # Disable random sampling for consistent output
    num_beams=4,         # Explore multiple candidate summaries
    length_penalty=1.8   # Encourage shorter, more concise summaries
)[0]['summary_text']

# 9. Display results ===
print("\n=== Original Text ===")
print(text)
print("\n=== Faithful Summary ===")
print(summary)
