# embed_list_of_documents.py
# This code demonstrates how to embed a list of documents using the HuggingFace Transformers library.
from transformers import AutoTokenizer, AutoModel
import torch

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Load the pre-trained model and tokenizer
# The model 'sentence-transformers/all-MiniLM-L6-v2' is suitable for generating embeddings.
# You can replace it with any other model available in HuggingFace.
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. Define a list of documents to be embedded
# Each document is a string that will be transformed into a vector representation.
documents = [
    "RAG is a powerful technique for combining retrieval with language models.",
    "Document embeddings enable efficient semantic search.",
    "Transformers provide contextualized token representations."
]

# 3. Tokenize the documents
# The tokenizer converts the list of documents into a format suitable for the model.
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

# 4. Embed the documents
def embed_documents(docs):
    encoded = tokenizer(docs, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded)
    return mean_pooling(output, encoded['attention_mask'])

# 5. Generate embeddings for the list of documents
# The embed_documents function is called with the list of documents to get their embeddings.
embeddings = embed_documents(documents)

# 6. Print the resulting embeddings
# Each embedding is a vector representation of the corresponding document.
for i, vec in enumerate(embeddings):
    print(f"Document {i+1} vector (first 5 dims): {vec[:5]}")
