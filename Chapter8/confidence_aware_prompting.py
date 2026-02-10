# confidence_aware_prompting.py
# Demonstrates confidence-aware prompting using a small LLM and sentence embeddings.
from sentence_transformers import SentenceTransformer, util
from transformers.pipelines import pipeline

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

class ConfidenceAwareResponder:
    def __init__(self, context_text: str):
        # 1. Prepare context sentences
        # Split the document into sentences for better retrieval granularity
        self.context_sentences = [s.strip() for s in context_text.split(".") if s.strip()]
        
        # 2. Context embeddings using SentenceTransformer
        # using a smaller model all-MiniLM-L6-v2 for efficiency
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.context_embeddings = self.embedding_model.encode(self.context_sentences, convert_to_tensor=True)

        # 3. Pipeline for text generation using a small model text2text-generation
        self.generator = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)

    def retrieve_context(self, query: str, top_k: int = 2):
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.context_embeddings, top_k=top_k)[0]
        return [(self.context_sentences[hit['corpus_id']], float(hit['score'])) for hit in hits] # type: ignore

    def answer(self, query: str, top_k: int = 2):
        # 4. Retrieve relevant contexts for the query
        retrieved = self.retrieve_context(query, top_k)

        # 5. Combine multiple contexts into a single string
        context_text = " ".join([ctx for ctx, _ in retrieved])
        
        # 6. Create a prompt with the query and retrieved context
        prompt = (
            f"Question: {query}\n"
            f"Context: {context_text}\n\n"
            f"Answer clearly and concisely."
        )

        # 7. Generate model output
        # Using max_length=200 and clean_up_tokenization_spaces=True for better output
        model_output = self.generator(prompt, clean_up_tokenization_spaces=True)[0]["generated_text"]

        # 8. Estimate retrieval confidence based on similarity scores
        avg_score = sum(score for _, score in retrieved) / len(retrieved)
        if avg_score > 0.7:
            confidence = "High"
        elif avg_score > 0.5:
            confidence = "Medium"
        else:
            confidence = "Low"

        # 9. Return the answer along with confidence and retrieved contexts
        return f"""
    Query: {query}
    Retrieved Contexts:
""" + "\n".join([f"- {ctx} (score={score:.3f})" for ctx, score in retrieved]) + f"""

    Model output:
{model_output}

    Confidence: {confidence} (avg similarity={avg_score:.3f})
"""


# Example usage
if __name__ == "__main__":
    document_text = """
    Retrieval-Augmented Generation (RAG) is a technique that improves AI responses by combining 
    document retrieval with language generation. 
    It reduces hallucinations by grounding answers in retrieved documents 
    instead of relying only on the model's memory.
    """

    bot = ConfidenceAwareResponder(document_text)
    query = "How does RAG reduce hallucinations?"
    # 10. Print the answer with confidence score
    print(bot.answer(query))
