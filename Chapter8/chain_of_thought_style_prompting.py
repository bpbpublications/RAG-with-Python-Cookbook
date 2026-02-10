# chain_of_thought_style_prompting.py
# Example of chain-of-thought style prompting using lightweight models suitable for CPU
from sentence_transformers import SentenceTransformer, util
from transformers.pipelines import pipeline

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

class ChainOfThoughtResponder:
    def __init__(self, context_text: str):
        # 1. Split context into sentences for finer retrieval granularity
        self.context_sentences = [s.strip() for s in context_text.split(".") if s.strip()]
        
        # 2. Load small embedding model for CPU 
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # 3. Pre-compute context embeddings. This speeds up retrieval during queries
        # Using convert_to_tensor for efficient similarity search
        self.context_embeddings = self.embedder.encode(self.context_sentences, convert_to_tensor=True)

        # 4. Load a lightweight LLM for text generation
        # Using a small model suitable for CPU
        self.llm = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)

    def answer(self, question: str, top_k: int = 2, max_length: int = 200):
        # 5. Embed the question for similarity search
        query_embedding = self.embedder.encode(question, convert_to_tensor=True)

        # 6. Retrieve top-k relevant context sentences
        hits = util.semantic_search(query_embedding, self.context_embeddings, top_k=top_k)[0]
        retrieved = [self.context_sentences[hit['corpus_id']] for hit in hits] # type: ignore

        # 7. Create ranked context string
        ranked_context = "\n".join([f"  Top {i+1}: {sent}" for i, sent in enumerate(retrieved)])

        # 8. Create a chain-of-thought style prompt
        # Encouraging step-by-step reasoning before the final answer
        prompt = f"""You are a reasoning assistant.
Use the context below to answer the question.

Context:
{ranked_context}

Question: {question}

Think step by step:
- Write 2–3 bullet points of reasoning based on the context.
- Then write the final answer clearly after 'Final Answer:'.

Format:
- Reasoning bullets
---
Final Answer: <short answer>
"""

        # 9. Generate the answer using the LLM
        # Limiting the length of the response for conciseness
        llm_output = self.llm(prompt, num_return_sequences=1)
        answer_text = llm_output[0]["generated_text"]

        # 10. Ensure the final answer is clearly marked
        if "Final Answer:" not in answer_text:
            answer_text += f"\nFinal Answer: {retrieved[0]}"

        # 11. Print the results for clarity
        # Print the question, ranked context, and final answer
        print("Question:", question)
        print("Retrieved Context (ranked):")
        print(ranked_context)
        print("\nModel output:\n")
        print(answer_text)

        return answer_text


# Example usage
if __name__ == "__main__":
    document_text = """
    Retrieval-Augmented Generation (RAG) is a technique that improves AI responses by combining 
    document retrieval with language generation. It reduces hallucinations by grounding answers 
    in retrieved documents instead of relying only on the model's memory.
    """

    bot = ChainOfThoughtResponder(document_text)
    bot.answer("How does RAG reduce hallucinations?")
