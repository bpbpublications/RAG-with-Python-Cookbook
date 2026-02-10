# context_grounded_prompting.py
# Example of context-grounded prompting using lightweight models suitable for CPU
from sentence_transformers import SentenceTransformer, util
from transformers.pipelines import pipeline

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Initialize models and context using a small embedding model for efficiency
# Pre-computing embeddings for context to speed up retrieval and splitting context into manageable chunks (sentences)
class ContextGroundedResponder:
    def __init__(self, context_text: str):
        # 1. Split context text into sentences for finer retrieval granularity 
        self.context_sentences = [s.strip() for s in context_text.split(".") if s.strip()]
        
        # 2. Load small embedding model sentence-transformers/all-MiniLM-L6-v2 suitable for CPU  
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # 3. Pre-compute context embeddings.This speeds up retrieval during queries using convert_to_tensor = True for efficient similarity search
        self.context_embeddings = self.model.encode(self.context_sentences, convert_to_tensor=True)

        # 4. Load a lightweight LLM for text generation using a small model suitable for CPU
        self.llm = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)

# Method to answer queries using retrieved context to ground the response and stronger prompt to ensure complete sentences
# Limiting response length for conciseness
    def answer(self, query: str, top_k: int = 2, max_length: int = 150):
        # 5. Embed the query for similarity search 
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # 6. Retrieve top-k relevant context sentences using semantic search to find the most relevant context
        hits = util.semantic_search(query_embedding, self.context_embeddings, top_k=top_k)[0]

        # 7. Combine retrieved sentences into a single context string 
        retrieved_context = " ".join([self.context_sentences[hit['corpus_id']] for hit in hits]) # type: ignore

        # 8. Create a strong prompt with retrieved context
        # Emphasizing clarity and completeness in the response
        prompt = f"""You are a helpful assistant. 
Use the following context to answer the question in one or two complete sentences. 
Do not just repeat words — provide a clear explanation.

Context: {retrieved_context}

Question: {query}

Answer:"""

        # 9. Generate the answer using the LLM
        # Limiting the length of the response for conciseness
        llm_output = self.llm(prompt, max_length=max_length, num_return_sequences=1)
        answer_text = llm_output[0]["generated_text"]

        # 10. Print the results for clarity  
        # Print the query, retrieved context, and final answer
        print("Query:", query)
        print("Retrieved Context:", retrieved_context)
        print("Answer:", answer_text, "\n")

        return answer_text


# Example usage
if __name__ == "__main__":
    document_text = """
    Retrieval-Augmented Generation (RAG) is a technique that improves AI responses by combining 
    document retrieval with language generation. It reduces hallucinations by grounding answers 
    in retrieved documents instead of relying only on the model's memory.
    """

    bot = ContextGroundedResponder(document_text)
    query = "How does RAG reduce hallucinations?"
    bot.answer(query)
