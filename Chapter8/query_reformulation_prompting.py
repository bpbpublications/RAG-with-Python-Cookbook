# query_reformulation_prompting.py
# Example of query reformulation prompting using lightweight models suitable for CPU
from sentence_transformers import SentenceTransformer, util
from transformers.pipelines import pipeline

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

class QueryReformulationResponder:
    def __init__(self, context_text: str):
        # 1. Split context into sentences for finer retrieval granularity
        self.context_sentences = [s.strip() for s in context_text.split(".") if s.strip()]
                     
        # 2. Pre-compute context embeddings with Embedding model (all-MiniLM-L6-v2, CPU only)
        # This speeds up retrieval during queries using convert_to_tensor for efficient similarity search
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.context_embeddings = self.embedding_model.encode(self.context_sentences, convert_to_tensor=True)

        # 3. Load a lightweight LLM for text generation using a small model suitable for CPU
        self.generator = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)

    def reformulate_query(self, query: str) -> str:
        # Simple reformulation (could be more complex)
        return query.replace("RAG", "Retrieval-Augmented Generation (RAG)")

    def answer(self, query: str, top_k: int = 2):
        # 4. Reformulate the query for clarity and specificity  
        reformulated_query = self.reformulate_query(query)

        # 5. Embed the reformulated query for similarity search
        query_embedding = self.embedding_model.encode(reformulated_query, convert_to_tensor=True)

        # 6. Retrieve top-k relevant context sentences using semantic search
        hits = util.semantic_search(query_embedding, self.context_embeddings, top_k=top_k)[0]
        retrieved_contexts = [self.context_sentences[hit['corpus_id']] for hit in hits] # type: ignore

        # 7. Create a prompt that encourages step-by-step reasoning and a final concise answer
        prompt = (
            f"Question: {reformulated_query}\n"
            f"Context:\n- {retrieved_contexts[0]}\n- {retrieved_contexts[1]}\n\n"
            "Answer step by step using the context, "
            "then give a concise conclusion prefixed with 'Final Answer:'."
        )

        # 8. Generate the answer using the LLM
        # Limiting the length of the response for conciseness
        model_output = self.generator(prompt, clean_up_tokenization_spaces=True)[0]["generated_text"]

        # 9. Print the results for clarity
        # Print the original query, reformulated query, retrieved context, and final answer
        return f"Original Query: {query}\nReformulated Query: {reformulated_query}\n" \
               f"Retrieved Context (ranked):\n  Top 1: {retrieved_contexts[0]}\n  Top 2: {retrieved_contexts[1]}\n\n" \
               f"Model output:\n\n{model_output}"


# Example usage
if __name__ == "__main__":
    document_text = """
    Retrieval-Augmented Generation (RAG) is a technique that improves AI responses by combining 
    document retrieval with language generation. 
    It reduces hallucinations by grounding answers in retrieved documents 
    instead of relying only on the model's memory.
    """
# Create the responder with the provided context
    bot = QueryReformulationResponder(document_text)
# Ask a question that benefits from query reformulation
    query = "Why does RAG help AI give better answers?"
    print(bot.answer(query))
