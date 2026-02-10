# multi_prompt_ensemble.py
# Demonstrates multi-prompting and ensemble techniques using a small LLM and sentence embeddings.
from sentence_transformers import SentenceTransformer, util
from transformers.pipelines import pipeline

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

class MultiPromptEnsembleResponder:
    def __init__(self, context_text: str):
        # 1. Prepare context sentences
        self.context_sentences = [s.strip() for s in context_text.split(".") if s.strip()]
        
        # 2. Context embeddings using SentenceTransformer
        # using a smaller model all-MiniLM-L6-v2 for efficiency
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.context_embeddings = self.embedding_model.encode(self.context_sentences, convert_to_tensor=True)

        # 3. Pipeline for text generation using a small model text2text-generation
        self.generator = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)

    def retrieve_context(self, query: str, top_k: int = 2):
        # Semantic search to find top_k relevant contexts
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.context_embeddings, top_k=top_k)[0]
        return [self.context_sentences[hit['corpus_id']] for hit in hits] # type: ignore

    def generate_with_prompts(self, query: str, contexts: list):
        # Multiple prompting strategies
        prompts = [
            f"Answer the question directly.\nQuestion: {query}\nContext: {contexts}",
            f"Explain step by step before giving the final answer.\nQuestion: {query}\nContext: {contexts}",
            f"Give a concise one-line answer.\nQuestion: {query}\nContext: {contexts}"
        ]

        outputs = []
        for i, prompt in enumerate(prompts, 1):
            result = self.generator(prompt, clean_up_tokenization_spaces=True)[0]["generated_text"]
            outputs.append(f"Prompt {i} output: {result}")
        return outputs

    def ensemble_answer(self, query: str, top_k: int = 2):
        # 4. Retrieve relevant contexts for the query
        retrieved_contexts = self.retrieve_context(query, top_k)

        # 5. Combine multiple contexts into a single string
        contexts_text = " ".join(retrieved_contexts)

        # 6. Generate outputs using multiple prompts
        prompt_outputs = self.generate_with_prompts(query, contexts_text) # type: ignore

        # 7. Ensemble the outputs into a final answer
        final_prompt = (
            f"Here are multiple answers to the same question:\n\n"
            f"{prompt_outputs[0]}\n\n{prompt_outputs[1]}\n\n{prompt_outputs[2]}\n\n"
            f"Now combine them into a single best answer.\nFinal Answer:"
        )
        final_answer = self.generator(final_prompt, clean_up_tokenization_spaces=True)[0]["generated_text"]

        # 8. Return the final answer along with retrieved contexts and individual prompt outputs
        return f"Query: {query}\n Retrieved Contexts:\n- {retrieved_contexts[0]}\n- {retrieved_contexts[1]}\n\n" \
               f"Multi-Prompt Outputs:\n" + "\n".join(prompt_outputs) + f"\n\n Ensemble Final Answer:\n{final_answer}"


# Example usage
if __name__ == "__main__":
    document_text = """
    Retrieval-Augmented Generation (RAG) is a technique that improves AI responses by combining 
    document retrieval with language generation. 
    It reduces hallucinations by grounding answers in retrieved documents 
    instead of relying only on the model's memory.
    """

    bot = MultiPromptEnsembleResponder(document_text)
    query = "How does RAG reduce hallucinations?"
    # 9. Print the ensemble answer
    print(bot.ensemble_answer(query))
