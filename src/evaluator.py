import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.llm_utils.llm_model import PromptKeywordsModel
from src.llm_handler import LLM_handler

import re

class Evaluator:
    def __init__(self, model_id, exp_setting):
        self.llm_model_id = model_id
    
    def _calculate_cosine_similarity(self, text1_list, text2):
        """Calculates the average cosine similarity between 2 texts."""

        # merge text1_list into a single string
        text1 = '\n'.join(text1_list)
        if not text1_list or not text2:
            return 0.0
        
        vectorizer = TfidfVectorizer()
        try:
            # Ensure original_text is not empty and chunks are not all empty strings
            if not text1.strip() or not text2.strip():
                print("Warning: Original text or all chunks are empty. Cosine similarity cannot be computed.")
                return 0.0

            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            return np.mean(cosine_similarities) if cosine_similarities.size > 0 else 0.0
        except ValueError:
            print(f"Warning: Could not compute TF-IDF, possibly due to empty vocabulary: {e}. Returning 0 for cosine similarity.")
            return 0.0

    def _get_llm_evaluation(self, question_text, std_answer, final_answer, documents):
        """Gets evaluation score from an LLM."""
        llm_handler = LLM_handler(model_id=self.llm_model_id)

        # Prompt
        std_answer = '\n'.join(std_answer)
        prompt_params = PromptKeywordsModel(
            question=question_text,
            document=documents,
            std = std_answer,
            judge = final_answer
        )
        judgement = llm_handler.generate_judgement(**prompt_params.model_dump(exclude_none=True))
        
        return judgement['score'] # KEY defined in Prompt

    def evaluate(self, gen_qas, std_qas, documents):
        """
            Calculates all evaluation metrics.

            Args:
                gen_qas (list of dict): Generated QAs with strict structure. (Key-Value)
                std_qas (list of dict): Standard QAs with strict structure. (Key-Value)
                documents (list of dict): Documents retrieved from vector store.
            
            Returns:
                metrics (dict): Evaluation metrics.

        """
        avg_cosine_sims, llm_scores= [], []
        for i in range(len(std_qas)):
            avg_cosine_sims.append(self._calculate_cosine_similarity(std_qas[i]['answer'], gen_qas[i]['final_answer']))
            llm_scores.append(self._get_llm_evaluation(std_qas[i]['question'],std_qas[i]['answer'], gen_qas[i]['final_answer'], documents[i]))

        metrics = {
            'total_processing_time_seconds': round(processing_time, 4),
            'llm_evaluation_score_1_to_5': llm_score if llm_score is not None else 'N/A',
            'avg_cosine_similarity_chunks_vs_original': round(avg_cosine_sim, 4),
            'number_of_chunks': len(chunks),
            'evaluation_module_runtime_seconds': round(evaluation_duration, 4)
        }
        return metrics

if __name__ == '__main__':
    # Example Usage
    # IMPORTANT: Set your OPENAI_API_KEY environment variable for LLM evaluation to run
    # export OPENAI_API_KEY='your_api_key_here'
    
    print("Initializing Evaluator...")
    # Pass API key directly if not in env, or for testing specific keys:
    # evaluator = Evaluator(llm_api_key='YOUR_OPENAI_API_KEY') 
    evaluator = Evaluator()

    sample_original_text = "This is a longer sample text designed for testing the evaluation metrics. It contains multiple sentences and ideas that should ideally be preserved across chunks. We are looking for coherence and completeness. The quick brown fox jumps over the lazy dog. This sentence adds more unique words. Another sentence to make the text longer and provide more substance for chunking and evaluation."
    sample_chunks_good = [
        "This is a longer sample text designed for testing the evaluation metrics.",
        "It contains multiple sentences and ideas that should ideally be preserved across chunks.",
        "We are looking for coherence and completeness.",
        "The quick brown fox jumps over the lazy dog.",
        "This sentence adds more unique words.",
        "Another sentence to make the text longer and provide more substance for chunking and evaluation."
    ]
    sample_chunks_fragmented = [
        "This is a longer sample text designed",
        "for testing the evaluation metrics. It contains multiple",
        "sentences and ideas that should ideally be preserved",
        "across chunks. We are looking for coherence and completeness.",
        "The quick brown fox jumps over the lazy dog. This sentence",
        "adds more unique words. Another sentence to make the text longer",
        "and provide more substance for chunking and evaluation."
    ]
    sample_processing_time = 0.05 # Example processing time in seconds

    print("\nEvaluating 'good' chunks...")
    metrics_good = evaluator.evaluate(sample_original_text, sample_chunks_good, sample_processing_time)
    print("Metrics for 'good' chunks:")
    for key, value in metrics_good.items():
        print(f"  {key}: {value}")

    print("\nEvaluating 'fragmented' chunks...")
    metrics_fragmented = evaluator.evaluate(sample_original_text, sample_chunks_fragmented, sample_processing_time + 0.02)
    print("Metrics for 'fragmented' chunks:")
    for key, value in metrics_fragmented.items():
        print(f"  {key}: {value}")

    print("\nEvaluating with empty original text (should handle gracefully)...")
    metrics_empty_orig = evaluator.evaluate("", sample_chunks_good, 0.01)
    print("Metrics for empty original text:")
    for key, value in metrics_empty_orig.items():
        print(f"  {key}: {value}")

    print("\nEvaluating with empty chunks (should handle gracefully)...")
    metrics_empty_chunks = evaluator.evaluate(sample_original_text, [], 0.01)
    print("Metrics for empty chunks:")
    for key, value in metrics_empty_chunks.items():
        print(f"  {key}: {value}")

    # Test case with very short text that might lead to empty vocabulary for TF-IDF
    short_text = "a a a"
    short_chunks = ["a a", "a"]
    print("\nEvaluating with very short text (potential empty TF-IDF vocabulary)...")
    metrics_short_text = evaluator.evaluate(short_text, short_chunks, 0.01)
    print("Metrics for short text:")
    for key, value in metrics_short_text.items():
        print(f"  {key}: {value}")

    print("\nNote: If LLM evaluation shows 'N/A' or errors, ensure OPENAI_API_KEY is correctly set.")