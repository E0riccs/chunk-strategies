import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.llm_handler import LLM_handler

import os
import re

class Evaluator:
    def __init__(self, model_id, exp_setting):
        self.llm_handler = LLM_handler(model_id=model_id)
    
    def _calculate_cosine_similarity(self, original_text, chunks):
        """Calculates the average cosine similarity between original text and its chunks."""
        if not chunks:
            return 0.0
        
        vectorizer = TfidfVectorizer()
        try:
            # Ensure original_text is not empty and chunks are not all empty strings
            if not original_text.strip() or all(not chunk.strip() for chunk in chunks):
                # print("Warning: Original text or all chunks are empty. Cosine similarity cannot be computed.")
                return 0.0

            tfidf_matrix = vectorizer.fit_transform([original_text] + chunks)
            cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            return np.mean(cosine_similarities) if cosine_similarities.size > 0 else 0.0
        except ValueError as e:
            # This can happen if vocabulary is empty (e.g., all stop words or very short text)
            # print(f"Warning: Could not compute TF-IDF, possibly due to empty vocabulary: {e}. Returning 0 for cosine similarity.")
            return 0.0

    def _get_llm_evaluation(self, original_text, chunks):
        """Gets evaluation score from an LLM (e.g., OpenAI GPT)."""
        if not self.llm_handler.api or not chunks:
            # print("LLM client not initialized or no chunks to evaluate. Skipping LLM evaluation.")
            return None # Or a default score like 0 or -1

        # Prepare a prompt for the LLM
        # This prompt needs to be carefully designed for the specific task
        prompt_text = f"""Please evaluate the quality of the following text chunks based on the original text. 
Consider coherence, completeness of information within chunks, and how well they represent the original content. 
Rate on a scale of 1 to 5, where 1 is poor and 5 is excellent. 
Provide only the numeric score (e.g., 4 or 3.5).

Original Text (first 200 chars for context):
{original_text[:200]}...

Chunks (first 50 chars of each, up to 5 chunks):
"""
        for i, chunk in enumerate(chunks[:5]): # Limit to 5 chunks for brevity in prompt
            prompt_text += f"{i+1}. {chunk[:50]}...\n"
        if len(chunks) > 5:
            prompt_text += "... (and more chunks)\n"
        
        prompt_text += "\nScore (1-5): "

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model_name,
                messages=[
                    {"role": "system", "content": "You are an expert text chunking evaluator."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.2, # Low temperature for more deterministic output
                max_tokens=10 
            )
            content = response.choices[0].message.content.strip()
            # Extract a number (integer or float) from the response
            match = re.search(r"(\d+(\.\d+)?)", content)
            if match:
                return float(match.group(1))
            else:
                print(f"Warning: LLM did not return a parseable numeric score. Response: '{content}'")
                return None
        except Exception as e:
            print(f"Error during LLM API call: {e}")
            return None

    def evaluate(self, gen_qas, std_qas):
        """
            Calculates all evaluation metrics.

            Args:
                gen_qas (list of dict): Generated QAs.
                std_qas (list of dict): Standard QAs.
            
            Returns:
                metrics (dict): Evaluation metrics.

        """
        avg_cosine_sim = self._calculate_cosine_similarity(std_qas, gen_qas)
        llm_score = self._get_llm_evaluation(std_qas, gen_qas)

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