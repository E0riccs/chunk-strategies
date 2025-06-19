import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.llm_utils.llm_model import PromptKeywordsModel
from src.llm_handler import LLM_handler
from src.eval_utils.eval_model import EvalResponseModel

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
                metrics (list of EvalResponseModel): Evaluation metrics.

        """
        metrics = []
        for i in range(len(std_qas)):
            cosine_sim = self._calculate_cosine_similarity(std_qas[i]['answer'], gen_qas[i]['final_answer'])
            llm_score = self._get_llm_evaluation(std_qas[i]['question'],std_qas[i]['answer'], gen_qas[i]['final_answer'], documents[i])

            metrics.append(EvalResponseModel(
                cosine_similarity=cosine_sim,
                llm_score=llm_score
            ))

        return metrics