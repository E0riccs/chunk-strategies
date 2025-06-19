"""
Evaluation model which define the metrics dict keywords.
"""
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field


class EvalResponseModel(BaseModel):
    """Pydantic model for standardizing evaluation metrics.
    
    This model defines the standard metrics that can be used in evaluation,
    ensuring type safety and validation.
    """
    cosine_similarity: Optional[float] = Field(None, description="Cosine similarity between std answer and gen answer")
    llm_score: Optional[float] = Field(None, description="LLM score for the generated answer")
    
    
