"""Embedding/Reranker API Response Model.
This module defines the standardized response format for all API platforms using Pydantic models.
"""

from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field

class APIResponseModel(BaseModel):
    """Pydantic model for API response standardization.
    
    This model defines the standard fields for API responses,
    ensuring type safety and validation.
    """
    tokens_used: int = Field(..., description="使用的总共token数量")
    model: str = Field(..., description="本次使用的模型名称")
    api_platform: str = Field(..., description="本次使用的API平台名称")
    response_content: Union[List[float], List[str], str] = Field(..., description="返回的内容")
    
    class Config:
        validate_assignment = True
        extra = "forbid"  # Prevents adding fields that aren't defined in the model

class RagAnswerModel(BaseModel):
    """Pydantic model for RAG answer standardization.
    
    This model defines the standard fields for RAG answers,
    ensuring type safety and validation.
    """
    final_answer: str = Field(..., description="最终答案")
    retrieved_documents_count: int = Field(..., description="检索到的文档数量")
    reranked_documents_count: int = Field(..., description="重排序后的文档数量")
    context_for_answer: Union[str, List[str]] = Field(..., description="答案上下文")
    
    class Config:
        validate_assignment = True
        extra = "forbid"  # Prevents adding fields that aren't defined in the model
    
    