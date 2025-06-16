"""
LLM API Response Model.
This module defines the standardized response format for all LLM API platforms.
"""
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field

class LLMResponseModel(BaseModel):
    """Pydantic model for LLM response standardization.
    
    This model defines the standard fields for LLM responses,
    ensuring type safety and validation.
    """
    tokens_used: int = Field(..., description="使用的总共token数量")
    gen_model: str = Field(..., description="本次使用的模型名称")
    llm_platform: str = Field(..., description="本次使用的API平台名称")
    ans_content: str = Field(..., description="回答的内容")
    thinking_content: Optional[str] = Field(None, description="思考模型的思考过程")
    
    class Config:
        validate_assignment = True
        extra = "forbid"  # Prevents adding fields that aren't defined in the model

class PromptKeywordsModel(BaseModel):
    """Pydantic model for standardizing prompt keywords.
    
    This model defines the standard keywords that can be used in prompts,
    ensuring type safety and validation when passing parameters to LLM calls.
    """
    document: Optional[Union[str, List[str]]] = Field(None, description="Document content for context")
    question: Optional[str] = Field(None, description="Question to be answered")
    std: Optional[str] = Field(None, description="Standard content for comparison")
    judge: Optional[str] = Field(None, description="Judge or evaluation criteria")