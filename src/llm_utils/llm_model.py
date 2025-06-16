"""
LLM API Response Model.
This module defines the standardized response format for all LLM API platforms.
"""
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field

class LLMResponseModel:
    # Standard keys for LLM response dictionary
    TOKENS_USED: str = 'tokens_used' # 使用的总共token数量
    GEN_MODEL: str = 'gen_model' # 本此使用的模型名称
    LLM_PLATFORM: str = 'llm_platform'   # 本此使用的 api 平台名称
    ANS_CONTENT: str = 'ans_content' # 回答的内容
    THINKING_CONTENT: str = 'thinking_content'  # 思考模型的思考过程

    # 对文本进行预处理的函数
    @staticmethod
    def preprocess_text(text: str) -> str:
        """Preprocess the input text for better model understanding.
        Args:
            text (str): The original text to preprocess.
        Returns:
            str: The preprocessed text.
        """
        if not text:
            return ""
        
        # 去除文本中所有 \n \\n 
        text = text.replace('\n', ' ').replace('\\n', ' ').strip()

        # 删除文本中第一个[前的内容
        if '[' in text:
            text = '[ ' + text.split('[', 1)[1]
        # 删除文本中最后一个]后的内容
        if ']' in text:
            text = text.rsplit(']', 1)[0] + ']'

        return text

    @classmethod
    def create_response(cls, 
                       tokens_used: int, 
                       gen_model: str, 
                       llm_platform: str, 
                       ans_content: str, 
                       thinking_content: Optional[str] = None) -> Dict[str, Any]:
        """Create a standardized response dictionary.
        
        Args: 
            The valus of the keys.
            
        Returns:
            Dict[str, Any]: Standardized response dictionary.
        """
        return {
            cls.TOKENS_USED: tokens_used,
            cls.GEN_MODEL: gen_model,
            cls.LLM_PLATFORM: llm_platform,
            cls.ANS_CONTENT: ans_content,
            cls.THINKING_CONTENT: thinking_content
        }

    @classmethod
    def get_keys(cls):
        """Get all standard keys as a dictionary.
        
        Returns:
            dict: Dictionary mapping key names to their values
        """
        return {
            'TOKENS_USED': cls.TOKENS_USED,
            'GEN_MODEL': cls.GEN_MODEL,
            'LLM_PLATFORM': cls.LLM_PLATFORM,
            'ANS_CONTENT': cls.ANS_CONTENT,
            'THINKING_CONTENT': cls.THINKING_CONTENT
        }

class PromptKeywordsModel(BaseModel):
    """Pydantic model for standardizing prompt keywords.
    
    This model defines the standard keywords that can be used in prompts,
    ensuring type safety and validation when passing parameters to LLM calls.
    """
    document: Optional[Union[str, List[str]]] = Field(None, description="Document content for context")
    question: Optional[str] = Field(None, description="Question to be answered")
    std: Optional[str] = Field(None, description="Standard content for comparison")
    judge: Optional[str] = Field(None, description="Judge or evaluation criteria")