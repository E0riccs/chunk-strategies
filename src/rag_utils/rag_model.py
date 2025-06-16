"""
Embedding/Reranker API Response Model.
Prompt keyword replace Model.
This module defines the standardized response format for all API platforms.
"""

from typing import Optional, Dict, Any # 建议也导入 Dict 和 Any 用于返回类型提示

class APIResponseModel:
    # Standard keys for LLM response dictionary
    TOKENS_USED: str = 'tokens_used' # 使用的总共token数量
    MODEL: str = 'model' # 本此使用的模型名称
    API_PLATFORM: str = 'api_platform'   # 本此使用的 api 平台名称
    RESPONSE_CONTENT: str = 'response_content' # 返回的内容

    @classmethod
    def create_response(cls, 
                       tokens_used: int, 
                       model: str, 
                       api_platform: str, 
                       response_content: str) -> Dict[str, Any]:
        """Create a standardized response dictionary.
        
        Args: 
            The valus of the keys.
            
        Returns:
            Dict[str, Any]: Standardized response dictionary.
        """
        return {
            cls.TOKENS_USED: tokens_used,
            cls.MODEL: model,
            cls.API_PLATFORM: api_platform,
            cls.RESPONSE_CONTENT: response_content
        }

    @classmethod
    def get_keys(cls):
        """Get all standard keys as a dictionary.
        
        Returns:
            dict: Dictionary mapping key names to their values
        """
        return {
            'TOKENS_USED': cls.TOKENS_USED,
            'MODEL': cls.MODEL,
            'API_PLATFORM': cls.API_PLATFORM,
            'RESPONSE_CONTENT': cls.RESPONSE_CONTENT
        }
    
class RagAnswerModel:
    # Standard keys for RAG answer dictionary
    # Include the details that needed in experiment result

    FINAL_ANSWER: str = 'final_answer' # 最终答案
    RETRIEVED_DOCUMENTS_COUNT: str = 'retrieved_documents_count' # 检索到的文档数量
    RERANKED_DOCUMENTS_COUNT: str = 'reranked_documents_count' # 重排序后的文档数量
    CONTEXT_FOR_ANSWER: str = 'context_for_answer' # 答案上下文

    @classmethod
    def create_response(cls, 
                       final_answer: str, 
                       retrieved_documents_count: int, 
                       reranked_documents_count: int, 
                       context_for_answer: str) -> Dict[str, Any]:
        """Create a standardized response dictionary.
        
        Args: 
            The valus of the keys.
            
        Returns:
            Dict[str, Any]: Standardized response dictionary.
        """
        return {
            cls.FINAL_ANSWER: final_answer,
            cls.RETRIEVED_DOCUMENTS_COUNT: retrieved_documents_count,
            cls.RERANKED_DOCUMENTS_COUNT: reranked_documents_count,
            cls.CONTEXT_FOR_ANSWER: context_for_answer
        }
    
    @classmethod
    def get_keys(cls):
        """Get all standard keys as a dictionary.
        
        Returns:
            dict: Dictionary mapping key names to their values
        """
        return {
            'FINAL_ANSWER': cls.FINAL_ANSWER,
            'RETRIEVED_DOCUMENTS_COUNT': cls.RETRIEVED_DOCUMENTS_COUNT,
            'RERANKED_DOCUMENTS_COUNT': cls.RERANKED_DOCUMENTS_COUNT,
            'CONTEXT_FOR_ANSWER': cls.CONTEXT_FOR_ANSWER
        }
    
    