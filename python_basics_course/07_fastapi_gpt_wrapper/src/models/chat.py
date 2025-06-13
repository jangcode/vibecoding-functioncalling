from pydantic import BaseModel, Field
from typing import List, Optional

class ChatMessage(BaseModel):
    """채팅 메시지 모델

    Attributes:
        role (str): 메시지 작성자의 역할 (user, system, assistant)
        content (str): 메시지 내용
    """
    role: str = Field(..., pattern="^(user|system|assistant)$")
    content: str = Field(..., min_length=1)

class ChatRequest(BaseModel):
    """채팅 요청 모델

    Attributes:
        messages (List[ChatMessage]): 채팅 메시지 목록
        temperature (float, optional): 응답의 다양성 조절 (0.0 ~ 2.0)
        max_tokens (int, optional): 최대 토큰 수
    """
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(1000, gt=0)

class ChatResponse(BaseModel):
    """채팅 응답 모델

    Attributes:
        response (str): GPT 응답 텍스트
        usage (dict): 토큰 사용량 정보
    """
    response: str
    usage: dict
