from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import logging

from .models.chat import ChatRequest, ChatResponse
from .services.gpt_service import GPTService
from .utils.error_handlers import (
    openai_exception_handler,
    validation_exception_handler,
    generic_exception_handler
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="GPT API Wrapper",
    description="OpenAI GPT API의 안전하고 효율적인 래퍼 서비스",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 구체적인 도메인 지정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 에러 핸들러 등록
app.add_exception_handler(HTTPException, openai_exception_handler)
app.add_exception_handler(ValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """채팅 API 엔드포인트

    Args:
        request (ChatRequest): 채팅 요청 데이터

    Returns:
        ChatResponse: GPT 응답과 사용량 정보

    Raises:
        HTTPException: API 호출 중 오류 발생 시
    """
    logger.info(f"Chat request received: {len(request.messages)} messages")
    
    try:
        response = await GPTService.generate_chat_response(request)
        logger.info("Chat response generated successfully")
        return response
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트

    Returns:
        dict: 서버 상태 정보
    """
    return {
        "status": "healthy",
        "service": "gpt-wrapper",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
