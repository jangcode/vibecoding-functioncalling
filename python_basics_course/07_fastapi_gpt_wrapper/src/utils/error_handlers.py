from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from pydantic import ValidationError

async def openai_exception_handler(request: Request, exc: HTTPException):
    """OpenAI API 관련 예외 처리기

    Args:
        request (Request): FastAPI 요청 객체
        exc (HTTPException): 발생한 예외

    Returns:
        JSONResponse: 에러 응답
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "openai_error"
            }
        }
    )

async def validation_exception_handler(request: Request, exc: ValidationError):
    """입력 데이터 검증 예외 처리기

    Args:
        request (Request): FastAPI 요청 객체
        exc (ValidationError): 발생한 예외

    Returns:
        JSONResponse: 에러 응답
    """
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": "입력값이 올바르지 않습니다",
                "details": exc.errors(),
                "type": "validation_error"
            }
        }
    )

async def generic_exception_handler(request: Request, exc: Exception):
    """일반 예외 처리기

    Args:
        request (Request): FastAPI 요청 객체
        exc (Exception): 발생한 예외

    Returns:
        JSONResponse: 에러 응답
    """
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "서버에서 오류가 발생했습니다",
                "detail": str(exc),
                "type": "server_error"
            }
        }
    )
