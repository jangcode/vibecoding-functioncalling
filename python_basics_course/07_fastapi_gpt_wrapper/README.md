# 모듈 7: FastAPI로 GPT 래핑 – 기초

## 학습 목표
- FastAPI 기반의 GPT API 래퍼 서비스를 구현할 수 있다
- 안전하고 효율적인 API 설계 원칙을 이해한다
- 에러 처리와 입력 검증을 구현할 수 있다

## 준비 사항

### 1. 필요 패키지 설치
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# 패키지 설치
pip install fastapi uvicorn python-dotenv openai pydantic python-multipart pytest httpx

# requirements.txt 생성
pip freeze > requirements.txt
```

### 2. 프로젝트 구조
```
.
├── README.md                  # 프로젝트 문서
├── requirements.txt           # 패키지 목록
├── .env                      # 환경 변수 설정
├── src/                      # 소스 코드
│   ├── __init__.py
│   ├── main.py              # FastAPI 앱
│   ├── config.py            # 설정
│   ├── models/              # 데이터 모델
│   │   ├── __init__.py
│   │   └── chat.py
│   ├── services/            # 비즈니스 로직
│   │   ├── __init__.py
│   │   └── gpt_service.py
│   └── utils/               # 유틸리티
│       ├── __init__.py
│       └── error_handlers.py
└── tests/                   # 테스트
    ├── __init__.py
    ├── test_main.py
    └── test_gpt_service.py
```

## 1. 프로젝트 설정

### 1.1 환경 변수 설정
```ini
# .env
OPENAI_API_KEY=your-api-key
MAX_TOKENS=1000
TEMPERATURE=0.7
MODEL_NAME=gpt-3.5-turbo
```

### 1.2 설정 관리
```python
# src/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    openai_api_key: str
    max_tokens: int = 1000
    temperature: float = 0.7
    model_name: str = "gpt-3.5-turbo"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
```

## 2. API 구현

### 2.1 데이터 모델
```python
# src/models/chat.py
from pydantic import BaseModel, Field
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|system|assistant)$")
    content: str = Field(..., min_length=1)

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(1000, gt=0)

class ChatResponse(BaseModel):
    response: str
    usage: dict
```

### 2.2 GPT 서비스
```python
# src/services/gpt_service.py
import openai
from ..config import get_settings
from ..models.chat import ChatRequest, ChatResponse

settings = get_settings()
openai.api_key = settings.openai_api_key

class GPTService:
    @staticmethod
    async def generate_chat_response(request: ChatRequest) -> ChatResponse:
        try:
            response = await openai.ChatCompletion.acreate(
                model=settings.model_name,
                messages=[msg.dict() for msg in request.messages],
                temperature=request.temperature or settings.temperature,
                max_tokens=request.max_tokens or settings.max_tokens
            )
            
            return ChatResponse(
                response=response.choices[0].message.content,
                usage=response.usage
            )
            
        except openai.error.OpenAIError as e:
            raise HTTPException(
                status_code=500,
                detail=f"OpenAI API 오류: {str(e)}"
            )
```

### 2.3 에러 핸들링
```python
# src/utils/error_handlers.py
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.requests import Request

async def openai_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "openai_error"
            }
        }
    )

async def validation_exception_handler(request: Request, exc: HTTPException):
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
```

### 2.4 메인 애플리케이션
```python
# src/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .models.chat import ChatRequest, ChatResponse
from .services.gpt_service import GPTService
from .utils.error_handlers import openai_exception_handler, validation_exception_handler

app = FastAPI(title="GPT API Wrapper")

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

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    return await GPTService.generate_chat_response(request)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 3. 테스트

### 3.1 서비스 테스트
```python
# tests/test_gpt_service.py
import pytest
from src.services.gpt_service import GPTService
from src.models.chat import ChatRequest, ChatMessage

@pytest.mark.asyncio
async def test_chat_generation():
    request = ChatRequest(
        messages=[
            ChatMessage(role="user", content="Hello!")
        ]
    )
    
    response = await GPTService.generate_chat_response(request)
    assert response.response
    assert response.usage
```

### 3.2 API 테스트
```python
# tests/test_main.py
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_chat_endpoint():
    response = client.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        }
    )
    assert response.status_code == 200
    assert "response" in response.json()
    assert "usage" in response.json()
```

## 실습 과제

### 1. 기본 채팅 API 구현
1. 프로젝트 구조 설정
2. 환경 변수 구성
3. `/chat` 엔드포인트 구현
4. 기본 테스트 작성

### 2. 에러 처리 및 검증 추가
1. 입력 검증 로직 구현
2. 에러 핸들러 추가
3. CORS 설정
4. 테스트 케이스 보강

### 3. 기능 확장
1. 컨텍스트 관리 추가
2. 레이트 리미팅 구현
3. 로깅 시스템 추가
4. 캐싱 구현

## 서버 실행
```bash
# 개발 모드
uvicorn src.main:app --reload

# 프로덕션 모드
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## API 문서
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 참고 자료
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [OpenAI API 문서](https://platform.openai.com/docs/api-reference)
- [Pydantic 문서](https://pydantic-docs.helpmanual.io/)
