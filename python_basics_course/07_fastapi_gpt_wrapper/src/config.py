from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """애플리케이션 설정

    환경 변수에서 설정을 로드합니다.
    기본값이 있는 설정은 환경 변수가 없어도 동작합니다.
    """
    openai_api_key: str
    max_tokens: int = 1000
    temperature: float = 0.7
    model_name: str = "gpt-3.5-turbo"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    """설정 인스턴스를 캐싱하여 반환"""
    return Settings()
